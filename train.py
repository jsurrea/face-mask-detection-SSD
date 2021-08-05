import argparse
import time
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from eval import evaluate
from copy import deepcopy

# Data parameters
data_folder = './data'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_state_dict = "ssd300_PascalVOC.pth"  # State dict path from a pretrained model 
cudnn.benchmark = True

# Lists to save losses
epochs_list = []
train_losses = []
val_losses = []

## EXPERIMENT PARAMETERS (modify here)
exp_name = "."

# Learning parameters
lr = 0.01  # learning rate
decay_lr_at = []  # decay learning rate after these many epochs
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 0.0005  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

batch_size = 42  # batch size
epochs = 300  # number of epochs to train
workers = 4  # number of workers for loading data in the DataLoader

alpha = 1
neg_pos_ratio = 2
weight = [1., 2., 4., 3.]  # ("background", with_mask", "mask_weared_incorrect", "without_mask")

opt = "SGD"  # 'SGD' or 'Adam'
split = 'val'  # Change to test for final results

## END OF EXPERIMENT PARAMETERS

checkpoint = {}
os.makedirs(exp_name, exist_ok=True)
filename = os.path.join(exp_name, f"model_final.pth")
if os.path.exists(filename):
    checkpoint = torch.load(filename)
    print(f'\nLoaded checkpoint {filename} from epoch %d.\n' % (checkpoint['epoch'] + 1))

# Save parameters to file
def save_config(**kwargs):
    with open(os.path.join(exp_name, "config.txt"), "a") as f:
        # Note that this will append info to previous config file in the exp_name folder
        # This is in order to prevent lossing config info when running exp with different values
        print("Saving config at:", os.path.join(exp_name, "config.txt"))
        f.write("-"*20 + f" CONFIG FILE FOR {time.time()} " + "-"*20 + "\n")
        for k,v in kwargs.items():
            f.write(str(k) + "==" + str(v) + "\n")
        f.write("-"*20 + f" END OF CONFIG FILE FOR EXP {exp_name} " + "-"*20 + "\n")


save_config(lr=lr, decay_lr_at=decay_lr_at, decay_lr_to=decay_lr_to, momentum=momentum, weight_decay=weight_decay, grad_clip=grad_clip,
    batch_size=batch_size, epochs=epochs, workers=workers, alpha=alpha, neg_pos_ratio=neg_pos_ratio, weight=weight, split=split,
    optim=opt)  # Add other custom params if necessary

# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
model = SSD300(n_classes=n_classes)
model = model.to(device)
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)

if opt == "Adam":
    optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}])
elif opt == "SGD":
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], 
        lr=lr, momentum=momentum, weight_decay=weight_decay)



def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at, model, optimizer

    # Initialize model or load checkpoint
    if len(checkpoint) == 0:
        start_epoch = 0
        # Load pretrained weights
        if base_state_dict is not None:
            print("Loaded state dict from:", base_state_dict)
            loaded_state_dict = torch.load(base_state_dict)
            loaded_state_dict = {k:v for k,v in loaded_state_dict.items() if not k.startswith("pred_convs.cl_")}  # Drop class params
            model.load_state_dict(loaded_state_dict, strict=False)

    else:
        start_epoch = checkpoint['epoch'] + 1
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Move to default device
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, neg_pos_ratio=neg_pos_ratio, alpha=alpha, weight=weight).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Load evaluation data
    eval_dataset = PascalVOCDataset(data_folder,
                                    split=split,
                                    keep_difficult=keep_difficult)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=eval_dataset.collate_fn, num_workers=workers, pin_memory=True)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // 32)
    # decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    if len(checkpoint):
        best_val_loss = checkpoint['best_val_loss']
        print("Found prev best_val_loss:", best_val_loss)
    else:
        best_val_loss = float('inf')

    # Epochs
    for epoch in range(start_epoch, epochs + 1):

        # Prevent logging when no val has been evaluated
        val_loss = ""

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train_loss, val_loss = train(train_loader=train_loader,
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            eval_loader=eval_loader)

        if val_loss < best_val_loss and epoch > 40:

            best_val_loss = val_loss

            # Save checkpoint
            checkpoint = {'epoch': epoch,
                    'model_state_dict': deepcopy(model.state_dict()),
                    'optimizer': deepcopy(optimizer.state_dict()),
                    'best_val_loss': best_val_loss}
            print("Checkpoint!")

        # Save 
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


def train(train_loader, model, criterion, optimizer, epoch, eval_loader):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

    # Print status
    val_loss, eval_time = eval_loss(model, eval_loader, criterion)
    print('Epoch: [{0}]\t'
        'Train loss {loss.avg:.4f}\t'
        'Eval  loss {val_loss:.4f}\t'
        'Batch Time {batch_time.sum:.3f}\t'
        'Data  Time {data_time.sum:.3f}\t'
        'Eval  Time {eval_time:.3f}\t'.format(epoch,
                                            batch_time=batch_time,
                                            data_time=data_time, loss=losses, val_loss=val_loss, eval_time=eval_time))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

    return losses.avg, val_loss


def eval_loss(model, eval_loader, criterion):
    """
    Returns loss on evaluation set
    """
    model.eval()
    losses = AverageMeter()  # loss
    start = time.time()

    with torch.no_grad():
        for i, (images, boxes, labels, _) in enumerate(eval_loader):

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            losses.update(loss.item(), images.size(0))
    
    return losses.avg, time.time() - start


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        save_checkpoint(checkpoint, filename)
        print(f'\nSaved checkpoint {filename} from epoch %d.\n' % checkpoint['epoch'])
        if len(epochs_list):
            exists = os.path.exists(os.path.join(exp_name, "losses.txt"))
            with open(os.path.join(exp_name, "losses.txt"), "a") as f:
                if not exists:
                    f.write("Epoch,Train loss,Val loss\n")
                for i in range(len(epochs_list)):
                    f.write(str(epochs_list[i]) + "," + str(train_losses[i]) + "," + str(val_losses[i]) + "\n")


