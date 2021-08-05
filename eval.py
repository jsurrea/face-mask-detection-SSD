import argparse
from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from model import SSD300

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './data'

def evaluate(model, data_loader):
    """
    Evaluate.

    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(data_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    ## TODO it could be helpful to return the metrics


if __name__ == '__main__':

    exp = "."

    # Load evaluation data
    split='test'  # 'val' or 'test' 
    keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
    batch_size = 128
    workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dataset = PascalVOCDataset(data_folder,
                                    split=split,
                                    keep_difficult=keep_difficult)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=eval_dataset.collate_fn, num_workers=workers, pin_memory=True)

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(f"{exp}/model_final.pth")
    start_epoch = checkpoint['epoch']
    print(f'\nLoaded checkpoint {exp}/model_final.pth from epoch %d.\n' % start_epoch)
    state_dict = checkpoint['model_state_dict']
    model = SSD300(n_classes=4)  # Hard coded for 3 classes + background
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    evaluate(model, eval_loader)
