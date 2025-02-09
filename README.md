# Real-time face-mask detection with Single Shot Detector (SSD)

This paper proposes the Neural Network SSD architecture for the task of face mask detection. We trained this model to detect 3 different categories of people: those wearing a mask, those wearing it incorrectly and those not wearing it. For this proposal, a dataset with 853 images was used with annotations of these 3 classes. For the Neural Network, the weights were initialized from a pretrained model on the image dataset Pascal VOC and fine-tuned to finally achieve a mAP of 70.2% on the validation set. The mAP achieved on the test set was 66.7% which shows that despite the lack of a huge dataset, the architecture SSD is suitable for this task. Additionally, our experiments showed that the MultiBox Loss Function parameters can be modified to improve the performance on small and unbalanced datasets such as the one presented.Current work has focused on the task of classification or detection for a single class. Our work is a novel approach as it performs real-time detection and classification in 3 classes, making it suitable for natural scenarios such as public spaces.

Check our paper [here](https://ieeexplore.ieee.org/abstract/document/9626095)

![Real_time_f_m_d_SSD paper pdf-image-002](https://github.com/user-attachments/assets/86e510dd-f5e0-4e1a-bf44-add7e7f90c92)

## Configuration

### Create `./data` folder
This step might not be necessary because this folder will be tracked by git. I leave this snnipet as a reference
1. Create symbolic link to the dataset in PascalVOC format. The script will look for root at `../data/face-mask-detection`
2. Run `python create_data_lists.py` inside the repo

### Other files
Check you have the following files:
- `calibril.ttf`: Used to render text in PIL images (used in `detect.py`)
- `ssd300_PascalVOC.pth`: State dict from the tutorial's pretrained model on PascalVOC

### Initialize a screen
This will save all the outputs of the screen to a log file
1. Make sure you have a `logs/` dir
2. Choose a `exp_name` (or any name you wish for the log file). Make sure it doesn't already exist.
3. Run `screen -S mask-SSD -L -Logfile logs/exp_name.txt`
4. Activate environment: `conda activate pytorch`
5. Go to repo `cd a-PyTorch-Tutorial-to-Object-Detection`

### Run scripts:

#### Train
The models obtained during training will be saved in a folder with the same `exp_name`. Models will be saved with the syntax `model_{epoch}.pth`, where `epoch="final"` for the last epoch.
1. Define a new `exp_name` or continue an interrupted experiment (it must not contain a `model_final.pth` file) 
2. Change the parameters with which you want to experiment. The available options for learning are:
	- `lr`: Learning rate
	- `decay_lr_at`: Each time that training reaches one of these epochs, learning rate will decay
	- `decay_lr_to`: Learning rate will decay to this fraction of the existing learning rate
	- `momentum`: Momentum
	- `weight_decay`: Weight decay
	- `grad_clip`: Wheter to clip gradients in the range `(-grad_clip, grad_clip)` to avoid going to `inf`
3. Change other running parameters if necessary. Options include:
	- `print_freq`: How many batches per epoch to print training losses
	- `save_freq`: How many epochs to save the model and evaluate on validation
	- `batch_size`: How many images per batch in training and evaluating
	- `epochs`: Total number of epochs to train
	- `workers`: Number of workers for loading data
	- `split`: Should only be changed at the end from `'val'` to `'test'`
4. If there was a directory named `exp_name/`, the script will look for the last model. To use a specific checkpoint, set `checkpoint = "model_name.pth"` where appropiate
5. Specify GPUs and cores. For example: `CUDA_VISIBLE_DEVICES=0 taskset -c 0-7`
6. Run `python train.py --exp exp_name` with your experiment name
7. The losses for training and validation will be appended to the file `losses.txt`

#### Evaluate
The main function from this script is used in the final epoch of training. However, it can still be used with a saved model
1. Specify in `filename` the path to the model state dict
2. Optionally, adjust the parameters `batch_size`, `workers` and `split`
3. Run `python eval.py`

#### Visualize
This script creates a new folder inside your `exp_name/` directory which will contain all the validation images with their predictions
1. Specify in `filename` the path to the model state dict
2. Like before, you can adjust parameters like `batch_size`, `workers` and `split`
3. Run `python detect.py`

