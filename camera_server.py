# Modified from https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/
# import the necessary packages
from imutils import build_montages
from datetime import datetime
import time
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import torch
from model import SSD300
from detect import detect
from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to checkpoint of the model to be used")
ap.add_argument("-mW", "--montageW", required=True, type=int, help="montage frame width")  # Will build a grid with each camera output
ap.add_argument("-mH", "--montageH", required=True, type=int, help="montage frame height") # if just one device, use 1x1
args = ap.parse_args()

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

# load our serialized model from disk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.model)
start_epoch = checkpoint['epoch']
print(f'\nLoaded checkpoint {args.model} from epoch %d.\n' % start_epoch)
state_dict = checkpoint['model_state_dict']
model = SSD300(n_classes=4)  # Hard coded 3 classes + background
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

# Save each device's frame to display
frameDict = {}

# assign montage width and height so we can view all incoming frames in a single "dashboard"
mW = args.montageW
mH = args.montageH

# start looping over all the frames
while True:
	# receive RPi name and frame from the RPi and acknowledge the receipt
	start = time.time()
	(rpiName, frame) = imageHub.recv_image()
	imageHub.send_reply(b'OK')
	print(f"Time to receive {time.time() - start:.4f}", end="\t")
	# if a device is not in the frameD dictionary then it means that its a newly connected device
	if rpiName not in frameDict:
		print("New device: {}".format(rpiName))
	# Pass the frame throught the network
	start = time.time()
	frame = np.array(detect(Image.fromarray(frame[:,:,::-1]), min_score=0.2, max_overlap=0.5, top_k=200, model=model))[:,:,::-1]
	print(f"Time to detect  {time.time() - start:.4f}", end="\t")
	# update the new frame in the frame dictionary
	frameDict[rpiName] = frame
	# build a montage using images in the frame dictionary
	start = time.time()
	montages = build_montages(frameDict.values(), (300, 300), (mW, mH))
	# display the montage(s) on the screen
	for montage in montages:
		cv2.imshow(rpiName, montage)
	print(f"Time to montage {time.time() - start:.4f}")
	# detect any kepresses
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
