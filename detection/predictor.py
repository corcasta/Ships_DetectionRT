from pathlib import Path
from glob import glob
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import imutils

#Setting up working directory
if os.path.basename(os.getcwd()) != "Ships_DetectionRT":
    working_dir = str(Path(os.getcwd()).parent)
    
if os.path.exists(working_dir):
    os.chdir(working_dir)
    print("pwd: " + working_dir)
else:
    assert("Desired working directory doesn't exist")
    
    
    
#Loading trained weights
weights = sorted(glob(os.path.join(working_dir, 
                                   "training", 
                                   "runs", 
                                   "detect", 
                                   "train", 
                                   "weights", 
                                   "*.pt")))
best_weights = weights[0]
last_weights = weights[1]

#Loading Test dataset
#test_imgs = sorted(glob(os.path.join(working_dir, "SMDataset_YV8", "test", "images", "*")))

#Model Instance
model = YOLO(best_weights)

video = os.path.join(working_dir, "videos", "singapore_demo.mp4")

#model.predict() #Small bug, we need to call first model.predict before starting tracking
results = model.track(video, conf=0.40, show=True)
print(results)