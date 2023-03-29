from pathlib import Path
from glob import glob
import os
import pandas as pd
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import imutils
import supervision as sv
import time
import torch
import numpy as np



def denormalize(points, camera_resolution=(1920, 1080)):
    """
    Normalization Formula,

    Normalized(Xcentre) = (x_centre)/Image_Width

    Normalized(Ycentre) = (y_centre)/Image_Height

    Normalized(w) = w/Image_Width

    Normalized(h) = h/Image_Height
    """
    
    #DENORMALIZE
    points[:, 0] = points[:, 0]*camera_resolution[0] # x-center
    points[:, 1] = points[:, 1]*camera_resolution[1] # y-center
    points[:, 2] = points[:, 2]*camera_resolution[0] # width-bb
    points[:, 3] = points[:, 3]*camera_resolution[1] # height-bb
    
    return points
    
def shift(points):
    #Shift the point of interest from center to floor
    #of bounding box
    points[:, 1] = points[:, 1] + points[:, 3]/2
    
    return points
    

def localize(points):
    #Drone pose relative to World frame:
    phi   = 0
    theta = 0
    psi   = 0
    Od_x  = 0
    Od_y  = 0
    Od_z  = 0.945
    
    rot_x = [[1,           0,            0],
             [0, np.cos(phi), -np.sin(phi)],
             [0, np.sin(phi),  np.cos(phi)]]
    rot_x = np.asarray(rot_x)
    
    rot_y = [[np.cos(theta),  0, np.sin(theta)],
             [0,              1,             0],
             [-np.sin(theta), 0, np.cos(theta)]]
    rot_y = np.asarray(rot_y)
    
    rot_z = [[np.cos(psi), -np.sin(psi), 0],
             [np.sin(psi),  np.cos(psi), 0],
             [0,                      0, 1]]
    rot_z = np.asarray(rot_z)
    
    rot_wd = rot_x @ rot_y @ rot_z
    t_wd = np.vstack((rot_wd, [0,0,0]))
    t_wd = np.hstack((t_wd, [[Od_x], [Od_y], [Od_z], [1]]))
    
    
    #Camera pose relative to Drone frame:
    #For the moment we are just assuming that the camera is fixed
    #to the drone's movement. Later we will need to provide the gimball
    #orientation to have the correct matrix for the camera relative to 
    #the drone.
    t_dc = [[1,  0, 0, 0],
            [0,  0, 1, 0],
            [0, -1, 0, 0],
            [0,  0, 0, 1]]
    t_dc = np.asarray(t_dc)
    
    #Camera pose relative to World frame:
    t_wc = t_wd@t_dc
    
    
    #Camera Intrinsics
    cam_intrinsics = pd.read_csv("cam_intrinsics.txt", delim_whitespace=True, header=None) 
    cam_intrinsics = cam_intrinsics.to_numpy()
    
    #Localize point in World Coords
    points = np.vstack((points, np.ones((1, len(points[0,:])))))
    #print(points)
    A_cpx = np.linalg.inv(cam_intrinsics) @ points 
    #print(A_cpx)
    R_wc = t_wc[0:3, 0:3]
    O_wc = t_wc[0:3, 3][:, np.newaxis]
    Z = np.asarray([[0],[0],[1]])
    #print((-O_wc[2,:]/(np.dot(R_wc, A_cpx)[2,:])))
    #print(np.dot(R_wc, A_cpx))
    #print((-O_wc[2,:]/(np.dot(R_wc, A_cpx)[2,:]))*np.dot(R_wc, A_cpx))
    #print(O_wc)
    A_wk = O_wc + (-O_wc[2,:]/(np.dot(R_wc, A_cpx)[2,:]))*np.dot(R_wc, A_cpx)
    
    return A_wk



def main():
    model = YOLO("yolov8m.pt")
    #*********************FPS-DISPLAY-SETTINGS***********************
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    #*********************FPS-DISPLAY-SETTINGS***********************
    
    box_annotator = sv.BoxAnnotator(
            thickness=1,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5
        )


    frame_width, frame_height = 1920, 1080
    
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    while True:
        
        
        
    for result in model.track(source="3", classes=[49, 32], show=False, stream=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        
        points = denormalize(result.boxes.xywhn, camera_resolution=(1920, 1080))
        points = shift(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        points = torch.transpose(points[:, 0:2], 0, 1).numpy()
        points_world = localize(points)
        print("Pixel_Coords: {} ".format(points_world))

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int) 
            
        labels = [
            f"#:{tracker_id}, {model.model.names[class_id]}: {confidence:0.2f}"
            for _, confidence, class_id, tracker_id 
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # time when we finish processing for this frame
        new_frame_time = time.time()
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(frame, "FPS: {}".format(fps), (7, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)


        cv2.imshow("Ships-Detection-YoloV8n", frame)


        if (cv2.waitKey(20) == ord("q")):
            break
        
if __name__ == "__main__":
    main()				
    
    
    