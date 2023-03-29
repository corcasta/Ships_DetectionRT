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


class Stalker():
    def __init__(self, camera_resolution=(1920, 1080)):
        self.cam_width = camera_resolution[0]
        self.cam_height = camera_resolution[1]
        
        #The following two attributes come from the calibration script
        self.cam_intrinsics = pd.read_csv("cam_intrinsics.txt", delim_whitespace=True, header=None).to_numpy()
        self.newcam_intrinsics = pd.read_csv("newcam_intrinsics.txt", delim_whitespace=True, header=None).to_numpy()
        self.dist_coefficients = pd.read_csv("dist_coefficients.txt", delim_whitespace=True, header=None).to_numpy()
        
    

    def _denormalize(self, points):
        """
        Normalization Formulas:    
            Normalized(Xcentre) = (x_centre)/Image_Width
            Normalized(Ycentre) = (y_centre)/Image_Height
            Normalized(w) = w/Image_Width
            Normalized(h) = h/Image_Height
        """
        
        #DENORMALIZE
        # x-center-bb
        points[:, 0] = points[:, 0]*self.cam_width 
        # y-center-bb
        points[:, 1] = points[:, 1]*self.cam_height
        # width-bb 
        points[:, 2] = points[:, 2]*self.cam_width 
        # height-bb
        points[:, 3] = points[:, 3]*self.cam_height 
        
        return points
    
    def _shift(self, points):
        #Shift the point of interest from center to floor
        #of bounding box
        #                   y           height     weight
        points[:, 1] = points[:, 1] + points[:, 3]*(0.75)
        return points
    

    def _undistort(self, points, k, distortion, iter_num=5):
        #print("distortion: {}".format(distortion))
        k1, k2, p1, p2, k3 = np.squeeze(distortion)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[:2, 2]
        x = points[:, 0]
        y = points[:, 1]    
        
        
        #x, y = points#.astype(float)
        x = (x - cx) / fx
        x0 = x
        y = (y - cy) / fy
        y0 = y
        for _ in range(iter_num):
            r2 = x ** 2 + y ** 2
            k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
            delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
            delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
            x = (x0 - delta_x) * k_inv
            y = (y0 - delta_y) * k_inv
        return np.array((x * fx + cx, y * fy + cy))


    def localize(self, points):
        """_summary_

        Args:
            points (_type_): raw pixel points from yolo model

        Returns:
            _type_: _description_
        """
        if points.numpy().size == 0:
            return torch.transpose(points[:, 0:2], 0, 1).numpy()

        points = self._denormalize(points) 
        points = self._shift(points)
        points = points.numpy()[:,0:2]
        
        # These points will be rearrange to be:
        #   row0: x
        #   row1: y
        #   columns: represents individual points 
        # After calling _undistort     
        points = self._undistort(points, self.newcam_intrinsics, self.dist_coefficients)
        print(points)
        #points = torch.transpose(points[:, 0:2], 0, 1).numpy()
  
        #Drone pose relative to World frame:
        phi   = 0
        theta = 0
        psi   = 0
        Od_x  = 0
        Od_y  = 0
        Od_z  = 0.23
        
        rot_x = np.asarray([[1,           0,            0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi),  np.cos(phi)]])
        
        rot_y = np.asarray([[np.cos(theta),  0, np.sin(theta)],
                            [0,              1,             0],
                            [-np.sin(theta), 0, np.cos(theta)]])
        
        rot_z = np.asarray([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi),  np.cos(psi), 0],
                            [0,                      0, 1]])
        
        rot_wd = rot_x @ rot_y @ rot_z
        t_wd = np.vstack((rot_wd, [0,0,0]))
        t_wd = np.hstack((t_wd, [[Od_x], [Od_y], [Od_z], [1]]))
        
        
        #Camera pose relative to Drone frame:
        #For the moment we are just assuming that the camera is fixed
        #to the drone's movement. Later we will need to provide the gimball
        #orientation to have the correct matrix for the camera relative to 
        #the drone.
        t_dc = np.asarray([[1,  0, 0, 0],
                           [0,  0, 1, 0],
                           [0, -1, 0, 0],
                           [0,  0, 0, 1]])
        
        #Camera pose relative to World frame:
        t_wc = t_wd@t_dc
        
        
        #Camera Intrinsics
        #cam_intrinsics = pd.read_csv("cam_intrinsics.txt", delim_whitespace=True, header=None) 
        #cam_intrinsics = cam_intrinsics.to_numpy()
        
        #Localize point in World Coords
        #Transform Point to a Homogeneous Point
        points = np.vstack((points, np.ones((1, len(points[0,:])))))
        
        A_cpx = np.linalg.inv(self.newcam_intrinsics) @ points 
        R_wc = t_wc[0:3, 0:3]
        O_wc = t_wc[0:3, 3][:, np.newaxis]
        Z = np.asarray([[0],[0],[1]])

        #Point K with respect to World Frame
        A_wk = O_wc + (-O_wc[2,:]/(np.dot(R_wc, A_cpx)[2,:]))*np.dot(R_wc, A_cpx)
        return A_wk



def main():
    model = YOLO("yolov8m.pt")
    stalker = Stalker()
    device = 4
    
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

    
    """
    frame_width, frame_height = 1920, 1080    
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    #Camera Intrinsics
    cam_intrinsics = pd.read_csv("cam_intrinsics.txt", delim_whitespace=True, header=None).to_numpy()

    #Distortion Coefficients
    dist_coefficients = pd.read_csv("dist_coefficients.txt", delim_whitespace=True, header=None).to_numpy()

        
    while True:
        ret, frame = cap.read()        
        frame = cv2.undistort(frame, cam_intrinsics, dist_coefficients, None, cam_intrinsics)

        result = model.track(frame, classes=[49, 32], conf=0.3, tracker="bytetrack.yaml")[0]
        detections = sv.Detections.from_yolov8(result)
        
        points = result.boxes.xywhn
        points_world = stalker.localize(points)
        #points = denormalize(points, camera_resolution=(1920, 1080))
        #points = shift(points)
        #x_coords = points[:, 0]
        #y_coords = points[:, 1]
        #points = torch.transpose(points[:, 0:2], 0, 1).numpy()
        #points_world = localize(points)
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
        cv2.imshow("yolov8", imutils.resize(frame, width=640))
        if (cv2.waitKey(20) == ord("q")):
            break
        
        
    """
    for result in model.track(source="4", classes=[49, 32], conf=0.3, show=False, tracker="customtrack.yaml", stream=True):   
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        points = result.boxes.xywhn
        points_world = stalker.localize(points)
        
        #points = denormalize(result.boxes.xywhn, camera_resolution=(1920, 1080))
        #points = shift(points)
        #x_coords = points[:, 0]
        #y_coords = points[:, 1]
        #points = torch.transpose(points[:, 0:2], 0, 1).numpy()
        #points_world = localize(points)
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
        cv2.imshow("yolov8", imutils.resize(frame, width=640))
        
        if (cv2.waitKey(20) == ord("q")):
            break
    
        
    
    
if __name__ == "__main__":
    main()				
    
    
    