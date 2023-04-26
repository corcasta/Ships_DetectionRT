import os
import cv2
import time
import torch
import imutils
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt

class Stalker():
    def __init__(self, camera_resolution=(1920, 1080)):
        self.cam_width = camera_resolution[0]
        self.cam_height = camera_resolution[1]
        
        #The following two attributes come from the calibration script
        #Path(os.getcwd()).parent
        self.intrinsic_matrix = pd.read_csv(str(os.getcwd()) 
                                            + "/cam_calibration/details/intrinsic_matrix.txt", 
                                            delim_whitespace=True, 
                                            header=None).to_numpy()
        
        #self.intrinsic_newmatrix = pd.read_csv(str(os.getcwd()) 
        #                                     + "/cam_calibration/details/newcam_intrinsics.txt", 
        #                                     delim_whitespace=True, 
        #                                     header=None).to_numpy()
        
        self.dist_coefficients = pd.read_csv(str(os.getcwd()) 
                                             + "/cam_calibration/details/dist_coefficients.txt", 
                                             delim_whitespace=True, 
                                             header=None).to_numpy()
        
    
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
    
    
    def _cart2pol(self, x, y):
        """
        x:  numpy array composed of x coords for each detected object
        y:  numpy array composed of y coords for each detected object
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return np.asarray([rho, phi])
    
    
    def _pol2cart(self, rho, phi):
        """
        rho:  numpy array composed of rho for each detected object
        phi:  numpy array composed of phi for each detected object
        """
        
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.asarray([x, y])

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
        points = self._undistort(points, self.intrinsic_matrix, self.dist_coefficients)
        print(points)
        #points = torch.transpose(points[:, 0:2], 0, 1).numpy()
  
        #Drone pose relative to World frame:
        phi   = 0
        theta = 0
        psi   = 0
        Od_x  = 0
        Od_y  = 0
        Od_z  = 0.235
        
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
                
        #Localize point in World Coords
        #Transform Point to a Homogeneous Point
        points = np.vstack((points, np.ones((1, len(points[0,:])))))
        
        A_ckpx = np.linalg.inv(self.intrinsic_matrix) @ points 
        R_wc = t_wc[0:3, 0:3]
        O_wc = t_wc[0:3, 3][:, np.newaxis]
        Z = np.asarray([[0],[0],[1]])

        #Point K with respect to World Frame
        A_wk = O_wc + (-O_wc[2,:]/(np.dot(R_wc, A_ckpx)[2,:]))*np.dot(R_wc, A_ckpx)
        
        print("Before: Cart-Pixel_Coords: \n{}".format(A_wk))
        
        #Point K with respect to World Frame in Polar coords
        A_wk_polar = self._cart2pol(A_wk[0,:], A_wk[1,:])
        print("During: Polar-Pixel_Coords: \n{}".format(A_wk_polar))
        
        A_wk_cart = self._pol2cart(A_wk_polar[0,:], A_wk_polar[1,:])
        print("After: Cart-Pixel_Coords: \n{}".format(A_wk_cart ))
        return A_wk_polar


def main():
    model = YOLO(str(os.getcwd()) + "/model_training/yolov8n.pt")
    stalker = Stalker()
    device = 0
    
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

    for result in model.track(source=device, 
                              classes=[49, 32], 
                              conf=0.3, 
                              show=False, 
                              tracker=str(Path(os.getcwd()).parent) + "/customtrack.yaml", 
                              stream=True):   
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        points = result.boxes.xywhn
        polar_points_world = stalker.localize(points)
        
        #points = denormalize(result.boxes.xywhn, camera_resolution=(1920, 1080))
        #points = shift(points)
        #x_coords = points[:, 0]
        #y_coords = points[:, 1]
        #points = torch.transpose(points[:, 0:2], 0, 1).numpy()
        #points_world = localize(points)
        print("Pixel_Coords: \n{} ".format(polar_points_world))

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
    # Setting the starting path of the script
    script_path = os.path.realpath(__file__)
    os.chdir(Path(script_path).parent)
    print(os.getcwd())
    main()				
    
    
    