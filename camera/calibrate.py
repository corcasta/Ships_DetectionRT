import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

def calibrate(pattern_size, visualize=False):
    """
    pattern_size: A tuple of integers representing the number of inner 
                  corners in chessboard.
                  Ex: (vertical corners, horizontal corners)
                  
    visualize: Boolean, displays chessboard patterns during calibration.
    """
    images = sorted(glob(os.getcwd() + "/pictures/*"))
    assert images
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            if visualize:
                img = cv2.drawChessboardCorners(img, pattern_size, corners,ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = cv2.imread(images[0])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    
    #Saves camera intrinsics and distortion coefficients
    mtx_df = pd.DataFrame(mtx)
    mtx_df.to_string("details/intrinsic_matrix.txt", index=False, header=False)

    #data = pd.DataFrame(newcameramtx)
    #data.to_string("newcam_intrinsic_matrix.txt", index=False, header=False)
    
    dist_df = pd.DataFrame(dist)
    dist_df.to_string("details/dist_coefficients.txt", index=False, header=False)
    

def main(args):
    if args.corners == None:
        print(("ERROR: Please provide number of vertical " 
               "and horizontal corners with -c argument: e.g. -c 7 7"))
        sys.exit()
    elif len(args.corners) == 2:
        pattern_size = tuple(args.corners)
        calibrate(pattern_size)
    else:
        print("ERROR: Invalid input provided. Please provide just 2 integers")
        sys.exit()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corners", 
                        help=("Provide only 2 integers representing the number of " 
                              "inner corners in chessboard (vertical corners, horizontal corners)"),
                        nargs='+', 
                        type=int)
    args = parser.parse_args()
    
    script_path = os.path.realpath(__file__)
    os.chdir(Path(script_path).parent)
    main(args)
    