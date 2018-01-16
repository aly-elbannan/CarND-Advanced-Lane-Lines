import numpy as np
import cv2
import os
import glob

CHESSBOARD_CORNERS_WIDTH = 9
CHESSBOARD_CORNERS_HEIGHT = 6
CAMERA_CALIBRATION_PATH = '../camera_cal'
CAMERA_CALIBRATION_CHESSBOARD_OUTPUT_PATH = CAMERA_CALIBRATION_PATH + '/chessboard_corners'
CAMERA_CALIBRATION_UNDISTORTED_OUTPUT_PATH = CAMERA_CALIBRATION_PATH + '/undistorted'
CAMERA_CALIBRATION_MATRIX_OUTPUT_FILE = CAMERA_CALIBRATION_PATH + "/cal_matrix.p"

# Prepare output directories
for output_dir in [CAMERA_CALIBRATION_CHESSBOARD_OUTPUT_PATH, CAMERA_CALIBRATION_UNDISTORTED_OUTPUT_PATH]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_CORNERS_HEIGHT*CHESSBOARD_CORNERS_WIDTH,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_CORNERS_WIDTH, 0:CHESSBOARD_CORNERS_HEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the calib_images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration calib_images
calib_images = glob.glob(CAMERA_CALIBRATION_PATH + '/*.jpg')

# Step through the list and search for chessboard corners
for filename in calib_images:
    calib_test_img = cv2.imread(filename)
    gray = cv2.cvtColor(calib_test_img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_WIDTH,CHESSBOARD_CORNERS_HEIGHT), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw corners and write them to an output file
        cv2.drawChessboardCorners(calib_test_img, (CHESSBOARD_CORNERS_WIDTH,CHESSBOARD_CORNERS_HEIGHT), corners, ret)
        write_filename = CAMERA_CALIBRATION_CHESSBOARD_OUTPUT_PATH + "/" + os.path.basename(filename)
        cv2.imwrite(write_filename, calib_test_img)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Step through calibration images to undistort them for testing
for filename in calib_images:
    calib_test_img = cv2.imread(filename)
    calib_test_undistorted_img = cv2.undistort(calib_test_img, mtx, dist, None, mtx)
    write_filename = CAMERA_CALIBRATION_UNDISTORTED_OUTPUT_PATH + "/" + os.path.basename(filename)
    cv2.imwrite(write_filename,calib_test_undistorted_img)

import pickle
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open(CAMERA_CALIBRATION_MATRIX_OUTPUT_FILE, "wb" ) )
