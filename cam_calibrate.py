import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import os
import glob
import cv2

CALIBRATION_IMAGES_FOLDER = 'camera_cal'
PICKLE_NAME = 'wide_dist_pickle.p'

# Get 'objpoints' and 'imgpoints'
# display -> to diaplay found corners
# write -> to write corners drawn files to folder
# folder_name -> name of folder in which calibration files are kept
# files_name -> general name of files


def get_points(folder_name='camera_cal', files_name='calibration*.jpg', display=False, write=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(folder_name + '/' + files_name)

    print('Iterating through calibration images')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if display:
                cv2.drawChessboardCorners(img, (8, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            # Write drawn files to same folder
            if write:
                write_name = folder_name + '/' + 'corners_found' + '_' + str(idx) + '.jpg'
                cv2.imwrite(write_name, img)

    if display:
        cv2.destroyAllWindows()
    print('objpoints and imgpoints calculated')
    return (objpoints, imgpoints)

# Call this method from other classes to save the pickle


def get_calibration():
    # Get objpoints and imgpoints from above code
    objpoints, imgpoints = get_points(folder_name=CALIBRATION_IMAGES_FOLDER)

    # Read a sample image to get image size
    img = cv2.imread(CALIBRATION_IMAGES_FOLDER + '/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use
    # (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(CALIBRATION_IMAGES_FOLDER + '/' + PICKLE_NAME, 'wb'))

    print('Pickle of mtx and dist matrix dumped at : ' + CALIBRATION_IMAGES_FOLDER
          + '/' + PICKLE_NAME)

# Call this method to un-distort the sample images


def undistort_sample_images(visualise=False):
    # Get points
    objpoints, imgpoints = get_points(folder_name=CALIBRATION_IMAGES_FOLDER)
    # Get list of matching images
    images = glob.glob(CALIBRATION_IMAGES_FOLDER + '/' + 'calibration*.jpg')

    print('')
    print('Iterating to un-distort the images')
    for idx, img in enumerate(images):
        # Read a sample image to get image size
        img = cv2.imread(img)
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Undistort the image
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(CALIBRATION_IMAGES_FOLDER + '/calibrated_' + str(idx + 1) + '.jpg', dst)

        if visualise:
            # Visualize undistortion
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(dst)
            ax2.set_title('Undistorted Image', fontsize=30)
    print('Images Undistorted')

if __name__ == '__main__':
    undistort_sample_images(visualise=False)
