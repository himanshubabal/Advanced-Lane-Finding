import matplotlib
matplotlib.use("TkAgg")

from cam_calibrate import CALIBRATION_IMAGES_FOLDER, PICKLE_NAME, get_calibration
import os
import pickle
import cv2
import matplotlib.pyplot as plt


def get_undistorted_image(img):
    # Check if pickle is already present
    # If not, prepare pickle using method
    if not (os.path.isfile(CALIBRATION_IMAGES_FOLDER + '/' + PICKLE_NAME)):
        print('Making pickle for mtx and dist values')
        get_calibration()

    # Get mtx and dist from pickle file
    with open(CALIBRATION_IMAGES_FOLDER + '/' + PICKLE_NAME, 'rb') as file:
        calib_dict = pickle.load(file)

    mtx, dist = calib_dict['mtx'], calib_dict['dist']

    # Get un-distorted image
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


if __name__ == '__main__':
    i = cv2.imread('test_images/straight_lines1.jpg')
    plt.imshow(i)
    # plt.show()

    dst = get_undistorted_image(i)
    plt.imshow(dst)
    plt.show()
