import matplotlib
matplotlib.use("TkAgg")

from undistort_image import get_undistorted_image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from PIL import Image

# from moviepy.editor import VideoFileClip

from perspective_trans import get_perspectiveTransform
from colour_gradient import get_gradients


from lane_lines import get_lane_lines_window, get_lane_lines_using_prev_info, get_lane_line_pixels, get_curvature, draw


def pipeline(image, save_img=False):
    undist = get_undistorted_image(image)
    grad, grad_ = get_gradients(undist)
    binary_warped, Minv = get_perspectiveTransform(grad)

    if save_img:
        im = Image.fromarray(undist)
        im.save('test_images/undistorted_img.jpg')

        im = Image.fromarray(grad)
        im.save('test_images/grad1.jpg')

        im = Image.fromarray(grad_)
        im.save('test_images/grad2.jpg')

        im = Image.fromarray(binary_warped)
        im.save('test_images/bin_wrapped.jpg')

    ploty, left_fitx, right_fitx, left_fit, right_fit, window_img = get_lane_lines_window(binary_warped)
    left_fit, right_fit, ploty, prev_img = get_lane_lines_using_prev_info(binary_warped, left_fitx, right_fitx, left_fit, right_fit)
    # pixel_img = get_lane_line_pixels(ploty, left_fitx, right_fitx)
    l, r, m = get_curvature(image, ploty, left_fitx, right_fitx)
    final_img = draw(ploty, left_fitx, right_fitx, binary_warped, Minv, undist, image)

    cv2.putText(final_img, 'Left Curvature  : ' + str(l)[:6] + ' m',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    cv2.putText(final_img, 'Right Curvature : ' + str(r)[:6] + ' m',
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    cv2.putText(final_img, 'Off Center      : ' + str(m)[:6] + ' m',
                (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    return final_img

if __name__ == '__main__':
    image = mpimg.imread('test_images/vlc_42.jpg')
    f = pipeline(image, True)

    im = Image.fromarray(f)
    im.save('test_images/vlc_42_out.jpg')

    # plt.imshow(f)
    # plt.show()
