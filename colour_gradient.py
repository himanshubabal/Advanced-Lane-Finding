import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_gradients(img, s_thresh=(170, 255), sx_thresh=(20, 100), plot=False):
    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    if plot:
        plt.figure(figsize=(20, 25))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('original')

        plt.subplot(1, 3, 2)
        plt.imshow(color_binary)
        plt.title('color_binary')

        plt.subplot(1, 3, 3)
        plt.imshow(combined_binary)
        plt.title('combined_binary')

        plt.show()

    return (color_binary, combined_binary)


if __name__ == '__main__':
    image = mpimg.imread('test_images/test2.jpg')
    result, result2 = get_gradients(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    # ax1.imshow(image)
    # ax1.set_title('Original Image', fontsize=10)

    ax1.imshow(result)
    ax1.set_title('Pipeline Result', fontsize=10)

    ax2.imshow(result2)
    ax2.set_title('Pipeline Result2', fontsize=10)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
