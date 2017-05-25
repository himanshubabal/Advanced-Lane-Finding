import matplotlib
matplotlib.use("TkAgg")

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

p1, p2, p3, p4 = [585, 450], [695, 450], [200, 720], [1120, 720]
d1, d2, d3, d4 = [300, 0], [1000, 0], [300, 720], [1000, 720]


def get_perspectiveTransform(img, plot=False):
    img_size = (img.shape[1], img.shape[0])
    # print(img_size)

    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([d1, d2, d3, d4])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    if plot:
        plt.figure(figsize=(30, 40))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('original')

        plt.subplot(1, 2, 2)
        plt.imshow(warped)
        plt.title('pers_trans')

        plt.show()

    return(warped, Minv)

if __name__ == '__main__':
    image = mpimg.imread('test_images/test2.jpg')

    # cv2.line(image, (585, 450), (695, 450), [255, 0, 0], 5)
    # cv2.line(image, (695, 450), (1120, 720), [255, 0, 0], 5)
    # cv2.line(image, (200, 720), (1120, 720), [255, 0, 0], 5)
    # cv2.line(image, (200, 720), (585, 450), [255, 0, 0], 5)

    p = get_perspectiveTransform(image)
    plt.imshow(image)
    plt.show()

    plt.imshow(p)
    plt.show()
    # 360, 635
    # 1028, 635

    # 520, 425
    # 650, 425
