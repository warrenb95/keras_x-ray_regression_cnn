import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(path):
    cur_image = cv2.imread(path)
    # Find the edge bounadries
    low_x, high_x, low_y, high_y = find_boundary(cur_image)

    # plt.subplot(122)
    # plt.imshow(edges,cmap = 'gray')
    # plt.show()

    # Crop into the x-ray image
    cur_image = cur_image[low_y:high_y, low_x:high_x]
    # cv2.imshow("cropped image", cropped_img)
    # cv2.waitKey()
    cv2.imwrite(path, cur_image)

def find_boundary(image):
    # Find where the x-ray is within the image
    edges = cv2.Canny(cur_image, 0, 50)
    indices = np.where(edges != [0])

    # Find the edge bounadries
    low_x_ind = 0
    high_x_ind = 0
    for i in range(len(indices[1])):
        if indices[1][i] < indices[1][low_x_ind]:
            low_x_ind = i
        elif indices[1][i] > indices[1][high_x_ind]:
            high_x_ind = i

    low_x = indices[1][low_x_ind]
    high_x = indices[1][high_x_ind]

    low_y_ind = 0
    high_y_ind = 0
    for i in range(len(indices[0])):
        if indices[0][i] < indices[0][low_y_ind]:
            low_y_ind = i
        elif indices[0][i] > indices[0][high_y_ind]:
            high_y_ind = i

    low_y = indices[0][low_y_ind]
    high_y = indices[0][high_y_ind]

    print("X: {} -> {}    Y: {} -> {}".format(low_x, high_x, low_y, high_y))

    return low_x, high_x, low_y, high_y
