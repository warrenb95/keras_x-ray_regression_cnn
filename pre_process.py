import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def process_image(path):
    cur_image = cv2.imread(path)
    # Find the edge bounadries
    low_x, high_x, low_y, high_y = find_boundary(cur_image)

    if low_x == None:
        return

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
    edges = cv2.Canny(image, 0, 50)
    indices = np.where(edges != [0])

    if len(indices[0]) == 0 or len(indices[1]) == 0 or len(indices) == 0:
        return

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

    # print("X: {} -> {}    Y: {} -> {}".format(low_x, high_x, low_y, high_y))

    return low_x, high_x, low_y, high_y


train_elbow_folder = 'dataset/train/elbow'
train_finger_folder = 'dataset/train/finger'
train_forearm_folder = 'dataset/train/forearm'
train_hand_folder = 'dataset/train/hand'
train_humerus_folder = 'dataset/train/humerus'
train_shoulder_folder = 'dataset/train/shoulder'
train_wrist_folder = 'dataset/train/wrist'

valid_elbow_folder = 'dataset/valid/elbow'
valid_finger_folder = 'dataset/valid/finger'
valid_forearm_folder = 'dataset/valid/forearm'
valid_hand_folder = 'dataset/valid/hand'
valid_humerus_folder = 'dataset/valid/humerus'
valid_shoulder_folder = 'dataset/valid/shoulder'
valid_wrist_folder = 'dataset/valid/wrist'

sources =[train_shoulder_folder,
            train_wrist_folder,
            valid_elbow_folder,
            valid_finger_folder,
            valid_forearm_folder,
            valid_hand_folder,
            valid_humerus_folder,
            valid_shoulder_folder,
            valid_wrist_folder]

for src in sources:
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            try:
                process_image(full_file_name)
            except:
                print("Cannot process - {}".format(full_file_name))
