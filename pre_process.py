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
    # cv2.imshow('cropped image', cropped_img)
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

    # print('X: {} -> {}    Y: {} -> {}'.format(low_x, high_x, low_y, high_y))

    return low_x, high_x, low_y, high_y

def crop_image(path):
    cur_image = cv2.imread(path)
    cur_image = cv2.resize(cur_image, (112, 112))
    cv2.imwrite(path, cur_image)

folder_sources =['dataset/train/elbow',
            'dataset/train/finger',
            'dataset/train/forearm',
            'dataset/train/hand',
            'dataset/train/humerus',
            'dataset/train/shoulder',
            'dataset/train/wrist',
            'dataset/valid/elbow',
            'dataset/valid/finger',
            'dataset/valid/forearm',
            'dataset/valid/hand',
            'dataset/valid/humerus',
            'dataset/valid/shoulder',
            'dataset/valid/wrist']

# for src in folder_sources:
#     src_files = os.listdir(src)
#     for file_name in src_files:
#         full_file_name = os.path.join(src, file_name)
#         if os.path.isfile(full_file_name):
#             try:
#                 # process_image(full_file_name)
#                 crop_image(full_file_name)
#             except:
#                 print('Cannot process - {}'.format(full_file_name))

csv_sources =['dataset/train_elbow.csv',
            'dataset/train_finger.csv',
            'dataset/train_forearm.csv',
            'dataset/train_hand.csv',
            'dataset/train_humerus.csv',
            'dataset/train_shoulder.csv',
            'dataset/train_wrist.csv',
            'dataset/valid_elbow.csv',
            'dataset/valid_finger.csv',
            'dataset/valid_forearm.csv',
            'dataset/valid_hand.csv',
            'dataset/valid_humerus.csv',
            'dataset/valid_shoulder.csv',
            'dataset/valid_wrist.csv']

def remove_images():
    for source in csv_sources:
        print(f'Processing... {source}')
        new_csv_lines = []
        with open(source, 'r') as csv_file:
            for line in csv_file:
                copy = True

                file_path = line.split(',')[0]

                # If file does not exist
                if not os.path.isfile(file_path):
                    print(f'Removing... {file_path}')
                    copy = False

                if copy:
                    new_csv_lines.append(line)

        with open(source, 'w') as csv_file:
            csv_file.writelines(new_csv_lines)

def find_balance():
    balance_dict = {}
    for source in csv_sources:
        normal_count = 0
        abnormal_count = 0
        count = 0
        print(f'Processing {source}...')
        with open(source, 'r') as csv_file:
            for line in csv_file.readlines():
                classification = line.split(',')[1].strip()
                if classification == '1.0':
                    abnormal_count += 1
                elif classification == '0':
                    normal_count += 1

        balance_dict[source] = [normal_count, abnormal_count]

    return balance_dict

def balance_dataset():
    balance_dict = find_balance()

    for key, value in balance_dict.items():
        normal_count = 0
        abnormal_count = 0
        min_count = min(value)
        print(f'Processing... {key}')
        new_csv_lines = []
        with open(key, 'r') as csv_file:
            for line in csv_file:
                classification = line.split(',')[1].strip()
                if classification == '1.0':
                    if abnormal_count <= min_count:
                        new_csv_lines.append(line)
                        abnormal_count += 1
                else:
                    if normal_count <= min_count:
                        new_csv_lines.append(line)
                        normal_count += 1

        with open(key, 'w') as csv_file:
            csv_file.writelines(new_csv_lines)
