import csv
import os
import shutil
import typing
import pre_process

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

training_set = 0
valid_set = 1

image_count = 0

def copy_files(src: str , dest: str, value: int, model_type: str):
    global image_count
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        new_file_name = dest + '/' + str(image_count) + '.png'
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, new_file_name)
            pre_process.process_image(new_file_name)
            image_count += 1

            add_to_dataset_file(new_file_name, value, model_type)
        else:
            print("Copy failed :\(")


def add_to_dataset_file(path: str, value: int, model_type: str):
    model_dataset_path = 'dataset/' + model_type + '.csv'
    with open(model_dataset_path, 'a+') as dataset_file:
        dataset_file.write(path + "," + str(value) + '\n')


training_file_name = 'MURA-v1.1/train_labeled_studies.csv'
valid_file_name = 'MURA-v1.1/valid_labeled_studies.csv'

'''
This is to be used to sort the data for the regression models.
'''
with open(training_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if 'ELBOW' in row[0]:
            # copy_files(row[0], train_elbow_folder)
            if row[1] == '1':
                copy_files(row[0], train_elbow_folder, 100, 'train_elbow')
            else:
                copy_files(row[0], train_elbow_folder, 0, 'train_elbow')
        elif 'FINGER' in row[0]:
            # copy_files(row[0], train_finger_folder)
            if row[1] == '1':
                copy_files(row[0], train_finger_folder, 100, 'train_finger')
            else:
                copy_files(row[0], train_finger_folder, 0, 'train_finger')
        elif 'FOREARM' in row[0]:
            # copy_files(row[0], train_forearm_folder)
            if row[1] == '1':
                copy_files(row[0], train_forearm_folder, 100, 'train_forearm')
            else:
                copy_files(row[0], train_forearm_folder, 0, 'train_forearm')
        elif 'HAND' in row[0]:
            # copy_files(row[0], train_hand_folder)
            if row[1] == '1':
                copy_files(row[0], train_hand_folder, 100, 'train_hand')
            else:
                copy_files(row[0], train_hand_folder, 0, 'train_hand')
        elif 'HUMERUS' in row[0]:
            # copy_files(row[0], train_humerus_folder)
            if row[1] == '1':
                copy_files(row[0], train_humerus_folder, 100, 'train_humerus')
            else:
                copy_files(row[0], train_humerus_folder, 0, 'train_humerus')
        elif 'SHOULDER' in row[0]:
            # copy_files(row[0], train_shoulder_folder)
            if row[1] == '1':
                copy_files(row[0], train_shoulder_folder, 100, 'train_shoulder')
            else:
                copy_files(row[0], train_shoulder_folder, 0, 'train_shoulder')
        elif 'WRIST' in row[0]:
            # copy_files(row[0], train_wrist_folder)
            if row[1] == '1':
                copy_files(row[0], train_wrist_folder, 100, 'train_wrist')
            else:
                copy_files(row[0], train_wrist_folder, 0, 'train_wrist')
        else:
            print("Oops")

        print('Training: ', row_count)
        row_count += 1

with open(valid_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if 'ELBOW' in row[0]:
            # copy_files(row[0], valid_elbow_folder)
            if row[1] == '1':
                copy_files(row[0], valid_elbow_folder, 100, 'valid_elbow')
            else:
                copy_files(row[0], valid_elbow_folder, 0, 'valid_elbow')
        elif 'FINGER' in row[0]:
            # copy_files(row[0], valid_finger_folder)
            if row[1] == '1':
                copy_files(row[0], valid_finger_folder, 100, 'valid_finger')
            else:
                copy_files(row[0], valid_finger_folder, 0, 'valid_finger')
        elif 'FOREARM' in row[0]:
            # copy_files(row[0], valid_forearm_folder)
            if row[1] == '1':
                copy_files(row[0], valid_forearm_folder, 100, 'valid_forearm')
            else:
                copy_files(row[0], valid_forearm_folder, 0, 'valid_forearm')
        elif 'HAND' in row[0]:
            # copy_files(row[0], valid_hand_folder)
            if row[1] == '1':
                copy_files(row[0], valid_hand_folder, 100, 'valid_hand')
            else:
                copy_files(row[0], valid_hand_folder, 0, 'valid_hand')
        elif 'HUMERUS' in row[0]:
            # copy_files(row[0], valid_humerus_folder)
            if row[1] == '1':
                copy_files(row[0], valid_humerus_folder, 100, 'valid_humerus')
            else:
                copy_files(row[0], valid_humerus_folder, 0, 'valid_humerus')
        elif 'SHOULDER' in row[0]:
            # copy_files(row[0], valid_shoulder_folder)
            if row[1] == '1':
                copy_files(row[0], valid_shoulder_folder, 100, 'valid_shoulder')
            else:
                copy_files(row[0], valid_shoulder_folder, 0, 'valid_shoulder')
        elif 'WRIST' in row[0]:
            # copy_files(row[0], valid_wrist_folder)
            if row[1] == '1':
                copy_files(row[0], valid_wrist_folder, 100, 'valid_wrist')
            else:
                copy_files(row[0], valid_wrist_folder, 0, 'valid_wrist')

        print('Valid: ', row_count)
        row_count += 1

------------------------------------------------------------------------------------------------
