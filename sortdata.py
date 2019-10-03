import csv
import os
import shutil
import typing

train_normal_folder = 'dataset/train/normal'
train_abnormal_folder = 'dataset/train/abnormal'
valid_normal_folder = 'dataset/valid/normal'
valid_abnormal_folder = 'dataset/valid/abnormal'

image_count = 0

def copy_files(src: str , dest: str):
    global image_count
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest + '/' + str(image_count) + '.png')
            image_count += 1

training_file_name = 'MURA-v1.1/train_labeled_studies.csv'
valid_file_name = 'MURA-v1.1/valid_labeled_studies.csv'

with open(training_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if row[1] == '0':
            copy_files(row[0], train_normal_folder)
        else:
            copy_files(row[0], train_abnormal_folder)

        print('Training: ', row_count)
        row_count += 1

with open(valid_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if row[1] == '0':
            copy_files(row[0], valid_normal_folder)
        else:
            copy_files(row[0], valid_abnormal_folder)

        print('Valid: ', row_count)
        row_count += 1