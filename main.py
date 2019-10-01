import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

train_path = 'MURA-v1.1/train'
valid_path = 'MURA-v1.1/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='grayscale')
valid_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='grayscale')

imgs, labels = next(train_batches)

print(labels)