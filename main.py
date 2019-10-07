import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import helper_funcs

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='rgba', batch_size=30)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, color_mode='rgba', batch_size=30)

class_model = helper_funcs.create_new_model()
# class_model = helper_funcs.load_model()

class_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    class_model.fit_generator(train_batches, steps_per_epoch=1227, validation_data=valid_batches, validation_steps=106, epochs=5, verbose=1)
except KeyboardInterrupt:
    helper_funcs.save_model(class_model, "class_model")
else:
    helper_funcs.save_model(class_model, "class_model")

