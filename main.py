import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import helper_funcs

TESTING = False
steps_per_epoch = 250
validation_steps = 25
epochs = 50
decay = 1e-1 / epochs

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='rgba', batch_size=30)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, color_mode='rgba', batch_size=30)

callbacks = [LearningRateScheduler(helper_funcs.PolynomialDecay(maxEpochs = epochs, initAlpha=1e-1, power=5))]
opt = SGD(lr=1e-1, momentum=0.9, decay=decay)

model = helper_funcs.create_new_model()
# model = helper_funcs.load_model()

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

if not TESTING:
    print("----------------- TRAINING -----------------")
    try:
        model.fit_generator(train_batches, steps_per_epoch = steps_per_epoch, callbacks = callbacks, validation_data = valid_batches, validation_steps = validation_steps, epochs = epochs, verbose = 1)
    except KeyboardInterrupt:
        helper_funcs.save_model(model)
    else:
        helper_funcs.save_model(model)
else:
    print("----------------- TESTING -----------------")
    model.fit_generator(train_batches, steps_per_epoch = steps_per_epoch, callbacks = callbacks, validation_data = valid_batches, validation_steps = validation_steps, epochs = epochs, verbose = 1)
