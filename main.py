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
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import helper_funcs

TESTING = True
steps_per_epoch = 25
validation_steps = 5
epochs = 5
decay = 1e-1 / epochs

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='grayscale', batch_size=30, target_size=(224,244))
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, color_mode='grayscale', batch_size=30, target_size=(224,244))

callbacks = [LearningRateScheduler(helper_funcs.PolynomialDecay(maxEpochs = epochs, initAlpha=1e-1, power=5))]
opt = Adam(learning_rate=1e-1)

model = helper_funcs.create_new_model()
# model = helper_funcs.load_model()

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

if not TESTING:
    print("----------------- TRAINING -----------------")
    try:
        model.fit_generator(train_batches,
                            steps_per_epoch = steps_per_epoch,
                            callbacks = callbacks,
                            validation_data = valid_batches,
                            validation_steps = validation_steps,
                            epochs = epochs,
                            verbose = 1,
                            shuffle = True)
    except KeyboardInterrupt:
        helper_funcs.save_model(model, "class_model")
    else:
        helper_funcs.save_model(model, "class_model")
else:
    print("----------------- TESTING -----------------")
    model.fit_generator(train_batches,
                        steps_per_epoch = steps_per_epoch,
                        callbacks = callbacks,
                        validation_data = valid_batches,
                        validation_steps = validation_steps,
                        epochs = epochs,
                        verbose = 1,
                        shuffle = True)
