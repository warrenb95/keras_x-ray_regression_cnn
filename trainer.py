import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import mean_absolute_percentage_error
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

import helper_funcs
import settings

class Trainer:

    def __init__(self):
        self.TESTING = settings.TESTING
        self.batch_size = settings.batch_size
        self.epochs = settings.epochs
        self.decay = settings.decay
        self.body_part = settings.body_part

        self.train_path = 'dataset/train'
        self.valid_path = 'dataset/valid'

        self.train_dataset_file = 'dataset/' + 'train_' + self.body_part + '.csv'
        self.valid_dataset_file = 'dataset/' + 'valid_' + self.body_part + '.csv'

    def load_data(self):
        df = helper_funcs.load_dataset_attributes(self.train_dataset_file)

        images = helper_funcs.load_images(df)

        split = train_test_split(df,
                                images,
                                test_size=0.25,
                                random_state=2,
                                shuffle=True)

        (self.train_attribs_x,
        self.test_attribs_x,
        self.train_images_x,
        self.test_images_x) = split

        self.train_y = self.train_attribs_x['target']
        self.test_y = self.test_attribs_x['target']

    def train_new_classification(self):
        self.model = helper_funcs.create_new_model(False, 7)

        self.train()

    def train_classification(self):
        self.model = helper_funcs.load_model(self.body_part)

        self.train()

    def train_new_regression(self):
        self.model = helper_funcs.create_new_model(True, 0)

        opt = Adam(learning_rate=1e-1, decay=self.decay)
        self.model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        self.train()

    def train_regression(self):
        self.model = helper_funcs.load_model(self.body_part)

        opt = Adam(learning_rate=1e-1, decay=self.decay)
        self.model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        self.train()

    def train(self):

        if not self.TESTING:

            try:
                print("----------------- TRAINING -----------------")
                history = self.model.fit(np.array(self.train_images_x),
                                    np.array(self.train_y),
                                    validation_data=(np.array(self.test_images_x), np.array(self.test_y)),
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    shuffle=True)

                curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Plot training & validation data
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('training data')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['loss', 'val_loss'], loc='upper left')

                fname = "model_graphs/" + curr_datetime + '_' + self.body_part + '.jpg'
                plt.savefig(fname)

            except KeyboardInterrupt:
                helper_funcs.save_model(self.model, self.body_part)
            else:
                helper_funcs.save_model(self.model, self.body_part)
        else:
            print("----------------- TESTING -----------------")
            self.model.fit(np.array(self.train_images_x),
                                    np.array(self.train_y),
                                    validation_data=(np.array(self.test_images_x), np.array(self.test_y)),
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    shuffle=True)
