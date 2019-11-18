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
        self.data_gen_batch_size = settings.data_gen_batch_size
        self.batch_size = settings.batch_size
        self.epochs = settings.epochs
        self.decay = settings.decay
        self.body_part = settings.body_part

        self.train_path = 'dataset/train'
        self.valid_path = 'dataset/valid'

        self.train_dataset_file = 'dataset/' + 'train_' + self.body_part + '.csv'
        self.valid_dataset_file = 'dataset/' + 'valid_' + self.body_part + '.csv'

    def load_regression_data(self):
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

    def load_classification_data(self):
        self.classification_train_generator = ImageDataGenerator().flow_from_directory('dataset/train',
                                                                                    target_size = (224, 224),
                                                                                    batch_size = self.data_gen_batch_size)

        self.classification_valid_generator = ImageDataGenerator().flow_from_directory('dataset/valid',
                                                                                    target_size = (224, 224),
                                                                                    batch_size = self.data_gen_batch_size)

    def train_new_classification(self):
        self.model = helper_funcs.create_new_model(False, 7)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.train_classification_model()

    def train_classification(self):
        self.model = helper_funcs.load_model(self.body_part)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.train_classification_model()

    def train_new_regression(self):
        self.model = helper_funcs.create_new_model(True, 0)

        opt = Adam(learning_rate=1e-1, decay=self.decay)
        self.model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        self.train_regression_model()

    def train_regression(self):
        self.model = helper_funcs.load_model(self.body_part)

        opt = Adam(learning_rate=1e-1, decay=self.decay)
        self.model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        self.train_regression_model()

    def train_regression_model(self):

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


    def train_classification_model(self):

        if not self.TESTING:

            try:
                print("----------------- TRAINING -----------------")
                history = self.model.fit_generator(self.classification_train_generator,
                                    validation_data=self.classification_valid_generator,
                                    epochs=self.epochs,
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
            self.model.fit_generator(self.classification_train_generator,
                                    validation_data=self.classification_valid_generator,
                                    epochs=self.epochs,
                                    verbose=1,
                                    shuffle=True)

    def predict_abnormality(self):
        predictions = self.model.predict(self.test_images_x)

        diff = predictions.flatten() - self.test_y
        percentage_diff = (diff/ self.test_y) * 100
        abs_percentage = np.abs(percentage_diff)

        mean = np.mean(abs_percentage)

        print("Mean diff: {2f}%".format(mean))