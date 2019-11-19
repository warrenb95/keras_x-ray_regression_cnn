import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import mean_absolute_percentage_error, categorical_crossentropy
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
        self.body_part = settings.body_part

        self.train_path = 'dataset/train'
        self.valid_path = 'dataset/valid'

        self.train_dataset_file = 'dataset/' + 'train_' + self.body_part + '.csv'
        self.valid_dataset_file = 'dataset/' + 'valid_' + self.body_part + '.csv'

        self.train_images_x_total = 0
        self.test_images_x_total = 0

    def load_regression_data(self):
        df = helper_funcs.load_dataset_attributes(self.train_dataset_file)

        images = helper_funcs.load_images(df)

        split = train_test_split(df,
                                images,
                                test_size=0.25,
                                random_state=2,
                                shuffle=True)

        (train_attribs_x,
        test_attribs_x,
        train_images_x,
        test_images_x) = split

        train_images_x = np.array(train_images_x)
        test_images_x = np.array(test_images_x)

        train_y = np.array(train_attribs_x['target'])
        test_y = np.array(test_attribs_x['target'])

        self.train_images_x_total = len(train_images_x)
        self.test_images_x_total = len(test_images_x)

        regression_train_generator = ImageDataGenerator().flow(train_images_x, train_y, batch_size=self.data_gen_batch_size)
        regression_test_generator = ImageDataGenerator().flow(test_images_x, test_y, batch_size=self.data_gen_batch_size)

        return (regression_train_generator, regression_test_generator)


    def load_classification_data(self):
        classification_train_generator = ImageDataGenerator().flow_from_directory('dataset/train',
                                                                                    target_size = (224, 224),
                                                                                    batch_size = self.data_gen_batch_size)

        classification_valid_generator = ImageDataGenerator().flow_from_directory('dataset/valid',
                                                                                    target_size = (224, 224),
                                                                                    batch_size = self.data_gen_batch_size)

        return (classification_train_generator, classification_valid_generator)

    def train_new_classification(self, classification_train_generator, classification_valid_generator):
        model = helper_funcs.create_new_model(False, 7)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.train_classification_model(model, classification_train_generator, classification_valid_generator)

    def train_classification(self, classification_train_generator, classification_valid_generator):
        model = helper_funcs.load_model(self.body_part)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.train_classification_model(model, classification_train_generator, classification_valid_generator)

    def train_new_regression(self, regression_train_generator, regression_test_generator):
        model = helper_funcs.create_new_model(True, 0)

        opt = Adam(learning_rate=1e-1)
        model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        return self.train_regression_model(model, regression_train_generator, regression_test_generator)

    def train_regression(self, regression_train_generator, regression_test_generator):
        model = helper_funcs.load_model(self.body_part)

        opt = Adam(learning_rate=1e-1)
        model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

        return self.train_regression_model(model, regression_train_generator, regression_test_generator)

    def train_regression_model(self, model, regression_train_generator, regression_test_generator):

        if not self.TESTING:

            try:
                print("----------------- TRAINING -----------------")
                # history = self.model.fit(np.array(self.train_images_x),
                #                     np.array(self.train_y),
                #                     validation_data=(np.array(self.test_images_x), np.array(self.test_y)),
                #                     epochs=self.epochs,
                #                     batch_size=self.batch_size,
                #                     verbose=1,
                #                     shuffle=True)

                # curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # # Plot training & validation data
                # plt.plot(history.history['loss'])
                # plt.plot(history.history['val_loss'])
                # plt.title('training data')
                # plt.ylabel('Loss')
                # plt.xlabel('Epoch')
                # plt.legend(['loss', 'val_loss'], loc='upper left')

                # fname = "model_graphs/" + curr_datetime + '_' + self.body_part + '.jpg'
                # plt.savefig(fname)

            except KeyboardInterrupt:
                helper_funcs.save_model(model, self.body_part)
            else:
                helper_funcs.save_model(model, self.body_part)
        else:
            print("----------------- TESTING -----------------")

            model.fit_generator(regression_train_generator,
                                    validation_data=regression_test_generator,
                                    steps_per_epoch=round(self.train_images_x_total / self.batch_size),
                                    epochs=self.epochs,
                                    validation_steps=round(self.test_images_x_total / self.batch_size),
                                    verbose=1,
                                    shuffle=True)

        return model


    def train_classification_model(self, model, classification_train_generator, classification_valid_generator):

        if not self.TESTING:

            try:
                print("----------------- TRAINING -----------------")
                history = model.fit_generator(classification_train_generator,
                                    validation_data=classification_valid_generator,
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
                helper_funcs.save_model(model, self.body_part)
            else:
                helper_funcs.save_model(model, self.body_part)
        else:
            print("----------------- TESTING -----------------")
            model.fit_generator(classification_train_generator,
                                    validation_data=classification_valid_generator,
                                    epochs=self.epochs,
                                    verbose=1,
                                    shuffle=True)

        return model

    def predict_abnormality(self, model):

        df = helper_funcs.load_dataset_attributes(self.valid_dataset_file)

        valid_images_x = np.array(helper_funcs.load_images(df))

        valid_y = np.array(df['target'])

        predict_y = model.predict_generator(ImageDataGenerator().flow(valid_images_x, valid_y, batch_size=self.batch_size))

        # diff = predict_y.flatten() - valid_y
        # percentage_diff = (diff/ valid_y) * 100
        # abs_percentage = np.abs(percentage_diff)
        # mean = np.mean(abs_percentage)

        for i in range(len(predict_y)):
            print("Actual: {}, Prediction: {}".format(valid_y[i], predict_y[i]))
