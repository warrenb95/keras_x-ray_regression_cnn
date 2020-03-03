import settings
import helper_funcs
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2

from datetime import datetime

import helper_funcs
import settings

class Regression_Trainer:
    class __Regression_Trainer:

        def __init__(self):
            self.trainer = None

            # Settings
            # ---------------------------------------------------------------------
            self.TESTING = settings.TESTING
            self.batch_size = settings.batch_size
            self.epochs = settings.epochs
            self.opt = settings.regression_opt
            # ---------------------------------------------------------------------

            # File variables
            # ---------------------------------------------------------------------
            self.train_path = 'dataset/train'
            self.valid_path = 'dataset/valid'
            self.train_images_x_total = 0
            self.test_images_x_total = 0
            self.validation_images_x_total = 0
            # ---------------------------------------------------------------------

        def load_regression_data(self, body_part):
            '''Load the regression data and images for the 'body_part'.

            Parameters
            ----------
            body_part: str
                The body part to load.

            Returns
            -------
            train_images_x: [] cv2 image
                A list of training images.

            train_y: double
                The target values of the train_images_x.

            test_images_x: [] cv2 image
                A list of training images.

            test_y: double
                The target values of the test_y.
            '''

            train_dataset_file = 'dataset/' + 'train_' + body_part + '.csv'

            df = helper_funcs.load_dataset_attributes(train_dataset_file)

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

            train_y = train_attribs_x['target']
            test_y = test_attribs_x['target']

            self.train_images_x_total = len(train_images_x)
            self.test_images_x_total = len(test_images_x)

            return (train_images_x, train_y, test_images_x, test_y)

        def train_new(self, body_part, amount_of_models):
            '''Create and train a new model for the 'body_part'.

            Parameters
            ---------
            body_part: str
                The body part model to create and train.

            Returns
            -------
            model: Sequential
                A keras model
            '''

            # model = helper_funcs.create_new_model(True, 0)
            for model_num in range(amount_of_models):
                self.model = helper_funcs.create_desnet121()
                self.model.compile(optimizer = self.opt, loss = 'mse')
                self.train_regression_model(body_part, model_num)

        def train_old(self, body_part):
            '''Load and train the 'body_part' model.

            Parameters
            ----------
            body_part: str
                The body part model to load and train.

            Returns
            -------
            model: Sequential
                A keras model
            '''

            self.model = helper_funcs.load_model(body_part)
            self.model.compile(optimizer = self.opt, loss = 'mse')
            self.train_regression_model(body_part)

        def train_regression_model(self, body_part, model_num):
            ''' Set up and train the model for the body_part

            Parameters
            ----------
            model: Sequential
                A keras model to train.

            body_part: str
                The body part model.

            Returns
            -------
            model: Sequential
                A keras model.
            '''

            train_images_x, train_y, test_images_x, test_y = self.load_regression_data(body_part)

            train_generator = ImageDataGenerator().flow(train_images_x, train_y, batch_size = self.batch_size)
            test_generator = ImageDataGenerator().flow(test_images_x, test_y, batch_size = self.batch_size)

            if not self.TESTING:

                try:
                    print("----------------- TRAINING -----------------")
                    history = self.model.fit_generator(train_generator,
                                        validation_data = test_generator,
                                        steps_per_epoch = int(self.train_images_x_total / self.batch_size),
                                        validation_steps = int(self.test_images_x_total / self.batch_size),
                                        epochs = self.epochs,
                                        verbose = 1,
                                        shuffle = True)

                    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    # Plot training & validation data
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('training data')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['loss', 'val_loss'], loc='upper left')

                    fname = "model_graphs/" + curr_datetime + '_' + body_part + str(model_num) +'.jpg'
                    plt.savefig(fname)
                    plt.close()
                    history = None

                    helper_funcs.save_model(self.model, body_part, model_num)

                except KeyboardInterrupt:
                    helper_funcs.save_model(self.model, body_part, model_num)
                else:
                    helper_funcs.save_model(self.model, body_part, model_num)
            else:
                print("----------------- self.TESTING -----------------")
                history = self.model.fit_generator(train_generator,
                                        validation_data = test_generator,
                                        steps_per_epoch = int(self.train_images_x_total / self.batch_size),
                                        validation_steps = int(self.test_images_x_total / self.batch_size),
                                        epochs = self.epochs,
                                        verbose = 1,
                                        shuffle = True)

                # Plot training & validation data
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('training data')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['loss', 'val_loss'], loc='upper left')

                fname = "model_graphs/self.TESTING_" + body_part + '.jpg'
                plt.savefig(fname)
                plt.close()
                history = None

        def predict(self, cur_image, body_part):
            '''Predict the abnormality of the image_path.

            Parameters
            ----------
            image_path: str
                The path of the image.

            body_part: str
                The body part model.

            Returns
            -------
            prediction: double
                The prediction produced by the model.
            '''

            self.model = helper_funcs.load_model(body_part)

            self.model.compile(optimizer = self.opt, loss = 'mse')

            prediction_y = self.model.predict(cur_image)

            prediction = prediction_y[0][0]

            prediction *= 100

            if prediction < 1:
                prediction = 0.0
            elif prediction > 100.0:
                prediction = 100.0

            prediction = round(prediction, 1)

            return prediction

    instance = None

    def getInstance(self):
        if not Regression_Trainer.instance:
            Regression_Trainer.instance = Regression_Trainer.__Regression_Trainer()

        return Regression_Trainer.instance
