import settings
import helper_funcs
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import stats

from datetime import datetime
import time
import concurrent.futures


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

        def load_evaluation_data(self, body_part):
            '''Load the regression data and images for the 'body_part'.

            Parameters
            ----------
            body_part: str
                The body part to load.

            Returns
            -------
            eval_images_x: [] cv2 image
                A list of training images.

            eval_y: double
                The target values of the train_images_x.
            '''

            valid_dataset_file = 'dataset/' + 'valid_' + body_part + '.csv'

            df = helper_funcs.load_dataset_attributes(valid_dataset_file)

            images = helper_funcs.load_images(df)

            eval_images = np.array(images)

            eval_y = df['target']

            self.eval_images_x_total = len(eval_images)

            return (eval_images, eval_y)

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

        def evaluate_model(self, body_part, model_num):
            ''' Set up and train the model for the body_part

            Parameters
            ----------
            model: Sequential
                A keras model to train.

            body_part: str
                The body part model.
            '''

            self.model = helper_funcs.load_model(body_part + '-' + str(model_num))
            self.model.compile(optimizer = self.opt, loss = 'mse')

            eval_images, eval_y = self.load_evaluation_data(body_part)

            eval_generator = ImageDataGenerator().flow(eval_images, eval_y, batch_size = self.batch_size)

            try:
                print("----------------- Evaluating -----------------")
                return self.model.evaluate(eval_generator, verbose = 1)
            except:
                print('Unabale to evaluate model ' + body_part + '-' + model_num)

        def get_prediction(self, body_part, cur_image):
            graph = tf.Graph()

            with graph.as_default():
                session = tf.compat.v1.Session(graph=graph)
                with session.as_default():
                    prediction_model = helper_funcs.load_model(body_part)
                    prediction_model.compile(optimizer = self.opt, loss = 'mse')
                    prediction_y = prediction_model.predict(cur_image)

            prediction = prediction_y[0][0]

            prediction *= 100

            if prediction < 1:
                prediction = 0.0
            elif prediction > 100.0:
                prediction = 100.0

            return prediction

        def predict(self, cur_image, body_part, model_num):
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
            prediction_list = []

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor() as executor:

                results = [executor.submit(self.get_prediction, body_part + '-' + str(i), cur_image) for i in range(model_num)]

                for f in concurrent.futures.as_completed(results):
                    prediction_list.append(f.result())

            # mid = 1
            # if model_num > 1:
            #     mid = model_num // 2
            # min_average = sorted(prediction_list)[:mid]

            # average_prediction = sum(prediction_list) / model_num
            # average_prediction = sum(min_average) / model_num

            end_time = time.time()
            duration = end_time - start_time
            print(f'Time to predict (Seconds) {duration}')

            prediction_list.sort()
            print(prediction_list)

            below_50 = []
            above_50 = []

            for prediction in prediction_list:
                if prediction < 50.0:
                    below_50.append(prediction)
                else:
                    above_50.append(prediction)

            if len(below_50) > len(above_50):
                prediction_list = below_50
            elif len(below_50) < len(above_50):
                prediction_list = above_50
            else:
                # Otherwise trim and use average
                print('below_50 and above_50 have same length')
                return round(stats.trim_mean(prediction_list, 0.25), 2)

            group_counter = dict()

            for i in range(len(prediction_list)):
                for j in range(len(prediction_list)):
                    if i == j:
                        continue

                    if prediction_list[j] - prediction_list[i] <= 10:
                        if prediction_list[i] in group_counter.keys():
                            group_counter[prediction_list[i]] += 1
                        else:
                            group_counter[prediction_list[i]] = 1

            max_count = max(group_counter.values())
            max_count_list = []
            for key, val in group_counter.items():
                if val == max_count:
                    max_count_list.append(key)
            
            print(max_count_list)

            return round(sum(max_count_list) / len(max_count_list), 2)


    instance = None

    def getInstance(self):
        if not Regression_Trainer.instance:
            Regression_Trainer.instance = Regression_Trainer.__Regression_Trainer()

        return Regression_Trainer.instance
