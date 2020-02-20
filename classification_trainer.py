import settings
import helper_funcs
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from matplotlib import pyplot as plt
import cv2
import typing

class Classification_Trainer:

    def __init__(self):
        self.model = None
        self.trainer = None

        # Settings.
        # ---------------------------------------------------------------------
        self.TESTING = settings.TESTING
        self.batch_size = settings.batch_size
        self.epochs = settings.epochs
        self.opt = settings.classification_opt
        # ---------------------------------------------------------------------

        # File variables.
        # ---------------------------------------------------------------------
        self.train_path = 'dataset/train'
        self.valid_path = 'dataset/valid'
        self.classification_train_total = 36808
        self.classification_test_total = 3197
        # ---------------------------------------------------------------------

    def getInstance(self):
        if self.trainer is None:
            self.trainer = Classification_Trainer()

        return self.trainer

    def train_new(self):
        '''Creates and train a new classification model.
        '''

        self.model = helper_funcs.create_new_model(False, 7)
        self.model.compile(loss='categorical_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

        train_classification_model()

    def train_old(self):
        '''Load in the classification model and train it.
        '''

        self.model = helper_funcs.load_model('class')
        self.model.compile(loss='categorical_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

        train_classification_model()

    def train_classification_model(self):
        '''Set up and and train the model arg passed in.

        Parameters
        ----------
        model: Sequential
            A keras model to train
        '''

        classification_train_generator = ImageDataGenerator().flow_from_directory(self.train_path,
                                                                                    target_size = (112, 112),
                                                                                    batch_size = self.batch_size)

        classification_valid_generator = ImageDataGenerator().flow_from_directory(self.valid_path,
                                                                                    target_size = (112, 112),
                                                                                    batch_size = self.batch_size)

        if not self.TESTING:

            try:
                print("----------------- TRAINING -----------------")
                history = self.model.fit_generator(classification_train_generator,
                                    validation_data = classification_valid_generator,
                                    steps_per_epoch = int(self.classification_train_total / self.batch_size),
                                    validation_steps = int(self.classification_test_total / self.batch_size),
                                    epochs = self.epochs,
                                    verbose = 1,
                                    shuffle = True)

                curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Plot training & validation data.
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('training data')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')

                fname = "model_graphs/" + curr_datetime + '_class.jpg'
                plt.savefig(fname)

                helper_funcs.save_model(model, 'class')

            except KeyboardInterrupt:
                # Handle a keyboard interupts 'ctrl+c'
                helper_funcs.save_model(model, 'class')
        else:
            print("----------------- self.TESTING -----------------")
            self.model.fit_generator(classification_train_generator,
                                steps_per_epoch = int(self.classification_train_total / self.batch_size),
                                validation_data = classification_valid_generator,
                                validation_steps = int(self.classification_test_total / self.batch_size),
                                epochs = self.epochs,
                                verbose = 1,
                                shuffle = True)

    def predict_classification(self, image_path):
        '''Predict image classification.

        Parameters
        ----------
        model: Sequential
            Keras model

        image_path: str
            The path of the image

        Returns
        -------
        prediction_y: int
            The prediction of the model
        '''

        if self.model is None:
            self.model = helper_funcs.load_model('class')

        self.model.compile(loss = 'categorical_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

        cur_image = helper_funcs.load_single_image(image_path)

        prediction_y = self.model.predict_classes(cur_image)

        return prediction_y, cur_image
