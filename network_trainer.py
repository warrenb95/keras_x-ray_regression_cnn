import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt
import pandas

from datetime import datetime

import helper_funcs
import settings

TESTING = settings.TESTING
steps_per_epoch = settings.steps_per_epoch
validation_steps = settings.validation_steps
epochs = settings.epochs
decay = settings.decay
body_part = settings.body_part

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_dataset_file = 'dataset/' + 'train_' + body_part + '.csv'
valid_dataset_file = 'dataset/' + 'valid_' + body_part + '.csv'

train_df = pandas.read_csv(train_dataset_file, delimiter = ',', header=None, names=['path', 'target'])
valid_df = pandas.read_csv(valid_dataset_file, delimiter = ',', header=None, names=['path', 'target'])

# train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='grayscale', batch_size=30, target_size=(112,112))
# valid_batches = ImageDataGenerator().flow_from_directory(valid_path, color_mode='grayscale', batch_size=30, target_size=(112,112))

train_batches = ImageDataGenerator().flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='path',
    y_col='target',
    color_mode='grayscale',
    target_size=(112, 112),
    batch_size=30,
    class_mode='raw'
)

valid_batches = ImageDataGenerator().flow_from_dataframe(
    dataframe=valid_df,
    directory=None,
    x_col="path",
    y_col="target",
    color_mode='grayscale',
    target_size=(112, 112),
    batch_size=30,
    class_mode='raw'
)

opt = Adam(learning_rate=1e-3, decay=decay)

model = helper_funcs.create_new_model()
# model = helper_funcs.load_model(body_part)

model.compile(optimizer = opt, loss = 'mean_squared_error')

if not TESTING:
    print("----------------- TRAINING -----------------")
    try:
        history = model.fit_generator(train_batches,
                            steps_per_epoch = steps_per_epoch,
                            validation_data = valid_batches,
                            validation_steps = validation_steps,
                            epochs = epochs,
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

        fname = "model_graphs/" + curr_datetime + '_' + body_part + '.jpg'
        plt.savefig(fname)

    except KeyboardInterrupt:
        helper_funcs.save_model(model, body_part)
    else:
        helper_funcs.save_model(model, body_part)
else:
    print("----------------- TESTING -----------------")
    history = model.fit_generator(train_batches,
                        steps_per_epoch = steps_per_epoch,
                        validation_data = valid_batches,
                        validation_steps = validation_steps,
                        epochs = epochs,
                        verbose = 1,
                        shuffle = True)

