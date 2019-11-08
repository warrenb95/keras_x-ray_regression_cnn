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

TESTING = settings.TESTING
batch_size = settings.batch_size
epochs = settings.epochs
decay = settings.decay
body_part = settings.body_part

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_dataset_file = 'dataset/' + 'train_' + body_part + '.csv'
valid_dataset_file = 'dataset/' + 'valid_' + body_part + '.csv'

df = helper_funcs.load_dataset_attributes(train_dataset_file)

images = helper_funcs.load_images(df)

split = train_test_split(df, images, test_size=0.25, random_state=2, shuffle=True)
(train_attribs_x, test_attribs_x, train_images_x, test_images_x) = split

train_y = train_attribs_x['target']
test_y = test_attribs_x['target']

opt = Adam(learning_rate=1e-1, decay=decay)

model = helper_funcs.create_new_model()
# model = helper_funcs.load_model(body_part)

model.compile(optimizer = opt, loss = 'mean_absolute_percentage_error')

if not TESTING:

    try:
        print("----------------- TRAINING -----------------")
        history = model.fit(np.array(train_images_x),
                            np.array(train_y),
                            validation_data=(np.array(test_images_x), np.array(test_y)),
                            epochs=epochs,
                            batch_size=batch_size,
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

        fname = "model_graphs/" + curr_datetime + '_' + body_part + '.jpg'
        plt.savefig(fname)

    except KeyboardInterrupt:
        helper_funcs.save_model(model, body_part)
    else:
        helper_funcs.save_model(model, body_part)
else:
    print("----------------- TESTING -----------------")
    history = model.fit(np.array(train_images_x),
                            np.array(train_y),
                            validation_data=(np.array(test_images_x), np.array(test_y)),
                            epochs=epochs,
                            verbose=1,
                            shuffle=True)
