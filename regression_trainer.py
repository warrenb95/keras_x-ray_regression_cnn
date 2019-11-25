import settings
import helper_funcs
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

import helper_funcs
import settings

TESTING = settings.TESTING
batch_size = settings.batch_size
epochs = settings.epochs
body_part = settings.body_part
opt = settings.opt

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_dataset_file = 'dataset/' + 'train_' + body_part + '.csv'
valid_dataset_file = 'dataset/' + 'valid_' + body_part + '.csv'

train_images_x_total = 0
test_images_x_total = 0

def load_regression_data():
    global train_images_x_total
    global test_images_x_total

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

    train_y = np.array(train_attribs_x['target'])
    test_y = np.array(test_attribs_x['target'])

    train_images_x_total = len(train_images_x)
    test_images_x_total = len(test_images_x)

    return (train_images_x, train_y, test_images_x, test_y)

def train_new_regression():

    model = helper_funcs.create_new_model(True, 0)

    # model.compile(optimizer = opt, loss = 'msle')
    model.compile(optimizer = opt, loss = 'mse')

    return train_regression_model(model)

def train_old_regression():

    model = helper_funcs.load_model(body_part)

    # model.compile(optimizer = opt, loss = 'msle')
    model.compile(optimizer = opt, loss = 'mse')

    return train_regression_model(model)

def train_regression_model(model):

    train_images_x, train_y, test_images_x, test_y = load_regression_data()

    train_generator = ImageDataGenerator().flow(train_images_x, train_y, batch_size=batch_size)
    test_generator = ImageDataGenerator().flow(test_images_x, test_y, batch_size=batch_size)

    if not TESTING:

        try:
            print("----------------- TRAINING -----------------")
            history = model.fit_generator(train_generator,
                                validation_data=test_generator,
                                steps_per_epoch=int(train_images_x_total/batch_size),
                                validation_steps=int(test_images_x_total/batch_size),
                                epochs=epochs,
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

        model.fit_generator(train_generator,
                                validation_data=test_generator,
                                steps_per_epoch=int(train_images_x_total/batch_size),
                                validation_steps=int(test_images_x_total/batch_size),
                                epochs=epochs,
                                verbose=1,
                                shuffle=True)

    return model

def predict_abnormality(model):

    df = helper_funcs.load_dataset_attributes(valid_dataset_file)

    valid_images_x = np.array(helper_funcs.load_images(df))

    valid_y = np.array(df['target'])

    predict_y = model.predict_generator(ImageDataGenerator().flow(valid_images_x, valid_y, batch_size=batch_size))

    diff = predict_y.flatten() - valid_y
    # percentage_diff = (diff/ valid_y) * 100
    abs_percentage = np.abs(diff)
    mean = np.mean(abs_percentage)

    print("Mean diff: {}".format(mean))
