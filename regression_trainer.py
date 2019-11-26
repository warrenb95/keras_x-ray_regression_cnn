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
opt = settings.opt

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_images_x_total = 0
test_images_x_total = 0
validation_images_x_total =0

def load_regression_data(body_part):
    global train_images_x_total
    global test_images_x_total

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

    train_y = np.array(train_attribs_x['target'])
    test_y = np.array(test_attribs_x['target'])

    train_images_x_total = len(train_images_x)
    test_images_x_total = len(test_images_x)

    return (train_images_x, train_y, test_images_x, test_y)

def train_new(body_part):

    model = helper_funcs.create_new_model(True, 0)

    # model.compile(optimizer = opt, loss = 'msle')
    model.compile(optimizer = opt, loss = 'mse')

    return train_regression_model(model)

def train_old(body_part):

    model = helper_funcs.load_model(body_part)

    # model.compile(optimizer = opt, loss = 'msle')
    model.compile(optimizer = opt, loss = 'mse')

    return train_regression_model(model)

def train_regression_model(model):

    train_images_x, train_y, test_images_x, test_y = load_regression_data(body_part)

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

def validate(model, body_part):
    global validation_images_x_total

    valid_dataset_file = 'dataset/' + 'valid_' + body_part + '.csv'

    df = helper_funcs.load_dataset_attributes(valid_dataset_file)

    valid_images_x = np.array(helper_funcs.load_images(df))
    valid_y = np.array(df['target'])

    validation_images_x_total = len(valid_images_x)

    predict_y = model.predict_generator(ImageDataGenerator().flow(valid_images_x, valid_y, batch_size=batch_size),
                                        verbose=1)

    flat_y = predict_y.flatten()

    for i in range(flat_y):
        if valid_y[i] == 1.0 and flat_y[i] >= 0.6:
            flat_y[i] = 1.0

    diff = flat_y - valid_y
    mean = np.mean(diff)

    print("Mean difference: {:.2f}".format(np.abs(mean)))
