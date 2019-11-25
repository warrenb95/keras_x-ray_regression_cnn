import typing
import pandas as pd
import cv2
from keras.engine.saving import model_from_json
from keras.engine.sequential import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten

def save_model(model: typing.Type[Sequential], model_name: str):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved %s to disk" % model_name)

def load_model(model_name: str) -> Sequential:
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded %s from disk" % model_name)

    return loaded_model

def create_new_model(regression: bool, class_num: int) -> Sequential:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(112, 112, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))

    if regression:
        model.add(Dense(1, activation='linear'))
    else:
        model.add(Dense(class_num, activation='softmax'))

    return model

def load_dataset_attributes(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, sep=',', header=None, names=['path', 'target'])

def load_images(df: pd.DataFrame):

    images = []

    for path in df['path']:
        cur_image = cv2.imread(path)
        cur_image = cv2.resize(cur_image, (112, 112))

        images.append(cur_image)

    return images
