import pandas as pd
import cv2
from tensorflow.keras.models import model_from_json
from keras.engine.sequential import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras import applications
from keras.engine.training import Model

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

def save_model(model: Sequential, model_name: str):
    '''Save the current passed in model as model_name.

    Parameters
    ----------
    model: Sequential
        Keras model.

    model_name: str
        The name of the model.
    '''

    model_json = model.to_json()
    with open("models/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("models/" + model_name + ".h5")
    print("Saved %s to disk" % model_name)

def load_model(model_name: str) -> Sequential:
    '''Load in model_name and return Sequential.

    Parameters
    ----------
    model_name: str
        The name of the model to load.

    Returns
    -------
    loaded_model: Sequential
        The loaded keras model.
    '''
    
    json_file = open("models/" + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights("models/" + model_name + ".h5")
    print("Loaded %s from disk" % model_name)

    return loaded_model

def create_new_model(regression: bool, class_num: int) -> Sequential:
    '''Create a VGG16 model with 1/2 the layers, return the model.

    Parameters
    ----------
    regression: bool
        Is the model a regression model?

    class_num: int
        If the model not regression must specify the class num.

    Returns
    -------
    model: Sequential
        Keras model.
    '''

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

    # Handle if model is regression
    if regression:
        model.add(Dense(1, activation='linear'))
    else:
        model.add(Dense(class_num, activation='softmax'))

    return model

def create_resnet_model():

    base = applications.ResNet152V2(include_top=False, weights=None, input_shape=(112, 112, 3), pooling='max')

    x = base.output
    x = Dropout(0.2)(x)

    output = Dense(1, activation='linear')(x)
    model = Model(inputs=base.input, outputs=output)

    return model

def create_desnet121():
    base = applications.DenseNet121(include_top=False, weights=None, input_shape=(112, 112, 3), pooling='max')

    x = base.output
    x = Dropout(0.2)(x)

    output = Dense(1, activation='linear')(x)
    return Model(inputs=base.input, outputs=output)

def load_dataset_attributes(input_path: str) -> pd.DataFrame:
    '''Load the information from the 'input_path' csv file.

    Parameters
    ----------
    input_path: str
        The path to the attributes csv file.

    Returns
    -------
    dataframe: pd.DataFrame
        The attributes dataframe.
    '''

    return pd.read_csv(input_path, sep=',', header=None, names=['path', 'target'])

def load_images(df: pd.DataFrame):
    '''Load and return the correct images from the 'df'.

    Parameters
    ----------
    df: pd.DataFrame
        The attributes dataframe.

    Returns
    -------
    images: [] cv2 images
        A list of images.
    '''

    images = []

    for path in df['path']:
        try:
            cur_image = cv2.imread(path)
        except:
            print("Error: {}, not loaded".format(path))
            exit()

        images.append(cur_image)

    return images
