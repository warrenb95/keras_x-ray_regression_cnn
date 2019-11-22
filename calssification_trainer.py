import settings
import helper_funcs
from keras.preprocessing.image import ImageDataGenerator
import datetime
from matplotlib import pyplot as plt

TESTING = settings.TESTING
batch_size = settings.batch_size
epochs = settings.epochs
body_part = settings.body_part
opt = settings.opt

train_path = 'dataset/train'
valid_path = 'dataset/valid'

classification_train_total = 36808
classification_test_total = 3197

def train_new():
    model = helper_funcs.create_new_model(False, 7)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_classification_model(model)

def train_old():
    model = helper_funcs.load_model(body_part)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_classification_model(model)

def train_classification_model(model):

    classification_train_generator = ImageDataGenerator().flow_from_directory(train_path,
                                                                                target_size = (112, 112),
                                                                                batch_size = batch_size)

    classification_valid_generator = ImageDataGenerator().flow_from_directory(valid_path,
                                                                                target_size = (112, 112),
                                                                                batch_size = batch_size)

    if not TESTING:

        try:
            print("----------------- TRAINING -----------------")
            history = model.fit_generator(classification_train_generator,
                                validation_data=classification_valid_generator,
                                steps_per_epoch=int(classification_train_total/batch_size),
                                validation_steps=int(classification_test_total/batch_size),
                                epochs=epochs,
                                verbose=1,
                                shuffle=True)

            curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Plot training & validation data
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('training data')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')

            fname = "model_graphs/" + curr_datetime + '_' + body_part + '.jpg'
            plt.savefig(fname)

        except KeyboardInterrupt:
            helper_funcs.save_model(model, body_part)
        else:
            helper_funcs.save_model(model, body_part)
    else:
        print("----------------- TESTING -----------------")
        model.fit_generator(classification_train_generator,
                            steps_per_epoch=int(classification_train_total/batch_size),
                            validation_data=classification_valid_generator,
                            validation_steps=int(classification_test_total/batch_size),
                            epochs=epochs,
                            verbose=1,
                            shuffle=True)

