from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime
from keras import backend as k_back
import os

if __name__ == "__main__":

    # controller = Controller()
    # controller.run()

    # The following is for training purposes
    # model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    # for m in model_list:
    #     model = regression_trainer.train_old(m)
    #     model = None
    #     k_back.clear_session()

    # model = classification_trainer.train_old()

    # This will turn the PC off, use when training overnight
    os.system("shutdown /s /t 1")
