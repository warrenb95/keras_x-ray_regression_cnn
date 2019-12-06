from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime
from keras import backend as k_back
import os

if __name__ == "__main__":
    # Testing section
    # ---------------------------------------------------------------------
    model = regression_trainer.train_new('elbow')
    # ---------------------------------------------------------------------

    # Uncomment to run the GUI
    # ---------------------------------------------------------------------
    # controller = Controller()
    # controller.run()
    # ---------------------------------------------------------------------

    # Uncomment to train the regression models
    # ---------------------------------------------------------------------
    # model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    # for m in model_list:
    #     model = regression_trainer.train_old(m)
    #     model = None
    #     k_back.clear_session()
    # ---------------------------------------------------------------------

    # Uncomment to train classification model
    # ---------------------------------------------------------------------
    # model = classification_trainer.train_old()
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # This will turn the PC off, use when training overnight
    # os.system("shutdown /s /t 1")
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
