from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime
from keras import backend as k_back

if __name__ == "__main__":

    # print(
    #     "1. Launch the GUI application. (1)\n"
    #     "2. Train classification (2)\n"
    #     "3. Train regression ('elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist')\n"
    # )

    # usr_in = input("Please select: ")

    # if usr_in == '1':
    #     controller = Controller()
    #     controller.run()
    # elif usr_in == '2':
    #     model = classification_trainer.train_old()
    # elif usr_in == 'elbow':
    #     model = regression_trainer.train_old('elbow')
    # elif usr_in == 'finger':
    #     model = regression_trainer.train_old('finger')
    # elif usr_in == 'forearm':
    #     model = regression_trainer.train_old('forearm')
    # elif usr_in == 'hand':
    #     model = regression_trainer.train_old('hand')
    # elif usr_in == 'humerus':
    #     model = regression_trainer.train_old('humerus')
    # elif usr_in == 'shoulder':
    #     model = regression_trainer.train_old('shoulder')
    # elif usr_in == 'wrist':
    #     model = regression_trainer.train_old('wrist')
    # else:
    #     print("Invalid option... Bye")

    # The following is for training purposes
    model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    for m in model_list:
        model = regression_trainer.train_new(m)
        model = None
        k_back.clear_session()

    # model = regression_trainer.train_new(model_list[0])