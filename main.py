from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime

if __name__ == "__main__":

    print(
        "1. Launch the GUI application. (1)\n"
        "2. Train classification (2)\n"
        "3. Train regression ('elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist')\n"
    )

    usr_in = input("Please select: ")

    if int(usr_in) == 1:
        controller = Controller()
        controller.run()
    elif int(usr_in) == 2:
        model = classification_trainer.train_old()
    elif usr_in == 'elbow':
        model = regression_trainer.train_old('elbow')
    elif usr_in == 'finger':
        model = regression_trainer.train_old('finger')
    elif usr_in == 'forearm':
        model = regression_trainer.train_old('forearm')
    elif usr_in == 'hand':
        model = regression_trainer.train_old('hand')
    elif usr_in == 'humerus':
        model = regression_trainer.train_old('humerus')
    elif usr_in == 'shoulder':
        model = regression_trainer.train_old('shoulder')
    elif usr_in == 'wrist':
        model = regression_trainer.train_old('wrist')
    else:
        print("Invalid option... Bye")
