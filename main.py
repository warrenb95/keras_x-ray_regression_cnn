from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime

if __name__ == "__main__":

    # print(
    #     "1. Launch the GUI application.\n"
    #     "2. Train network"
    # )

    # usr_in = input("Please select: ")

    # if int(usr_in) == 1:
    #     controller = Controller()
    #     controller.run()
    # elif int(usr_in) == 2:

    #     model = trainer.train_new_classification()

    # calssification_trainer.train_new()

    # model = regression_trainer.train_new_regression()
    # regression_trainer.validate(model)

    controller = Controller()
    controller.run()
