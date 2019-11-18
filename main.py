from controller import Controller
from trainer import Trainer

if __name__ == "__main__":

    print(
        "1. Launch the GUI application.\n"
        "2. Train network"
    )

    usr_in = input("Please select: ")

    if int(usr_in) == 1:
        controller = Controller()
        controller.run()
    elif int(usr_in) == 2:
        trainer = Trainer()

        trainer.load_regression_data()
        trainer.train_new_regression()
        trainer.predict_abnormality()
