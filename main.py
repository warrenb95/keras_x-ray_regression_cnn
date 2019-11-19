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

        regression_train_generator, regression_test_generator = trainer.load_regression_data()
        model = trainer.train_new_regression(regression_train_generator, regression_test_generator)
        trainer.predict_abnormality(model)
