from controller import Controller
import trainer as trainer

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

    model = trainer.train_new_classification()