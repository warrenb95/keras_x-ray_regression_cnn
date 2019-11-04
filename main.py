from controller import Controller

if __name__ == "__main__":

    print(
        "1. Launch the GUI application.\n"
        "2. Train Classificatioin network"
    )

    usr_in = input("Please select: ")

    if int(usr_in) == 1:
        controller = Controller()
        controller.run()
    elif int(usr_in) == 2:
        # Todo
        pass
