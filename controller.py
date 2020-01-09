import tkinter as tk
from model import Model
from view import View
from PIL import Image, ImageTk

class Controller():
    '''GUI Controller

    Controlls the View using the Model.
    '''

    def __init__(self):
        print("init controller")
        self.root = tk.Tk()
        self.root.geometry("1200x720")
        self.root.resizable(0, 0)
        self.model = Model()
        self.view = View(self.root)

        self.view.add_btn.bind("<Button-1>", self.model.add_files)
        self.view.upload_btn.bind("<Button-1>", self.upload_images)

        self.view.prev_btn.bind("<Button-1>", self.display_prev_image)
        self.view.next_btn.bind("<Button-1>", self.display_next_image)

        self.view.process_btn.bind("<Button-1>", self.process_image)

    def run(self):
        self.root.title("X-Ray Processor")
        self.root.deiconify()
        self.root.mainloop()

    def upload_images(self, event):
        '''Load in the images from the user_in folder.
        '''

        self.model.upload_files()
        self.display_cur_image()

    def display_cur_image(self):
        '''Display the currently selected image in the view.
        '''

        self.view.set_image(self.model.image_paths[self.model.current_image])

    def display_prev_image(self, event):
        '''Display the previous image in the view,
        if none then displays the current image.
        '''

        self.model.set_prev_image_path()
        self.view.set_image(self.model.image_paths[self.model.current_image])
        print("Previous image clicked, cur image is {}".format(self.model.current_image))

    def display_next_image(self, event):
        '''Display the next image in the view,
        if none then display the current image.
        '''

        self.model.set_next_image_path()
        self.view.set_image(self.model.image_paths[self.model.current_image])
        print("Next image clicked, cur image is {}".format(self.model.current_image))

    def process_image(self, event):
        '''Process the current image.

        Classify and then perform the abnormality
        check.
        '''

        image_path = self.model.image_paths[self.model.current_image]

        class_result = self.model.classify_image(image_path)

        if class_result == 0:
            self.view.set_classification_result("Class - Elbow")
            # print("Prediction: {}".format(class_result))
        elif class_result == 1:
            self.view.set_classification_result("Class - Finger")
            # print("Prediction: {}".format(class_result))
        elif class_result == 2:
            self.view.set_classification_result("Class - Forearm")
            # print("Prediction: {}".format(class_result))
        elif class_result == 3:
            self.view.set_classification_result("Class - Hand")
            # print("Prediction: {}".format(class_result))
        elif class_result == 4:
            self.view.set_classification_result("Class - Humerus")
            # print("Prediction: {}".format(class_result))
        elif class_result == 5:
            self.view.set_classification_result("Class - Shoulder")
            # print("Prediction: {}".format(class_result))
        elif class_result == 6:
            self.view.set_classification_result("Class - Wrist")
            # print("Prediction: {}".format(class_result))

        regress_result = self.model.predict_abnormality(image_path, class_result)

        # print("Prediction is {}".format(regress_result))

        if regress_result == None:
            return

        colour = None

        if regress_result <= 25:
            colour = 'Green'
        elif regress_result > 25 and regress_result <= 50:
            colour = 'Yellow'
        elif regress_result > 50 and regress_result <= 75:
            colour = 'Orange'
        elif regress_result > 75:
            colour = 'Red'

        self.view.set_regression_result(regress_result, colour)
