import os
import helper_funcs
import classification_trainer

class Model():

    def __init__(self):
        print("init model")
        self.current_image = 0
        self.user_in_path = os.path.dirname(os.path.abspath(__file__)) + "/user_in"

    def add_files(self, event):
        os.startfile(self.user_in_path)

    def upload_files(self):
        src_files = os.listdir(self.user_in_path)

        self.image_paths = []

        for file_name in src_files:
            full_file_name = os.path.join(self.user_in_path, file_name)

            if os.path.isfile(full_file_name):
                self.image_paths.append(full_file_name)

    def get_cur_image_path(self):
        return self.image_paths[self.current_image]

    def set_prev_image_path(self):
        if self.current_image - 1 >= 0:
            self.current_image -= 1

    def set_next_image_path(self):
        if self.current_image + 1 < len(self.image_paths):
            self.current_image += 1

    def classify_image(self, image_path):
        class_model = helper_funcs.load_model("class")
        return classification_trainer.predict_classification(class_model, image_path)
