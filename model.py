import os
import helper_funcs
import classification_trainer as ct
import regression_trainer as rt

class Model():

    def __init__(self):
        print("init model")
        self.current_image = 0
        self.user_in_path = os.path.dirname(os.path.abspath(__file__)) + "/user_in"
        self.class_trainer = ct.Classification_Trainer().getInstance()
        self.regres_trainer = rt.Regression_Trainer().getInstance()

    def add_files(self, event):
        '''Opens file explorer at 'user_in_path'.
        '''

        os.startfile(self.user_in_path)

    def upload_files(self):
        '''Uploads the files from 'user_in_path' to the program.
        '''

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

    def predict_abnormality(self, image_path, model_count):
        class_result, cur_image = self.class_trainer.predict_classification(image_path)

        if class_result == 0:
            return self.regres_trainer.predict(cur_image, 'elbow', model_count)
        elif class_result == 1:
            return self.regres_trainer.predict(cur_image, 'finger', model_count)
        elif class_result == 2:
            return self.regres_trainer.predict(cur_image, 'forearm', model_count)
        elif class_result == 3:
            return self.regres_trainer.predict(cur_image, 'hand', model_count)
        elif class_result == 4:
            return self.regres_trainer.predict(cur_image, 'humerus', model_count)
        elif class_result == 5:
            return self.regres_trainer.predict(cur_image, 'shoulder', model_count)
        elif class_result == 6:
            return self.regres_trainer.predict(cur_image, 'wrist', model_count)
        else:
            print('Invalid class_result {}'.format(class_result))
            return None


