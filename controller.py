import tkinter as tk
from model import Model
from view import View
from PIL import Image, ImageTk

class Controller():
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

    def run(self):
        self.root.title("X-Ray Processor")
        self.root.deiconify()
        self.root.mainloop()

    def upload_images(self, event):
        self.model.upload_files()

        self.display_cur_image()

    def display_cur_image(self):
        self.view.set_image(self.model.image_paths[self.model.current_image])

    def display_prev_image(self, event):
        self.model.set_prev_image_path()
        self.view.set_image(self.model.image_paths[self.model.current_image])
        print("Previous image clicked, cur image is {}".format(self.model.current_image))

    def display_next_image(self, event):
        self.model.set_next_image_path()
        self.view.set_image(self.model.image_paths[self.model.current_image])
        print("Next image clicked, cur image is {}".format(self.model.current_image))
