import tkinter as tk
from model import Model
from view import View

class Controller():
    def __init__(self):
        print("init controller")
        self.root = tk.Tk()
        self.root.geometry("1200x720")
        self.root.resizable(0, 0)
        self.model = Model()
        self.view = View(self.root)

    def run(self):
        self.root.title("X-Ray Processor")
        self.root.mainloop()
