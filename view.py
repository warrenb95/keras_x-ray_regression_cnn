import tkinter as tk
class View():

    def __init__(self, master):
        print("init view")
        self.make_image_frame(master)
        self.make_side_frame(master)

    def make_image_frame(self, master):
        master.update()
        self.img_frame = tk.Frame(master, width = master.winfo_width() * 0.60, height = master.winfo_height(), bg = "grey")
        self.img_frame.grid(row = 0, column = 0)

    def make_side_frame(self, master):
        master.update()
        self.side_frame = tk.Frame(master, width = master.winfo_width() * 0.40, height = master.winfo_height(), bg = "white")
        self.side_frame.grid(row = 0, column = 1)

        self.make_file_frame(self.side_frame)

        self.result_label = tk.Label(self.side_frame, text = "RESULT", font = "14")
        self.result_label.pack()

    def make_file_frame(self, master):
        master.update()
        self.file_frame = tk.Frame(master, width = master.winfo_width(), height = master.winfo_height() * 0.3, bg = "grey")
        self.file_frame.pack()

        self.add_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Add", font = "10")
        self.add_btn.grid(row = 0, column = 1, padx = 50, pady = 12.5)

        self.delete_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Delete", font = "10")
        self.delete_btn.grid(row = 1, column = 1, padx = 50, pady = 12.5)

