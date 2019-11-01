import tkinter as tk
class View():

    def __init__(self, master):
        print("init view")

        self.result_val = 0.0
        self.result_str = str(self.result_val) + " - Abnormal"

        self.file_list = []

        self.make_image_frame(master)
        self.make_side_frame(master)

    def make_image_frame(self, master):
        master.update()
        self.img_frame = tk.Frame(master, width = master.winfo_width() * 0.60, height = master.winfo_height(), bg = "grey")
        self.img_frame.pack(side = tk.LEFT)

    def make_side_frame(self, master):
        master.update()
        self.side_frame = tk.Frame(master, width = master.winfo_width() * 0.40, height = master.winfo_height(), bg = "white")
        self.side_frame.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.make_file_frame(self.side_frame)

        self.result_label = tk.Label(self.side_frame, text = self.result_str, font = "40 40", fg = "#ff5c33", borderwidth = 2, relief = "solid")
        self.result_label.pack(fill = tk.X, pady = 75, padx = 5)

        self.process_btn = tk.Button(self.side_frame, text = "Process X-Ray", font = "18")
        self.process_btn.pack(pady = 25, padx = 5, side = "bottom")

    def make_file_frame(self, master):
        master.update()
        self.file_frame = tk.Frame(master, width = master.winfo_width(), height = master.winfo_height() * 0.3, bg = "grey")
        self.file_frame.pack(fill = tk.X, padx = 10, pady = 10)

        self.add_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Add", font = "10", bg = "#70db70")
        self.add_btn.grid(row = 0, column = 1, padx = 50, pady = 12.5)

        self.delete_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Delete", font = "10", bg = "#ff5c33")
        self.delete_btn.grid(row = 1, column = 1, padx = 50, pady = 12.5)
