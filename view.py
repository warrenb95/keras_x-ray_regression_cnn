import tkinter as tk
from PIL import ImageTk, Image
class View():

    def __init__(self, root):
        print("init view")
        self.root = root

        self.regression_result_str = ""

        self.file_list = []

        self.make_image_frame()
        self.make_side_frame()
        self.add_tmp_img()

    def make_image_frame(self):
        self.root.update()

        self.image_canvas = tk.Canvas(self.root, width = self.root.winfo_width() * 0.60, height = self.root.winfo_height())
        self.image_canvas.pack(side = tk.LEFT)

    def add_tmp_img(self):
        self.img = ImageTk.PhotoImage(Image.open("instructions.png").resize((self.image_canvas.winfo_width(), self.image_canvas.winfo_height()), Image.ANTIALIAS))
        self.image_on_canvas =  self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def set_image(self, image_path):
        self.img = ImageTk.PhotoImage(Image.open(image_path).resize((self.image_canvas.winfo_width(), self.image_canvas.winfo_height()), Image.ANTIALIAS))
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.img)

        self.set_regression_result("", "white")

    def make_side_frame(self):
        self.root.update()
        self.side_frame = tk.Frame(self.root, width = self.root.winfo_width() * 0.40, height = self.root.winfo_height(), bg = "white")
        self.side_frame.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.regression_result_label = tk.Label(self.side_frame, text = self.regression_result_str, font = "20 20", fg = "#ff5c33", borderwidth = 2, relief = "solid")
        self.regression_result_label.pack(fill = tk.X, pady = 25, padx = 5)

        self.make_btn_frame(self.side_frame)

    def make_btn_frame(self, side_frame):
        self.root.update()

        self.btn_frame = tk.Frame(side_frame, width = side_frame.winfo_width(), height = side_frame.winfo_height() * 0.3, bg = "grey")
        self.btn_frame.pack(fill = tk.X, padx = 10, pady = 10, side=tk.BOTTOM)

        self.add_btn = tk.Button(self.btn_frame, height = 2, text = "Add Radiograph(s)", bg = "#70db70")
        self.add_btn.grid(row = 0, column = 0, columnspan=2, pady = 10)

        self.upload_btn = tk.Button(self.btn_frame, height = 2, text = "Upload Radiograph(S)", bg = "#70db70")
        self.upload_btn.grid(row = 1, column = 0, columnspan=2, pady = 10)

        # self.delete_btn = tk.Button(self.btn_frame, height = 2, text = "Close Radiograph(s)", font = "10", bg = "#ff5c33")
        # self.delete_btn.grid(row = 2, column = 0, columnspan=2, pady = 10)

        self.prev_btn = tk.Button(self.btn_frame, text = "Previous Radiograph")
        self.prev_btn.grid(row=3, column=0, padx = 50, pady = 10)

        self.next_btn = tk.Button(self.btn_frame, text = "Next Radiograph")
        self.next_btn.grid(row=3, column=1, padx = 50, pady = 10)

        self.process_btn = tk.Button(self.btn_frame, text = "Process Radiograph", font = "2")
        self.process_btn.grid(row=4, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W, padx = 50, pady = 10)

    def set_regression_result(self, regress_result, colour):
        self.regression_result_str = str(regress_result) + '% Abnormal'
        self.regression_result_label.configure(text=self.regression_result_str)

        self.side_frame.configure(bg=colour)

    def set_loading_txt(self):
        self.regression_result_str = "Processing radiograph(x-ray)..."
        self.regression_result_label.configure(text=self.regression_result_str)

        self.side_frame.configure(bg = "white")

        self.root.update()
