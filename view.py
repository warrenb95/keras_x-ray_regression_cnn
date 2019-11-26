import tkinter as tk
from PIL import ImageTk, Image
class View():

    def __init__(self, root):
        print("init view")

        self.classification_result_str = "NO_CLASSIFICATION"

        self.regression_result_val = 0.0
        self.regression_result_str = str(self.regression_result_val) + " - Abnormal"

        self.file_list = []

        self.make_image_frame(root)
        self.make_side_frame(root)
        self.add_tmp_img()

    def make_image_frame(self, root):
        root.update()

        self.image_canvas = tk.Canvas(root, width = root.winfo_width() * 0.60, height = root.winfo_height())
        self.image_canvas.pack(side = tk.LEFT)

    def add_tmp_img(self):
        self.img = ImageTk.PhotoImage(Image.open("temp_image.png").resize((self.image_canvas.winfo_width(), self.image_canvas.winfo_height()), Image.ANTIALIAS))
        self.image_on_canvas =  self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def set_image(self, image_path):
        self.img = ImageTk.PhotoImage(Image.open(image_path).resize((self.image_canvas.winfo_width(), self.image_canvas.winfo_height()), Image.ANTIALIAS))
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.img)

    def make_side_frame(self, root):
        root.update()
        self.side_frame = tk.Frame(root, width = root.winfo_width() * 0.40, height = root.winfo_height(), bg = "white")
        self.side_frame.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.make_file_frame(self.side_frame)

        self.classification_result_label = tk.Label(self.side_frame, text = self.classification_result_str, font = "20 20", fg = "#ff5c33", borderwidth = 2, relief = "solid")
        self.classification_result_label.pack(fill = tk.X, pady = 25, padx = 5)

        self.regression_result_label = tk.Label(self.side_frame, text = self.regression_result_str, font = "40 40", fg = "#ff5c33", borderwidth = 2, relief = "solid")
        self.regression_result_label.pack(fill = tk.X, pady = 25, padx = 5)

        self.make_btn_frame(self.side_frame)

    def make_file_frame(self, root):
        root.update()
        self.file_frame = tk.Frame(root, width = root.winfo_width(), height = root.winfo_height() * 0.3, bg = "grey")
        self.file_frame.pack(fill = tk.X, padx = 10, pady = 10)

        self.upload_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Upload", font = "10", bg = "#70db70")
        self.upload_btn.grid(row = 0, column = 1, padx = 50, pady = 12.5)

        self.add_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Add", font = "10", bg = "#70db70")
        self.add_btn.grid(row = 0, column = 0, padx = 50, pady = 12.5)

        self.delete_btn = tk.Button(self.file_frame, width = 5, height = 2, text = "Delete", font = "10", bg = "#ff5c33")
        self.delete_btn.grid(row = 1, column = 1, padx = 50, pady = 12.5)

    def make_btn_frame(self, root):
        root.update()

        self.btn_frame = tk.Frame(root, width = root.winfo_width(), height = root.winfo_height() * 0.3, bg = "grey")
        self.btn_frame.pack(fill = tk.X, padx = 10, pady = 10, side=tk.BOTTOM)

        self.prev_btn = tk.Button(self.btn_frame, text = "Previous image", font = "18")
        self.prev_btn.grid(row=0, column=0, sticky=tk.E, padx = 50, pady = 12.5)

        self.next_btn = tk.Button(self.btn_frame, text = "Next image", font = "18")
        self.next_btn.grid(row=0, column=1, sticky=tk.W, padx = 50, pady = 12.5)

        self.process_btn = tk.Button(self.btn_frame, text = "Process X-Ray", font = "24")
        self.process_btn.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W, padx = 50, pady = 12.5)

    def set_classification_str(self, classification):
        self.classification_result_str = classification
        self.classification_result_label.configure(text=self.classification_result_str)
