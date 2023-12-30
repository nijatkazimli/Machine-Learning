import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import os
from interfaces import GUI, FaceRecognitionHandler


class GUIImplementation(GUI):
    def __init__(self, handler: FaceRecognitionHandler):
        super().__init__(handler)
        self.root = tk.Tk()
        self.root.title("Face Recognition / Age Detection")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Set the window size
        window_width = 600
        window_height = 400

        x = int((self.screen_width / 2) - (window_width / 2))
        y = int((self.screen_height / 2) - (window_height / 2))

        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10))

        self.btn_choose_folder = ttk.Button(
            self.root, text="Choose Photo Folder", command=self.choose_folder
        )
        self.btn_choose_folder.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.btn_choose_video = ttk.Button(
            self.root, text="Choose Video", command=self.choose_video
        )
        self.btn_choose_video.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        self.btn_live_feed = ttk.Button(
            self.root, text="Start Live Camera", command=self.start_camera
        )
        self.btn_live_feed.grid(row=5, column=0, sticky="nsew", padx=10, pady=5)

        self.root.columnconfigure(0, weight=1)

        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(5, weight=1)

        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(4, weight=0)
        self.root.rowconfigure(6, weight=0)

    def choose_folder(self):
        folder_path = filedialog.askdirectory()
        folder_path = os.path.normpath(folder_path)
        if not folder_path:
            raise Exception("No folder selected")
        print("Folder Selected:", folder_path)

        saved_folder_path = os.path.join(folder_path, "Processed_Images")
        if not os.path.exists(saved_folder_path):
            os.makedirs(saved_folder_path)

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.jfif')):
                image_path = os.path.join(folder_path, file)
                print("Processing image:", image_path)
                processed_image = self.handler.handle_image(image_path, self.screen_width, self.screen_height)
                saved_image_path = os.path.join(saved_folder_path, file)
                cv2.imwrite(saved_image_path, processed_image)

    def choose_video(self):
        try:
            filetypes = [
                ('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv'),
                ('All files', '*.*')
            ]
            file_path = filedialog.askopenfilename(filetypes=filetypes)
            if file_path:
                processed_video_path = self.handler.handle_video(file_path, self.screen_width, self.screen_height)
                print("Processed video saved at:", processed_video_path)
                messagebox.showinfo("Success", f"Processed video saved at: {processed_video_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_camera(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Failed to open camera")
            self.handler.handle_camera(cap)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def show(self):
        self.root.mainloop()
