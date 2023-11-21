import tkinter as tk
from tkinter import filedialog, ttk
import face_recognition
import cv2
import os


def choose_folder():
    folder_path = filedialog.askdirectory()
    print("Folder Selected:", folder_path)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.jfif')):
            image_path = os.path.join(folder_path, file)
            print("Processing image:", image_path)
            face_recognition.handle_image(image_path)


def choose_video():
    filetypes = [
        ('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv'),  # Add or remove formats as needed
        ('All files', '*.*')
    ]
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    if file_path:  # Check if a file was actually selected
        print("File Selected:", file_path)
        face_recognition.handle_video(file_path, screen_width, screen_height)


def start_camera():
    cap = cv2.VideoCapture(0)
    face_recognition.handle_camera(cap)


root = tk.Tk()
root.title("Face Recognition / Age Detection")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

style = ttk.Style()
style.configure('TButton', font=('Arial', 10))

btn_choose_folder = ttk.Button(root, text="Choose Photo Folder", command=choose_folder)
btn_choose_folder.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

btn_choose_video = ttk.Button(root, text="Choose Video", command=choose_video)
btn_choose_video.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)

btn_live_feed = ttk.Button(root, text="Start Live Camera", command=start_camera)
btn_live_feed.grid(row=5, column=0, sticky='nsew', padx=10, pady=5)

root.columnconfigure(0, weight=1)

root.rowconfigure(1, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(5, weight=1)

root.rowconfigure(0, weight=0)
root.rowconfigure(2, weight=0)
root.rowconfigure(4, weight=0)
root.rowconfigure(6, weight=0)

root.mainloop()
