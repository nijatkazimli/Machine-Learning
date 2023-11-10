import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image
import face_recognition
import cv2
import os

def choose_folder():
    folder_path = filedialog.askdirectory()
    print("Folder Selected:", folder_path)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, file)
            print("Processing image:", image_path)
            face_recognition.handle_image(image_path)

def choose_video():
    file_path = filedialog.askopenfilename()
    print("File selected:", file_path)
    face_recognition.handle_video(file_path)

def start_camera():
    cap = cv2.VideoCapture(0)
    face_recognition.handle_camera(cap)

root = tk.Tk()
root.title("Face Recognition / Age Detection")

style = ttk.Style()
style.configure('TButton', font=('Arial', 10))

btn_choose_folder = ttk.Button(root, text="Choose Photo Folder", command=choose_folder)
btn_choose_folder.pack(fill='x', padx=10, pady=5)

btn_choose_video = ttk.Button(root, text="Choose Video", command=choose_video)
btn_choose_video.pack(fill='x', padx=10, pady=5)

btn_live_feed = ttk.Button(root, text="Start Live Camera", command=start_camera)
btn_live_feed.pack(fill='x', padx=10, pady=5)

root.mainloop()

