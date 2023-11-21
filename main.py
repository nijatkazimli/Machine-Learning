import tkinter as tk
from tkinter import filedialog, ttk , messagebox
import face_recognition
import cv2
import os


def choose_folder():
    try:
        folder_path = filedialog.askdirectory()
        folder_path = os.path.normpath(folder_path)
        if not folder_path:
            raise Exception("No folder selected")
        print("Folder Selected:", folder_path)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.jfif')):
                image_path = os.path.join(folder_path, file)
                print("Processing image:", image_path)
                face_recognition.handle_image(image_path, screen_width, screen_height)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def choose_video():
    try:
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv'),
            ('All files', '*.*')
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if not file_path:
            raise Exception("No video file selected")
        print("File Selected:", file_path)
        face_recognition.handle_video(file_path, screen_width, screen_height)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def start_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        face_recognition.handle_camera(cap)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == '__main__':
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
