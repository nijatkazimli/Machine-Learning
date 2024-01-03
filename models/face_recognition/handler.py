import cv2
import os
from interfaces import FaceRecognitionHandler, FaceRecognitionModel
from tkinter import filedialog
from tkinter import Tk
import numpy as np
import threading


class OpenCVFaceRecognition(FaceRecognitionHandler):
    def __init__(self, models: [FaceRecognitionModel]):
        self.models = models
        super().__init__(models)
        self.save_frame = None
        self.frame_lock = threading.Lock()

    def draw_face_rectangles(self, image, faces, color):
        if color is None:
            color = (0, 255, 0)
        for (x, y, w, h) in faces:
            x, y, w, h = np.int32(x), np.int32(y), np.int32(w), np.int32(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # def handle_camera(self, cap):
    #     while True:
    #         ret, frame = cap.read()
    #         if ret:
    #             for model in self.models:
    #                 faces = model.detect_faces(frame)
    #                 self.draw_face_rectangles(frame, faces, model.color)
    #             cv2.imshow('Camera Face Detection', frame)
    #             cv2.setWindowProperty('Camera Face Detection', cv2.WND_PROP_TOPMOST, 1)
    #             key = cv2.waitKey(1)
    #             if key == ord('q') or key == 27 or cv2.getWindowProperty('Camera Face Detection',
    #                                                                      cv2.WND_PROP_VISIBLE) < 1:
    #                 break
    #     cap.release()
    #     cv2.destroyAllWindows()

    def handle_camera(self, cap):
        cv2.namedWindow('Camera Face Detection', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if ret:
                # Original frame aspect ratio
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height

                # Get current window size
                window_width, window_height = cv2.getWindowImageRect('Camera Face Detection')[2:]
                if window_width == 0 or window_height == 0:
                    continue
                window_aspect_ratio = window_width / window_height

                # Calculate new frame dimensions
                if window_aspect_ratio > aspect_ratio:
                    # Window is wider than the frame
                    new_height = window_height
                    new_width = int(aspect_ratio * new_height)
                else:
                    # Window is taller than the frame
                    new_width = window_width
                    new_height = int(new_width / aspect_ratio)

                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Perform face detection and draw rectangles on the resized_frame
                for model in self.models:
                    faces = model.detect_faces(resized_frame)
                    self.draw_face_rectangles(resized_frame, faces, model.color)

                # Update the frame to be saved under lock
                with self.frame_lock:
                    self.save_frame = np.copy(resized_frame)

                # Create a black canvas and center the resized frame
                canvas = np.zeros((window_height, window_width, 3), dtype='uint8')
                x_offset = (window_width - new_width) // 2
                y_offset = (window_height - new_height) // 2
                canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

                # Display the save instruction
                instruction_text = "Press 's' to save the frame"
                cv2.putText(canvas, instruction_text, (10, window_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('Camera Face Detection', canvas)
                cv2.setWindowProperty('Camera Face Detection', cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27 or cv2.getWindowProperty('Camera Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Save frame on 's' key press
                if key == ord('s'):
                    threading.Thread(target=self.save_image_thread, daemon=True).start()

        cap.release()
        cv2.destroyAllWindows()

    def save_image_thread(self):
        # Tkinter dialog in a separate thread
        root = Tk()
        root.withdraw()

        # Keep the file dialog always on top
        root.attributes('-topmost', True)

        with self.frame_lock:
            save_frame_copy = np.copy(self.save_frame)

        file_path = filedialog.asksaveasfilename(parent=root, defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, save_frame_copy)

        root.destroy()

    def handle_image(self, image_path, width, height):
        image = cv2.imread(image_path)
        for model in self.models:
            faces = model.detect_faces(image)
            self.draw_face_rectangles(image, faces, model.color)

        if image.shape[1] > width or image.shape[0] > height:
            r = min(width / image.shape[1], height / image.shape[0])
            dim = (int(image.shape[1] * r), int(image.shape[0] * r))
            image = cv2.resize(image, dim)

        cv2.imshow('Image Face Detection', image)
        cv2.setWindowProperty('Image Face Detection', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('Image Face Detection', 0, 0)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27 or cv2.getWindowProperty('Image Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()

        return image

    def handle_video(self, video_path, width, height):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        processed_video_path = os.path.splitext(video_path)[0] + "_processed.mp4"
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            r = min(width / frame.shape[1], height / frame.shape[0])
            dim = (int(frame.shape[1] * r), int(frame.shape[0] * r))
            frame = cv2.resize(frame, dim)

            for model in self.models:
                faces = model.detect_faces(frame)
                self.draw_face_rectangles(frame, faces, model.color)

            out.write(frame)
            cv2.imshow('Video Face Detection', frame)
            cv2.setWindowProperty('Video Face Detection', cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow('Video Face Detection', 0, 0)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or cv2.getWindowProperty('Video Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return processed_video_path
