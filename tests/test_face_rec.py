import os
import time
import cv2
from models.face_recognition.handler import OpenCVFaceRecognition
from models.face_recognition.haar_fast import HaarFaceModel

def count_images_with_faces(face_recognition, directory):
    count = 0
    total = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            total += 1
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            faces = face_recognition.model.detect_faces(img)
            if len(faces) > 0:
                count += 1
    return count, total,

haar_model = HaarFaceModel()
haar_face_recognition = OpenCVFaceRecognition(haar_model)

models = [haar_face_recognition]

faces_dir = './tests/data/faces/'
other_dir = './tests/data/other/'

for model in models:
    print(f"Model: {type(model.model).__name__}")
    start_time = time.time()
    faces_count, total_faces = count_images_with_faces(model, faces_dir)
    other_count, total_other = count_images_with_faces(model, other_dir)
    time_taken = time.time() - start_time

    total_images = total_faces + total_other
    print(f"Faces detected in {faces_count / total_faces * 100}% of images from the 'faces' directory.")
    print(f"Faces detected in {other_count / total_other * 100}% of images from the 'other' directory.")
    print(f"Time taken to test {total_images} images: {time_taken} seconds ({total_images/time_taken} images/second)")