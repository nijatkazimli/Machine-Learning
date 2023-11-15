import os
import sys
import cv2
from face_rec import detect_faces

def count_images_with_faces(directory):
    count = 0
    total = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            total += 1
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            faces = detect_faces(img)
            if len(faces) > 0:
                count += 1
    return count, total

faces_dir = './test/faces/'
other_dir = './test/other/'

faces_count, total_faces = count_images_with_faces(faces_dir)
other_count, total_other = count_images_with_faces(other_dir)

print(f"Faces detected in {faces_count / total_faces * 100}% of images from the 'faces' directory.")
print(f"Faces detected in {other_count / total_other * 100}% of images from the 'other' directory.")