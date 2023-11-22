import cv2
import numpy as np
from interfaces import FaceRecognitionModel

class HaarFrontModel(FaceRecognitionModel):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.color = (255, 0, 0)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return faces
    
class HaarProfileModel(FaceRecognitionModel):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.color = (255, 0, 255)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
        return faces

class HaarCombinedModel(FaceRecognitionModel):
    def __init__(self):
        self.front_model = HaarFrontModel()
        self.profile_model = HaarProfileModel()
        self.color = (0, 0, 255)

    def detect_faces(self, image):
        front_faces = self.front_model.detect_faces(image)
        profile_faces = self.profile_model.detect_faces(image)

        r = []
        
        if isinstance(profile_faces, np.ndarray):
            for face in profile_faces:
                r.append(face)

        if isinstance(profile_faces, tuple) and len(profile_faces) > 0:
            r.append(profile_faces)

        if isinstance(front_faces, tuple) and len(front_faces) > 0:
            r.append(front_faces)

        if isinstance(front_faces, np.ndarray):
            for face in front_faces:
                r.append(face)

        return r
