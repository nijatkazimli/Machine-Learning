import cv2
import numpy as np
import os
from interfaces import FaceRecognitionModel

class ResNetModel(FaceRecognitionModel):
    def __init__(self, model_path="", prototxt_path=""):
        self.color = (0, 255, 0)
        prefix = './models/face_recognition/'
        model_path = prefix + 'res10_300x300_ssd_iter_140000.caffemodel'
        prototxt_path = prefix + 'deploy.prototxt'
        prototxt_path = os.path.normpath(prototxt_path)
        model_path = os.path.normpath(model_path)
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect_faces(self, image, confidence_threshold=0.5):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX-startX, endY-startY))

        return faces