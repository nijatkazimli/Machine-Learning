import os
import numpy as np
import json
from interfaces import FaceRecognitionModel
from cv2 import FaceDetectorYN

class YuNetModel(FaceRecognitionModel):
    def __init__(self, model_path=".", input_size=(320, 320), threshold=0.9):
        self.color = (255, 128, 0)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, 'face_detection_yunet_2023mar.onnx')
        self.detector = FaceDetectorYN.create(model_path, "", input_size, score_threshold=threshold, nms_threshold=0.3, top_k=5000)

    def detect_faces(self, image):
        # Set input size before inference
        self.detector.setInputSize(image.shape[1::-1])
        faces = self.detector.detect(image)

        if faces[1] is None:
            return np.array([])
        
        # Only keep the first four columns (x1, y1, w, h)
        faces = faces[1][:, :4]
        # assert faces.shape[1] == 4, "Every row in faces should have 4 columns"
        return faces