from abc import ABC, abstractmethod

class FaceRecognitionModel(ABC):
    @abstractmethod
    def detect_faces(self, image):
        pass

class FaceHandler(ABC):
    def __init__(self, model: FaceRecognitionModel):
        self.model = model

    @abstractmethod
    def handle_image(self, image_path, width, height):
        pass

    @abstractmethod
    def handle_camera(self, cap):
        pass

    @abstractmethod
    def handle_video(self, video_path, width, height):
        pass

class GUI(ABC):
    def __init__(self, handler: FaceHandler):
        self.handler = handler

    @abstractmethod
    def show(self):
        pass