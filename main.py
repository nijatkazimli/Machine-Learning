from models.face_recognition.haar import HaarFastModel, HaarSlowModel
from models.face_recognition.handler import OpenCVFaceRecognition
from app.gui import GUIImplementation

def main():
    model = HaarSlowModel()
    face_recognition = OpenCVFaceRecognition(model)

    gui = GUIImplementation(face_recognition)
    gui.show()

if __name__ == "__main__":