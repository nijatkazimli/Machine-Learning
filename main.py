from models.face_recognition.haar_fast import HaarFastModel
from models.face_recognition.handler import OpenCVFaceRecognition
from app.gui import GUIImplementation

def main():
    model = HaarFastModel()
    face_recognition = OpenCVFaceRecognition(model)

    gui = GUIImplementation(face_recognition)
    gui.show()

if __name__ == "__main__":
    main()