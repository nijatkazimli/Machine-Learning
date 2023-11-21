from models.face_recognition.haar_fast import HaarFaceModel
from models.face_recognition.handler import OpenCVFaceRecognition
from app.gui import GUIImplementation

def main():
    model = HaarFaceModel()
    face_recognition = OpenCVFaceRecognition(model)

    gui = GUIImplementation(face_recognition)
    gui.show()  # Start the GUI

if __name__ == "__main__":
    main()