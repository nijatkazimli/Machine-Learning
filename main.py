from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet import ResNetModel
from models.face_recognition.handler import OpenCVFaceRecognition
from app.gui import GUIImplementation

def main():
    resnetmodel = ResNetModel()
    haar_front_model = HaarCombinedModel()
    face_recognition = OpenCVFaceRecognition([resnetmodel, haar_front_model])

    gui = GUIImplementation(face_recognition)
    gui.show()

if __name__ == "__main__":
    main()