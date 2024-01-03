from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet_caffee import ResNetCaffeeModel
from models.face_recognition.yunet import YuNetModel
from models.face_recognition.handler import OpenCVFaceRecognition
from app.gui import GUIImplementation

def main():
    resnetmodel = ResNetCaffeeModel()
    haar_front_model = HaarFrontModel()
    haar_profile_model = HaarProfileModel()
    yn_model = YuNetModel()
    face_recognition = OpenCVFaceRecognition([resnetmodel, yn_model])

    gui = GUIImplementation(face_recognition)
    gui.show()

if __name__ == "__main__":
    main()