from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet_caffee import ResNetCaffeeModel
from models.face_recognition.yunet import YuNetModel
from models.face_recognition.handler import OpenCVFaceHandler
from tensorflow.keras.models import load_model
from app.gui import GUIImplementation

def main():
    age_model = load_model('./models/age_detection/age_detection_model_augmentation_16_30.h5')
    yn_model = YuNetModel()
    face_recognition = OpenCVFaceHandler([yn_model], age_model)

    gui = GUIImplementation(face_recognition)
    gui.show()

if __name__ == "__main__":
    main()