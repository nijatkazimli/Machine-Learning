from math import sqrt
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from models.face_recognition.yunet import YuNetModel
from models.face_recognition.handler import OpenCVFaceHandler
import matplotlib.pyplot as plt

class ModelWrapper:
    def __init__(self, keras_model, name):
        self.keras_model = keras_model
        self.name = name

def load_one_model(model_path):
    keras_model = load_model(model_path)
    model = ModelWrapper(keras_model, os.path.basename(model_path))
    return model

def get_image_files(dataset_dir):
    return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

def calculate_mse(true_ages, predicted_ages):
    return mean_squared_error(true_ages, predicted_ages)

model_path = './models/age_detection/age_detection_model_augmentation_16_30.h5'
dataset_dir = './tests/data/age_detection/'

model = load_one_model(model_path)
handler = OpenCVFaceHandler([], model.keras_model)
yunet_model = YuNetModel()

# Initialize a dictionary to store true and predicted ages
ages = {'true': [], 'predicted': []}

for image_file in get_image_files(dataset_dir):
    true_age = int(os.path.basename(image_file).split('_')[0])
    image = cv2.imread(image_file)
    faces = yunet_model.detect_faces(image)

    if len(faces) == 0:
        continue

    predicted_ages = handler.age_detection(image, faces)
    if predicted_ages[0] == -1:
        continue

    # Append the true and predicted ages for this image to the respective lists
    ages['true'].extend([true_age]*len(predicted_ages))
    ages['predicted'].extend(predicted_ages)

# After all images have been processed, calculate and print the MSE
mse = calculate_mse(ages['true'], ages['predicted'])
print(f'RMSE for model {model.name}: {sqrt(mse)}')

# Calculate the error for each prediction
errors = [abs(true - pred) for true, pred in zip(ages['true'], ages['predicted'])]

# Plot the error as a function of the true age
plt.figure(figsize=(10, 6))
plt.scatter(ages['true'], errors)
plt.xlabel('True Age')
plt.ylabel('Error')
plt.title(f'Error as a function of True Age for model {model.name}')
plt.show()