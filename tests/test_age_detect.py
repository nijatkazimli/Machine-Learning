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

def load_models(models_dir):
    models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.h5'):
            keras_model = load_model(os.path.join(models_dir, filename))
            model = ModelWrapper(keras_model, filename)
            models.append(model)
    return models

def create_handlers(models):
    handlers = []
    for model in models:
        handler = OpenCVFaceHandler([], model.keras_model)
        handlers.append(handler)
    return handlers

def get_image_files(dataset_dir):
    return [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

def calculate_mse(true_ages, predicted_ages):
    #print(f"true {true_ages}, predicted_ages {predicted_ages}")
    return mean_squared_error(true_ages, predicted_ages)

models_dir = './models/age_detection/'
dataset_dir = './tests/data/age_detection/'

models = load_models(models_dir)

for model in models:
    print(model.name)

handlers = create_handlers(models)
yunet_model = YuNetModel()

# Initialize a dictionary to store true and predicted ages for each model
ages = {model.name: {'true': [], 'predicted': []} for model in models}

for image_file in get_image_files(dataset_dir):
    print(os.path.basename(image_file))
    true_age = int(os.path.basename(image_file).split('_')[0])
    image = cv2.imread(image_file)
    faces = yunet_model.detect_faces(image)

    if len(faces) == 0:
        continue

    for model, handler in zip(models, handlers):
        predicted_ages = handler.age_detection(image, faces)
        if predicted_ages[0] == -1:
            continue
        # Append the true and predicted ages for this image to the respective lists
        print(f"{model.name}: true:{true_age} predicted:{predicted_ages}")
        ages[model.name]['true'].extend([true_age]*len(predicted_ages))
        ages[model.name]['predicted'].extend(predicted_ages)

# After all images have been processed, calculate and print the MSE for each model
for model_name, age_data in ages.items():
    mse = calculate_mse(age_data['true'], age_data['predicted'])
    print(f'RMSE for model {model_name}: {sqrt(mse)}')

    # Calculate the error for each prediction
    errors = [abs(true - pred) for true, pred in zip(age_data['true'], age_data['predicted'])]

    # Plot the error as a function of the true age
    plt.figure(figsize=(10, 6))
    plt.scatter(age_data['true'], errors)
    plt.xlabel('True Age')
    plt.ylabel('Error')
    plt.title(f'Error as a function of True Age for model {model_name}')
    plt.show()