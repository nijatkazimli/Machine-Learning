import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, models

script_dir = os.path.dirname(__file__)

dataset_folder = os.path.join(script_dir, "..", "..", "dataset")

# Function to extract age from file name
def extract_age(file_name):
    # Assuming the age is the first part of the file name before '_'
    return int(file_name.split('_')[0])

# Generator function to yield batches of data
def data_generator(batch_size=32):
    while True:
        images = []
        labels = []
        for file_name in os.listdir(dataset_folder):
            if file_name.endswith(".jpg"):
                # Load and preprocess the image
                image_path = os.path.join(dataset_folder, file_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))
                image = img_to_array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                # Extract age from file name
                age = extract_age(file_name)

                images.append(image)
                labels.append(age)

                if len(images) == batch_size:
                    yield (np.vstack(images), np.array(labels))
                    images = []
                    labels = []

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='linear')  # Added activation function
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model using the generator
model.fit_generator(generator=data_generator(), epochs=10, steps_per_epoch=23700//32, validation_steps=100, validation_data=data_generator())