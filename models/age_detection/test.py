import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained age detection model
age_model = load_model('age_detection_model_augmentation_16_30.h5')

# Load the OpenCV face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray_frame[y:y + h, x:x + w]

        # Preprocess the face image
        face_image = cv2.resize(face_roi, (128, 128))  # Resize to match the expected input shape
        face_image = np.expand_dims(face_image, axis=-1)  # Add a channel dimension
        face_image = face_image / 255.0  # Normalize to [0, 1]
        face_image = np.expand_dims(face_image, axis=0)  # Add a batch dimension

        # Predict the age using the age detection model
        predicted_age = age_model.predict(face_image)[0][0]

        # Display the predicted age on the frame
        cv2.putText(frame, f'Age: {int(predicted_age)} years', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with age predictions
    cv2.imshow('Age Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
