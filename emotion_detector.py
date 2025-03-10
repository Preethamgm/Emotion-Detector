# -*- coding: utf-8 -*-
"""Emotion Detector.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GnIXQT9qtfO2qozcRhZxTL8awUBoBZwl

Load and Preprocess Images
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = "archive/train"
test_dir = "archive/test"

# Create ImageDataGenerator for loading and augmenting images
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load train and test datasets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

"""Define the CNN Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""Train the Model"""

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=test_data.samples // test_data.batch_size
)

"""Evaluate the Model"""

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy}")

Predict on New Images

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model
#model = ...  # Load your trained model here
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face ROI
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match model input size
        face = face.astype('float32') / 255.0  # Normalize pixel values
        face = img_to_array(face)  # Convert to array
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict the emotion
        predictions = model.predict(face)
        emotion = emotion_labels[np.argmax(predictions)]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame with bounding boxes and emotion labels
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

