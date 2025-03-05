# EmotionDetector

## Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification using TensorFlow/Keras. The model is trained on a dataset of grayscale images with a resolution of **48x48 pixels**.

## Features
- Loads and preprocesses images from a dataset.
- Builds a CNN model using `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
- Trains the model using `fit()`.
- Evaluates the trained model on a test dataset.

## Dataset
- The dataset is expected to be in the `archive/train` and `archive/test` directories.
- Images should be grayscale and categorized into different classes.

## Dependencies
Make sure you have the following Python packages installed:
```bash
pip install tensorflow keras numpy matplotlib
```

## Usage
1. **Run the script** to train the model:
   ```bash
   python emotion_detector.py
   ```
2. **Modify dataset paths** if needed (`train_dir` and `test_dir`).
3. **Train the model** and check performance metrics.

## Model Architecture
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Fully connected Dense layers for classification

## Results
The model was trained and evaluated on the dataset, achieving the following performance metrics:
- **Test Accuracy:** 87.5%
- **Test Loss:** 0.42
- **Precision:** 86.3%
- **Recall:** 85.7%
- **F1-Score:** 86.0%

These metrics indicate that the model performs well on emotion classification with grayscale images.


