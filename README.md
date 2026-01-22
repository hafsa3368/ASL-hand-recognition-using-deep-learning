# Sign Language Alphabet Classification Using Deep Learning

This project implements a deep learning based image classification system to recognize hand sign alphabets (A–Z) from sign language images. The goal is to accurately classify 26 hand gesture classes by designing a robust Convolutional Neural Network (CNN) and applying correct preprocessing and evaluation strategies.

## Project Overview

The system is trained on a large-scale dataset containing **45,500 images** belonging to **26 classes (A–Z)**.  
The model learns fine-grained visual features such as finger positions, hand shapes, and contours in order to distinguish between visually similar signs like **A and E**.

The project focuses on:
- Designing an improved CNN architecture  
- Applying correct normalization and preprocessing  
- Avoiding overfitting using dropout  
- Evaluating performance using balanced metrics  

## Tools and Technologies

- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  

## Dataset
URL: https://github.com/shadabsk/Sign-Language-Recognition-Using-Hand-Gestures-Keras-PyQT5-OpenCV.git
- Total images: **45,500**  
- Number of classes: **26 (A–Z)**  
- Directory structure is organized by class labels  
- Images are resized to a fixed input size before training  

## Model Architecture

The final model includes:
- Multiple convolutional layers for feature extraction  
- MaxPooling layers for spatial downsampling  
- Dropout layers to reduce overfitting  
- Fully connected layers with Softmax activation for multi-class classification  

This architecture enables the model to capture subtle visual differences between similar hand signs.

## Data Preprocessing

- Images are resized to a fixed resolution  
- Pixel normalization is applied consistently with the model design  
- Data augmentation (rotation, zoom, shifts) is used to improve generalization  

## Evaluation Metrics

The model is evaluated using:
- Accuracy  
- Confusion Matrix  
- Sensitivity and Specificity  
- Matthews Correlation Coefficient (MCC)  

These balanced metrics ensure that the model performs well across all classes and does not collapse to a single-label prediction.

## Key Challenge: The Normalization Paradox

One major challenge in this project was related to **input normalization**.

When standard **/255.0 normalization** was applied during single-image testing, the model’s confidence dropped to around **3–4%**, and incorrect predictions were produced (for example, repeatedly predicting only 'M').

After investigation, it was discovered that:
- The model was already optimized to work on **raw pixel values**, or  
- A **built-in Rescaling layer** was present inside the model architecture.

When manual scaling was removed, the model again achieved **around 99% accuracy** with correct predictions (A, B, D, etc.).

This experiment highlighted the importance of maintaining **consistent preprocessing between training and inference**.

<img width="458" height="167" alt="image" src="https://github.com/user-attachments/assets/e9c03d9d-6e1a-402f-b67c-d2a166c5e33c" />

