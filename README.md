ASL Hand Recognition using Deep Learning
This project implements a Deep Learning model to recognize American Sign Language (ASL) alphabets (A-Z) with high precision. The model was trained on a large-scale dataset and achieved nearly 99% accuracy on the test set.

🚀 Performance Summary
Training Accuracy: 99.8%

Independent Test Accuracy: ~99%

Input Size: 64x64 RGB images

Classes: 26 (A to Z)

🛠️ Tech Stack
Language: Python 3.12

Framework: TensorFlow / Keras

Library:
Python 3.12: Main programming language.

TensorFlow: For building and training the CNN model.

Keras: High-level API used for model architecture and evaluation.

NumPy: Used for numerical operations and array handling.
📂 Project Structure
Data_coll.py: Script for collecting or preparing the dataset.

model_making.py: The CNN architecture and training pipeline.

testing_model.py: Script to evaluate the model on an independent test set.

sign_language_model.keras: The final trained model file.

💡 Key Features
Normalization: Uses rescaling layers to normalize pixel values (0-1), significantly improving convergence.

Dropout Regularization: Prevents overfitting by randomly dropping neurons during training.

Independent Testing: Verified accuracy using a separate test_set to ensure real-world reliability.
