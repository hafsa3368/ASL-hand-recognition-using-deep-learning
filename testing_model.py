import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Model Load
model = tf.keras.models.load_model('sign_language_model.keras')

# 2. Test Image ka path dein (E.g., Test set se koi 'A' ki image)
# Path check
img_path = r"Dataset/test_set/A/1.png"

# 3. Image Preprocessing
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 # Normalization boht zaroori hai

# 4. Prediction
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
prediction = model.predict(img_array)
result = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"\nModel Prediction: {result}")
print(f"Confidence: {confidence:.2f}%")

