import tensorflow as tf
from tensorflow.keras import models

# 1. Pehle apna model file se load karein
# Agar aapne model .keras extension se save kiya hai to wo likhein
my_model = tf.keras.models.load_model('sign_language_model.keras')

# 2. Dataset load karein (Jo aapne pehle hi kiya hua hai)
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"Dataset/test_set",
    image_size=(64, 64),
    batch_size=32
)

# 3. AB EVALUATE KAREIN (Yahan galti thi)
# 'models.evaluate' ki jagah 'my_model.evaluate' likhein
results = my_model.evaluate(test_ds)

print(f"Test Set Accuracy: {results[1]*100:.2f}%")