import tensorflow as tf

# Model load karein
model = tf.keras.models.load_model('sign_language_model.h5')

# Independent Test Set load karein
test_path = r"Dataset/test_set"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(64, 64),
    batch_size=32
)

# Accuracy check karein
results = model.evaluate(test_ds)
print(f"\nIndependent Test Accuracy: {results[1]*100:.2f}%")
print(f"Test Loss: {results[0]:.4f}")