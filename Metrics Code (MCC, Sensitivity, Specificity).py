import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, matthews_corrcoef

# 1. Model Loading
my_model = tf.keras.models.load_model('sign_language_model.keras')

# 2. Dataset Loading
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"Dataset/test_set",
    image_size=(64, 64),
    batch_size=32,
    shuffle=False  # for Metrics: shuffle=False(must include)
)

# 3. Predictions
print("Generating predictions for metrics...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
predictions = my_model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# 4. Confusion Matrix aur Metrics Calculation
cm = confusion_matrix(y_true, y_pred)
classes = test_ds.class_names

print("\n--- Detailed Metrics per Class ---")
for i in range(len(classes)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sn
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Sp

    print(f"Class {classes[i]}: Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

# 5. Overall MCC
mcc = matthews_corrcoef(y_true, y_pred)
print(f"\nOverall Matthews Correlation Coefficient (MCC): {mcc:.4f}")
# 6.  EVALUATE
results = my_model.evaluate(test_ds)

print(f"Test Set Accuracy: {results[1]*100:.2f}%")