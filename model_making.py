import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Data Loading with Validation Split
train_path = r"Dataset/training_set"
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(64, 64),
    batch_size=32
)

# 2. Improved Model (Normalization with  Dropout)
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Rescaling(1. / 255),  # Pixels ko normalize karna zaroori hai

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  # Overfitting se bachne ke liye
    layers.Dense(26, activation='softmax')  # 26 alphabets ke liye
])

# 3. Optimizer aur Loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Training
model.fit(train_ds, epochs=10)

# 5. Save in native Keras format
model.save('sign_language_model.keras')