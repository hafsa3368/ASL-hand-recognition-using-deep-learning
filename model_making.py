import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Data load karein (Ensure variable name is 'train_data')
train_path = r"Dataset/training_set"
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(64, 64),
    batch_size=32,
    label_mode='int'
)


# 2. Model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 3. training (Using 'train_data')
model.fit(train_data, epochs=30)

# 4. Model save karein
model.save('sign_language_model.h5')
print("Model trained and saved!")