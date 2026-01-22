import tensorflow as tf
import os

#  path
train_path = r"Dataset/training_set"

print("Checking directory:", os.path.abspath(train_path))

try:
    # Images loading
    train = tf.keras.utils.image_dataset_from_directory( train_path,image_size=(64, 64),
        batch_size=32,
        label_mode='int'
    )

    # Check image found or no
    if len(train.file_paths) == 0:
        print("Error: Folder found but images (jpg/png) no found!")
    else:
        print("Success! Found", len(train.file_paths), "images.")
        print("Classes:", train.class_names)

except Exception as e:
    print("Error loading data:", e)
