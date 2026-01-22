import tensorflow as tf
import numpy as np

# 1. Model load
model = tf.keras.models.load_model('sign_language_model.keras')

# 2. Dataset load karein (Single image ke bajaye generator use karein)
# Is se preprocessing wahi hogi jo training mein thi
test_ds = tf.keras.utils.image_dataset_from_directory(
    r"Dataset/test_set",
    image_size=(64, 64),
    batch_size=1,
    shuffle=False
)

# 3. On 1 image check prediction
for images, labels in test_ds.take(1):
    # Training rescaling
    predictions = model.predict(images)

    predicted_index = np.argmax(predictions[0])
    actual_index = labels[0].numpy()

    classes = test_ds.class_names
    confidence = np.max(tf.nn.softmax(predictions[0])) * 100

    print(f"\nActual Letter: {classes[actual_index]}")
    print(f"Predicted Letter: {classes[predicted_index]}")
    print(f"Confidence: {confidence:.2f}%")

    # Agar A folder mein 250 images hain, to 250 skip karke pehli image B ki hogi.
    # Hum 300 skip kar dete hain taaki pakka B folder ke andar hon.
    for images, labels in test_ds.skip(300).take(1):
        # Prediction (Rescaling agar model ke andar hai to direct chalayein)
        predictions = model.predict(images)

        predicted_index = np.argmax(predictions[0])
        actual_index = labels[0].numpy()

        classes = test_ds.class_names
        # Softmax confidence ke liye
        confidence = np.max(tf.nn.softmax(predictions[0])) * 100

        print(f"\n--- Testing Class B ---")
        print(f"Actual Letter: {classes[actual_index]}")
        print(f"Predicted Letter: {classes[predicted_index]}")
        print(f"Confidence: {confidence:.2f}%")

        for images, labels in test_ds.skip(300).take(1):
            # 1. Baghair /255 ke predict karein (kyunki isi se A/B sahi aa raha tha)
            predictions = model.predict(images)

            # 2. Predicted index nikalne ke liye softmax ki zaroorat nahi
            predicted_index = np.argmax(predictions[0])

            # 3. Confidence sahi dekhne ke liye Softmax yahan apply karein
            probabilities = tf.nn.softmax(predictions[0])
            confidence = np.max(probabilities) * 100

            print(f"\nActual: {test_ds.class_names[labels[0].numpy()]}")
            print(f"Predicted: {test_ds.class_names[predicted_index]}")
            print(f"Confidence: {confidence:.2f}%")

            import tensorflow as tf
            import numpy as np

            # 1. Model load
            model = tf.keras.models.load_model('sign_language_model.keras')

            # 2. Dataset load
            test_ds = tf.keras.utils.image_dataset_from_directory(
                r"Dataset/test_set",
                image_size=(64, 64),
                batch_size=1,
                shuffle=False
            )

            # 3. Class D par janay ke liye skip karein (A, B, C ko guzarna hai)
            # Agar har class ki 250 images hain, to 750+ skip karne se D shuru hoga
            for images, labels in test_ds.skip(800).take(1):
                # Prediction (Baghair /255.0 ke, jaisa pichli bar work kiya)
                predictions = model.predict(images)

                predicted_index = np.argmax(predictions[0])
                actual_index = labels[0].numpy()

                classes = test_ds.class_names
                # Softmax for confidence calculation
                confidence = np.max(tf.nn.softmax(predictions[0])) * 100

                print(f"\n--- Testing Class D ---")
                print(f"Actual Letter: {classes[actual_index]}")
                print(f"Predicted Letter: {classes[predicted_index]}")
                print(f"Confidence: {confidence:.2f}%")