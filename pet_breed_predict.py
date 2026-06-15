import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# 1. Re-create your class lookup dictionary (must match training order)
DATA_DIR = "./oxford_pet_dataset"
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
raw_labels = [os.path.basename(f).rsplit('_', 1)[0] for f in files]
encoder = LabelEncoder()
encoder.fit(raw_labels)
class_names = encoder.classes_

# 2. Load your pre-trained model instantly
model = tf.keras.models.load_model("my_pet_classifier_model.keras")

# 3. Define the quick prediction tool
def predict_pet_breed(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    img_tensor = tf.expand_dims(img, axis=0)
    predictions = model.predict(img_tensor)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    predicted_breed = class_names[predicted_idx]
    
    print(f"\nPredicted Breed: {predicted_breed} ({confidence:.2f}% Confidence)")

# 4. Run it on any photo you want!

predict_pet_breed("Pomeranian.jpg")
predict_pet_breed("Pug.jpg")