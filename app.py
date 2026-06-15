import os
import numpy as np
import tensorflow as tf
import gradio as gr
from sklearn.preprocessing import LabelEncoder

# 1. Re-create class lookup array
DATA_DIR = "./oxford_pet_dataset"
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
raw_labels = [os.path.basename(f).rsplit('_', 1)[0] for f in files]
encoder = LabelEncoder()
encoder.fit(raw_labels)
class_names = encoder.classes_

# 2. Load the trained model architecture
model = tf.keras.models.load_model("my_pet_classifier_model.keras")

# 3. Define prediction function for the web interface
def predict_input_image(input_img):
    if input_img is None:
        return None
        
    # Resize and match MobileNetV2 preprocessing standards
    img = tf.image.resize(input_img, [224, 224])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img_tensor = tf.expand_dims(img, axis=0)
    
    # Generate prediction array
    predictions = model.predict(img_tensor,verbose=0)[0]
    
    # Return dictionary of classes mapped to float probabilities
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# 4. Launch the local web server dashboard
interface = gr.Interface(
    fn=predict_input_image,
    inputs=gr.Image(sources=["upload", "clipboard"],label="Upload or Paste Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Breed Predictions"),
    title="🐾 Core Pet Breed Classifier Engine",
    description="Capstone project deployment prototype using MobileNetV2 Transfer Learning."
)

if __name__ == "__main__":
    interface.launch(share=True)