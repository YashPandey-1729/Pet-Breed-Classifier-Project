import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================================
# DATA RESOLUTION UTILITIES (Resolves the "Undeclared" errors)
# =====================================================================
DATA_DIR = "./oxford_pet_dataset"

# 1. Parse files and labels
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
raw_labels = [os.path.basename(f).rsplit('_', 1)[0] for f in files]

# 2. Encode to retrieve class_names
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(raw_labels)
class_names = encoder.classes_

# 3. Partition data splits to isolate the exact 20% validation files
_, x_val, _, y_val = train_test_split(files, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

# 4. Image translation processing function
def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# 5. Build val_ds data pipeline
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# =====================================================================
# CORE EVALUATION LOOP ENGINE
# =====================================================================
if os.path.exists("my_pet_classifier_model.keras"):
    model = tf.keras.models.load_model("my_pet_classifier_model.keras")
    
    y_true = []
    y_pred = []
    
    print("Evaluating model across entire validation dataset stream...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    print("\n=== COMPLETE CAPSTONE CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
else:
    print("Error: 'my_pet_classifier_model.keras' weight file not found. Run your training script first.")