import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
# CORE EVALUATION LOOP ENGINE WITH TOP-K METRICS
# =====================================================================
if os.path.exists("my_pet_classifier_model.keras"):
    model = tf.keras.models.load_model("my_pet_classifier_model.keras")
    
    y_true = []
    y_pred = []
    
    # Initialize metric counters
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    total_samples = 0
    
    print("Evaluating model strength across validation dataset stream...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        labels_np = labels.numpy()
        
        y_true.extend(labels_np)
        y_pred.extend(np.argmax(preds, axis=1))
        
        # Calculate Top-k hits per batch
        for i in range(len(labels_np)):
            true_class = labels_np[i]
            # Sort prediction probabilities in ascending order and get top indices
            sorted_indices = np.argsort(preds[i])
            
            # Check matches in top 1, 3, and 5 positions
            if true_class == sorted_indices[-1]:
                top_1_correct += 1
            if true_class in sorted_indices[-3:]:
                top_3_correct += 1
            if true_class in sorted_indices[-5:]:
                top_5_correct += 1
                
            total_samples += 1

    # 1. Print standard text report
    print("\n=== COMPLETE CAPSTONE CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Compute final percentage scores
    top_1_acc = (top_1_correct / total_samples) * 100
    top_3_acc = (top_3_correct / total_samples) * 100
    top_5_acc = (top_5_correct / total_samples) * 100
    
    print(f"\nResults Calculated:")
    print(f" -> Top-1 Accuracy: {top_1_acc:.2f}%")
    print(f" -> Top-3 Accuracy: {top_3_acc:.2f}%")
    print(f" -> Top-5 Accuracy: {top_5_acc:.2f}%")

    # 3. Generate the Model Strength Graph
    print("\nGenerating model strength comparison chart...")
    categories = ['Top-1 Accuracy\n(Exact Match)', 'Top-3 Accuracy\n(In Top 3 Guesses)', 'Top-5 Accuracy\n(In Top 5 Guesses)']
    scores = [top_1_acc, top_3_acc, top_5_acc]
    
    plt.figure(figsize=(8, 5.5))
    # Using a clean professional color palette
    colors = ['#e67e22', '#3498db', '#2ecc71'] 
    bars = plt.bar(categories, scores, color=colors, edgecolor='black', width=0.5)
    
    # Graph styling
    plt.title('Model Soft-Max Strength: Prediction Inclusion Limits', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Accuracy Percentage (%)', fontsize=11, fontweight='bold')
    plt.ylim(0, 115) # Leave room on top for text labels
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Annotate percentage numbers precisely on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    
    output_filename = 'model_strength_top_k.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[SUCCESS] High-resolution evaluation graph saved as: '{output_filename}'")

else:
    print("Error: 'my_pet_classifier_model.keras' weight file not found. Run your training script first.")