import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =====================================================================
# STEP 1: INGESTION & LABEL PARSING
# =====================================================================
def parse_filepaths_and_raw_labels(data_dir):
    extensions = ('.jpg', '.jpeg', '.png')
    
    # Isolate valid image paths from the folder
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(extensions)]
    
    # Extract breed text using a right-side split to handle multiple underscores safely
    raw_labels = [os.path.basename(f).rsplit('_', 1)[0] for f in files]
    
    return files, raw_labels

# =====================================================================
# STEP 2: NUMERICAL LABEL ENCODING
# =====================================================================
def encode_labels(raw_labels):
    encoder = LabelEncoder()
    # Converts text strings to sequential integers sorted alphabetically
    encoded_labels = encoder.fit_transform(raw_labels)
    class_names = encoder.classes_
    
    return encoded_labels, class_names

# =====================================================================
# STEP 3: VISUAL EXPLORATION (SANITY CHECK)
# =====================================================================
def plot_class_distribution(labels, class_names):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 5))
    plt.bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title("Class Distribution Check (Detecting Imbalances)")
    plt.tight_layout()
    plt.show()

def display_samples(files, labels, class_names):
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        # Grab a completely random index to preview the dataset variety
        idx = np.random.randint(0, len(files))
        
        # Load and decode image binary payload into RGB pixel values
        img = tf.image.decode_image(tf.io.read_file(files[idx]), channels=3)
        
        plt.imshow(img.numpy())
        plt.title(class_names[labels[idx]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# =====================================================================
# STEP 4: OPTIMIZED STREAMING PIPELINE (tf.data)
# =====================================================================
def split_dataset(files, labels):
    # Splits the arrays into 80% train and 20% validation splits evenly across classes
    return train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

def parse_image(filename, label):
    # On-the-fly image operations execution loop
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    # Normalizes pixel spaces to [-1.0, 1.0] to match MobileNetV2 standards
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def build_data_pipeline(files, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(files))
    
    # Asynchronously processes images across available CPU cores
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Stacks items into batches and prefetches next data block ahead of the GPU
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# =====================================================================
# STEP 5: TRANSFER LEARNING & TRAINING
# =====================================================================
def create_transfer_model(num_classes):
    # Load Google's pre-trained body without its 1000-class output head
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    # Freeze layers so the pre-trained weights remain unchanged/read-only
    base_model.trainable = False
    
    # Assembly lines pairing base features with custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # Squeezes 7x7x1280 matrix to 1280 vector
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),             # Randomly disconnects 30% nodes to stop overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =====================================================================
# APPLICATION EXECUTION ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    # CHANGE THIS to your extracted dataset path
    DATA_DIR = "./oxford_pet_dataset" 
    
    if os.path.exists(DATA_DIR):
        # 1. Run Data Ingestion
        paths, raw_labels = parse_filepaths_and_raw_labels(DATA_DIR)
        
        # 2. Run Alphabetical Integer Encoding
        labels, classes = encode_labels(raw_labels)
        
        # 3. Run Profiling & Visualization Windows
        print("Displaying dataset diagnostic charts...")
        plot_class_distribution(labels, classes)
        display_samples(paths, labels, classes)
        
        # 4. Construct Partition Streams
        x_train, x_val, y_train, y_val = split_dataset(paths, labels)
        train_ds = build_data_pipeline(x_train, y_train, shuffle=True)
        val_ds = build_data_pipeline(x_val, y_val, shuffle=False)
        
        # 5. Initialize network and trigger execution fitting loop
        model = create_transfer_model(len(classes))
        model.summary()
        
        print("\nLaunching optimization execution loops...")
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=10
        )
        print("\nModel training lifecycle complete!")

        print("Saving the smart model...")
        model.save("my_pet_classifier_model.keras")

    else:
        print(f"Error: The directory folder path '{DATA_DIR}' does not exist.")
