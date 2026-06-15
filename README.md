# Core Pet Breed Classifier Engine

### 🐾 Internship Capstone Project | Production Prototype
An end-to-end deep learning system designed to classify 37 distinct breeds of dogs and cats with high precision. This system utilizes Transfer Learning with a frozen Google MobileNetV2 backbone and features a decoupled web prototype for real-time testing.

## 📊 Performance Summary
* **Validation Accuracy:** 91% (across 1,478 unseen validation images)
* **Macro Average F1-Score:** 0.91
* **Weighted Average F1-Score:** 0.91
* **Architecture:** MobileNetV2 (Feature Extractor) + Custom Dense/Dropout Classification Head

## 📂 Project Architecture
```text
Pet_Classifier_Capstone/
│
├── oxford_pet_dataset/           <-- Source training data
├── my_pet_classifier_model.keras <-- Pre-trained model weights
│
├── pet_classifier.py            <-- Pipeline execution & training loop
├── evaluate.py                  <-- Automated metrics & classification reporting
├── predict.py                   <-- Isolated local CLI inference script
├── app.py                       <-- Gradio production web dashboard
└── requirements.txt             <-- Environment dependency manifest