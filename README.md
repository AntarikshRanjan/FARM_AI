# Farmai

**Farmai** is an AI-powered platform that assists farmers by diagnosing plant diseases or pests from photos or live camera feeds and provides natural, easy-to-make remedies, considering the current environment (like humidity and weather). This solution aims to empower agriculture with accessible technology.

---

## 📌 Overview

Farmai combines computer vision, machine learning, and generative AI to:
- Identify plant diseases/pests using images.
- Factor in environmental conditions (location, temperature, humidity).
- Generate simple, local-language remedies.
- Work with embedded devices for real-time, offline diagnosis.

---

## ⚙️ Features

| Feature | Description |
|--------|-------------|
| Image-Based Diagnosis | Upload photo or use live camera for detection. |
| AI Model | Trained CNN for classifying 5 disease/pest types. |
| Live Environment Data | Weather, temperature, and humidity included. |
| GPT/Gemini Integration | Suggests natural remedies using LLMs. |
| Hardware Support | ESP32 / Raspberry Pi for real-time use. |
| Snapshot System | Extracts 1 frame per 5 for efficient analysis. |
| Offline Deployment | TFLite model enables mobile or field use. |

---

## 🧠 Machine Learning Model Training

### 📁 Dataset

- **Classes**: Leaf Blight, Powdery Mildew, Leaf Spot, Aphid Infestation, Healthy Leaf
- **Images**: 100+ per class, augmented for variety
- **Sources**: Public datasets + custom field photos

### 🔄 Preprocessing

- Resizing: 224x224
- Normalization: [0, 1] range
- Augmentations: Rotation, Flip, Zoom, Brightness

### 🧪 Model Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')
])
