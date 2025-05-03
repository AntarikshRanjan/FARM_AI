from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the model
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})

# Define your class labels (example)
CLASS_NAMES = ['leaf_blight', 'rust', 'powdery_mildew', 'healthy', 'yellow_spot']

def predict_disease(img_path):
    # Load image and resize
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image according to MobileNetV2's expectations
    img_array = preprocess_input(img_array)

    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    return predicted_class
