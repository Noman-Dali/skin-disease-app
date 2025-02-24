import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Google Drive file ID
file_id = "1y5KvxnC1KskaVAZKYPv7BJODBL7RIKcJ"
model_path = "model.keras"

# Download model if not already present
if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model
@st.cache_resource
def load_trained_model():
    model = load_model("model.keras", compile=False)  # Ensure model is in the same folder
    return model

model = load_trained_model()

# Define class names (Update with your dataset's labels)
class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma",
               "Melanoma", "Nevus", "Vascular Lesion"]

# Streamlit UI
st.title("Skin Disease Prediction System")
st.write("Upload an image of a skin lesion to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded before using it
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Display image
    st.image(img, caption="Uploaded Image", use_container_width=True)  # ✅ Fixed

    # Preprocess image
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Show result
    st.write(f"### Prediction: {class_names[predicted_class]}")  # ✅ Ensure this runs only if an image is uploaded