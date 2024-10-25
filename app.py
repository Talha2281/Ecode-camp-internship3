import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('cnn.keras')  # Make sure the filename matches your model

# Function for model prediction
def predict_image(img):
    img = img.resize((32, 32))  # CIFAR-10 image size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title("CIFAR-10 Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    st.write("Prediction:", prediction)
