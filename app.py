import streamlit as st
from PIL import Image
from model.caption_generator import generate_caption
from model.feature_extractor import extract_features
import os

# Streamlit App Layout
st.title("Image Captioning using CNN and LSTM")
st.write("Upload an image, and the model will generate a descriptive caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Save and process the uploaded image
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Extract features and generate caption
    features = extract_features(image_path)
    caption = generate_caption(features)
    st.subheader("Generated Caption:")
    st.write(caption)

    # Clean up temporary image file
    os.remove(image_path)
