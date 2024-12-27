import streamlit as st
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt_tab')
from model.caption_generator import generate_caption_greedy
from model.caption_generator import generate_caption_beam_search
from model.feature_extractor import extract_features
import os

# Streamlit App Layout
st.title("Image Captioning using CNN and LSTM")
st.write("Upload an image, generate a caption, and compare with a reference using BLEU.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# An optional text input for the user to provide a 'reference caption'
reference_caption = st.text_input("If you have a known ground-truth caption, enter it here:")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Save and process the uploaded image
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Extract features and generate caption
    features = extract_features(image_path)
    search_type = st.radio("Choose search method", ("Greedy", "Beam"))
    if search_type == "Greedy":
        caption = generate_caption_greedy(features)
    else:
        caption = generate_caption_beam_search(features, beam_size=3)
    st.subheader("Generated Caption:")
    st.write(caption)

    # Optionally compute BLEU score if reference caption is provided
    if reference_caption.strip():
        # Tokenize: BLEU expects lists of tokens (words)
        # We do lowercase just to reduce mismatch due to case
        reference_tokens = nltk.word_tokenize(reference_caption.lower())
        candidate_tokens = nltk.word_tokenize(caption.lower())

        # sentence_bleu expects a list of reference lists, hence [[ref_tokens]]
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        
        st.subheader("BLEU Score:")
        st.write(f"{bleu:.4f}")
    else:
        st.info("Provide a reference caption above to see BLEU score.")

    # Clean up temporary image file
    os.remove(image_path)
