import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.model_loader import load_model_and_tokenizer
from model.text_preprocessing import idx_to_word

# Load model and tokenizer
model, tokenizer, max_length, features_dict = load_model_and_tokenizer()

def generate_caption(features):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([features, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()
