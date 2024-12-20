import pickle
from tensorflow.keras.models import load_model

def load_model_and_tokenizer():
    model = load_model("assets/model.keras")
    with open("assets/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    max_length = 34  # Max caption length
    return model, tokenizer, max_length, {}
