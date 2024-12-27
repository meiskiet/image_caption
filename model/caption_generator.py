import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.model_loader import load_model_and_tokenizer
from model.text_preprocessing import idx_to_word

# Load model and tokenizer
model, tokenizer, max_length, features_dict = load_model_and_tokenizer()

def generate_caption_greedy(features):
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


def generate_caption_beam_search(features, beam_size=3):
    """
    Generate a caption using beam search.
    
    :param features: Extracted image features from the CNN (shape [1, feature_dim]).
    :param beam_size: How many candidate sequences to keep at each step.
    :return: The best caption string.
    """

    # Start with a single sequence: ["startseq"], score = 0
    start_seq = ["startseq"]
    # We store (sequence_of_words, cumulative_log_prob)
    beam = [(start_seq, 0.0)]
    
    # This will store completed sequences that reach "endseq"
    completed_sequences = []

    for _ in range(max_length):
        # A temporary list for all possible expansions of the current beam
        all_candidates = []

        # Expand each candidate in the beam
        for seq, score in beam:
            # If the sequence already ends with 'endseq', add it to completed and skip
            if seq[-1] == "endseq":
                completed_sequences.append((seq, score))
                continue

            # Convert sequence to token indices
            in_text = " ".join(seq)
            sequence_indices = tokenizer.texts_to_sequences([in_text])[0]
            sequence_indices = pad_sequences([sequence_indices], maxlen=max_length)

            # Predict next word probabilities
            y_pred = model.predict([features, sequence_indices], verbose=0)[0]  # shape = [vocab_size]

            # Get the top beam_size probabilities and their indices
            # argsort returns ascending order, so we do [-beam_size:]
            top_indices = np.argsort(y_pred)[-beam_size:]
            
            # Create new candidate sequences
            for idx in top_indices:
                predicted_word = idx_to_word(idx, tokenizer)
                if predicted_word is None:
                    # If idx_to_word can’t find the word (unlikely with a proper vocab), skip
                    continue
                
                # Calculate new score (add log probability)
                prob = y_pred[idx]
                new_score = score + np.log(prob + 1e-10)  # +1e-10 to avoid log(0)
                
                new_seq = seq + [predicted_word]
                all_candidates.append((new_seq, new_score))
        
        # If we have no candidates (e.g., everything ended), break early
        if not all_candidates:
            break
        
        # Sort candidates by score in descending order
        all_candidates.sort(key=lambda tup: tup[1], reverse=True)
        
        # Select the top k for the next round
        beam = all_candidates[:beam_size]

    # Add any beam sequences that haven’t ended yet into completed
    for seq, score in beam:
        if seq[-1] == "endseq":
            completed_sequences.append((seq, score))

    # If no sequence ended with "endseq", just pick the best from the beam
    if len(completed_sequences) == 0:
        completed_sequences = beam

    # Sort all completed sequences by their scores (descending) and pick the best
    completed_sequences.sort(key=lambda tup: tup[1], reverse=True)
    best_sequence, best_score = completed_sequences[0]

    # Convert to a readable caption, removing start/end tokens
    caption = " ".join(best_sequence)
    caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return caption