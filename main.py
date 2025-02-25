import torch
import argparse
import math
import pandas as pd
from Encoder import TransformerCharEncoder
from GlossingModel import MorphemeGlossingModel
from MorphemeSegmenter import MorphemeSegmenter


torch.manual_seed(42)

def segment_text(segmentation_model, sentence):
    """Segment a sentence into morphemes if needed."""
    return segmentation_model(sentence)


class DummyBatch:
    def __init__(self, sentences, sentence_lengths):
        self.sentences = sentences
        self.sentence_lengths = sentence_lengths
        # Other attributes required by the forward method
        self.word_extraction_index = torch.arange(sentences.size(1)).unsqueeze(0).expand(sentences.size(0), -1)
        self.word_lengths = sentence_lengths
        self.word_target_lengths = torch.tensor([1] * sentences.size(0), dtype=torch.long)  # Placeholder for number of morphemes

        


def tokenize_sentence(sentence, vocab_size):
    """Tokenize a sentence into a list of character indices and ensure they are within the vocab size."""
    return [ord(char) % vocab_size for char in sentence if isinstance(char, str)]

def predict_gloss(encoder, glossing_model, sentence, translation):
    """Encode input language and translation, pass through glossing model, and return gloss prediction."""
    vocab_size = encoder.embedding.num_embeddings  # Get the vocab size from the embedding layer

    # Tokenize the input sentence and translation
    input_tokens = tokenize_sentence(sentence, vocab_size)
    translation_tokens = tokenize_sentence(translation, vocab_size)
    combined_tokens = input_tokens + translation_tokens
    sentence_lengths = torch.tensor([len(combined_tokens)], dtype=torch.long)

    # Convert tokens to tensor and add batch dimension
    combined_tokens_tensor = torch.tensor(combined_tokens, dtype=torch.long).unsqueeze(0)

    # Create a dummy batch object
    dummy_batch = DummyBatch(combined_tokens_tensor, sentence_lengths)

    # Switch the model to evaluation mode
    glossing_model.eval()  # Set the model to evaluation mode

    # Compute gloss using the glossing model
    gloss_output = glossing_model(dummy_batch, training=False)
    gloss = gloss_output["morpheme_scores"]

    # Debugging steps - output the token numbers to see if they are being inputted correctly
    #print(f"Input tokens: {input_tokens}")
    #print(f"Translation tokens: {translation_tokens}")
    #print(f"Combined tokens: {combined_tokens}")
    return gloss





def train_and_evaluate(csv_path, encoder, glossing_model, segmentation_model=None):
    """Train the model on a dataset and evaluate with a test example."""
    data = pd.read_csv(csv_path)
    print(data.columns)  # Print column names to verify

    for index, row in data.iterrows():
        input_sentence = str(row['Language'])  # Convert to string
        translation = str(row['Translation'])  # Convert to string
        reference_gloss = row['Gloss']    # Adjust column name

        # Apply segmentation only for Track 1 data
        if segmentation_model and "track 1" in csv_path.lower():
            input_sentence = segment_text(segmentation_model, input_sentence)

        predicted_gloss = predict_gloss(encoder, glossing_model, input_sentence, translation)
        print(f"Input: {input_sentence}")
        print(f"Translation: {translation}")
        print(f"Predicted Gloss: {predicted_gloss}")
        print(f"Reference Gloss: {reference_gloss}\n")

        #debugging step
        #predicted_gloss = predict_gloss(encoder, glossing_model, input_sentence, translation)
        #print(f"Predicted Gloss (before return): {predicted_gloss}")




def main(csv_path, segmented):
    """Main function to load models, train, and evaluate."""
    encoder = TransformerCharEncoder(vocab_size=100)
    glossing_model = MorphemeGlossingModel(source_alphabet_size=100, target_alphabet_size=50)
    
    segmentation_model = None
    if not segmented:
        segmentation_model = MorphemeSegmenter(hidden_size=256)
    
    train_and_evaluate(csv_path, encoder, glossing_model, segmentation_model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset")
    parser.add_argument("--segmented", action='store_true', help="Indicate if input is already segmented")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.csv, args.segmented)
