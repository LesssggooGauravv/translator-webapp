"""
translate.py - Inference Script for Seq2Seq English-Hindi Translator

Loads the trained model checkpoint and translates English text to Hindi.

Usage (standalone):
    cd translator_webapp
    python -m ml.translate "hello how are you"

Usage (as module - called by backend):
    from ml.translate import Translator
    translator = Translator()
    result = translator.translate("hello how are you")
"""

import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.model import build_model
from ml.dataset import Vocabulary, tokenize_en, SOS_IDX, EOS_IDX


class Translator:
    """
    Loads trained Seq2Seq model and performs greedy decoding for translation.
    
    This class is designed to be instantiated ONCE (e.g., at server startup)
    and reused for multiple translation requests.
    """

    def __init__(self, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found at {config_path}. "
                "Please run 'python -m ml.train' first to train the model."
            )

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load vocabularies
        print("Loading vocabularies...")
        self.src_vocab = Vocabulary.load(os.path.join(checkpoint_dir, "src_vocab.json"))
        self.trg_vocab = Vocabulary.load(os.path.join(checkpoint_dir, "trg_vocab.json"))
        print(f"  Source vocab: {len(self.src_vocab)} tokens")
        print(f"  Target vocab: {len(self.trg_vocab)} tokens")

        # Build model with same architecture
        print("Loading model weights...")
        self.model = build_model(
            src_vocab_size=len(self.src_vocab),
            trg_vocab_size=len(self.trg_vocab),
            device=self.device,
            embed_size=self.config.get("embed_size", 256),
            hidden_size=self.config.get("hidden_size", 256),
            num_layers=self.config.get("num_layers", 1),
            dropout=0.0,  # No dropout during inference
        )

        # Load saved weights
        weights_path = os.path.join(checkpoint_dir, "seq2seq_model.pth")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model loaded successfully!")

    @torch.no_grad()
    def translate(self, text: str, max_len: int = 50) -> str:
        """
        Translate an English sentence to Hindi using greedy decoding.

        Args:
            text: English input sentence
            max_len: Maximum number of tokens to generate

        Returns:
            Translated Hindi string
        """
        # Tokenize and encode the source sentence
        tokens = tokenize_en(text)
        src_indices = self.src_vocab.encode(tokens)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)

        # Encode the source
        encoder_outputs, hidden, cell = self.model.encoder(src_tensor)

        # Start decoding with <sos> token
        input_token = torch.tensor([SOS_IDX], dtype=torch.long).to(self.device)

        output_indices = []

        for _ in range(max_len):
            prediction, hidden, cell, _ = self.model.decoder(
                input_token, hidden, cell, encoder_outputs
            )

            # Greedy: pick the token with highest probability
            top1 = prediction.argmax(1)
            predicted_idx = top1.item()

            # Stop if <eos> is generated
            if predicted_idx == EOS_IDX:
                break

            output_indices.append(predicted_idx)
            input_token = top1

        # Decode indices back to Hindi text
        output_tokens = self.trg_vocab.decode(output_indices)
        return " ".join(output_tokens)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ml.translate \"your english sentence here\"")
        print("Example: python -m ml.translate \"hello how are you\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    print(f"\nInput (EN):  {input_text}")

    translator = Translator()
    result = translator.translate(input_text)
    print(f"Output (HI): {result}")


if __name__ == "__main__":
    main()
