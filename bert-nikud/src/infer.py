"""
Inference script for Hebrew Nikud BERT model.

This module handles loading the trained model and generating nikud predictions.
"""

import torch
from pathlib import Path
from typing import List, Optional
from safetensors.torch import load_file
import re

from model import HebrewNikudModel
from decode import reconstruct_text_from_predictions
from tokenizer_utils import load_tokenizer


class NikudPredictor:
    """Predictor class for adding nikud to Hebrew text."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on (None for auto-detect)
            tokenizer_path: Tokenizer path or name (required)
        """
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.device = device
        print(f"Loading model on device: {device}")

        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)

        # Load model
        self.model = HebrewNikudModel()

        # Load checkpoint weights
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.is_dir():
            checkpoint_file = checkpoint_file / "model.safetensors"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        self.model.load_state_dict(load_file(checkpoint_file, device=str(device)))

        self.model.to(device)
        self.model.eval()

        print("Model loaded successfully!")

    def predict(self, text: str) -> str:
        """
        Add nikud marks to Hebrew text.

        Args:
            text: Plain Hebrew text without nikud (can contain mixed content)

        Returns:
            Text with predicted nikud marks on Hebrew letters, other characters preserved
        """
        # Don't filter! Keep the original text as-is
        if not text.strip():
            return text

        # Remove diacritics from the text
        text = re.sub(r"[\u0590-\u05CF]", "", text)

        # Tokenize the full text (tokenizer will handle all characters)
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0]

        # Get predictions (returns dict with vowel, dagesh, sin, stress)
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)

        # Reconstruct text with nikud using shared function
        nikud_text = reconstruct_text_from_predictions(
            text,
            input_ids[0],
            offset_mapping,
            predictions["vowel"][0],
            predictions["dagesh"][0],
            predictions["sin"][0],
            predictions["stress"][0],
            predictions["prefix"][0],
            self.tokenizer,
        )

        return nikud_text

    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Add nikud marks to multiple Hebrew texts.

        Args:
            texts: List of plain Hebrew texts

        Returns:
            List of texts with predicted nikud marks
        """
        return [self.predict(text) for text in texts]


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict nikud for Hebrew text")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו!",
        help="Text to add nikud to",
    )
    parser.add_argument(
        "--file", type=str, default=None, help="File containing text to add nikud to"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="dicta-il/dictabert-large-char-menaked",
        help="Tokenizer path or name",
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = NikudPredictor(
        args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer,
    )

    # Get input text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [args.text]

    # Generate predictions
    print("\n" + "=" * 80)
    print("Predictions:")
    print("=" * 80)

    if args.file:
        # Print clean output to stdout
        for text in texts:
            nikud_text = predictor.predict(text)
            print(nikud_text)
    else:
        for i, text in enumerate(texts):
            nikud_text = predictor.predict(text)
            print(f"\nInput:  {text}")
            print(f"Output: {nikud_text}")

            if i < len(texts) - 1:
                print("-" * 80)


if __name__ == "__main__":
    main()
