#!/usr/bin/env python3
"""Verify training data alignment with menaked tokenizer."""

from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parents[2] / "bert-nikud"
sys.path.insert(0, str(base_dir / "src"))

from dataset import prepare_training_data
from tokenizer_utils import load_tokenizer


def main():
    tokenizer_dir = base_dir / "tokenizer" / "dictabert-large-char-menaked"
    tokenizer = load_tokenizer(str(tokenizer_dir))

    sample = "הָאִישׁ רָצָה"
    data = prepare_training_data(sample, tokenizer)

    print("=" * 70)
    print("ALIGNMENT CHECK")
    print("=" * 70)
    print(f"Sample: {sample}")
    print(f"Plain:  {data['plain_text']}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(data['input_ids'])}")
    print(f"Offsets: {data['offset_mapping']}")
    print(f"Token count: {len(data['input_ids'])}")
    print(f"Expected: {len(data['plain_text']) + 2}")


if __name__ == "__main__":
    main()
