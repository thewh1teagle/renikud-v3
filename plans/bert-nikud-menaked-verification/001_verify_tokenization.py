#!/usr/bin/env python3
"""Verify menaked tokenizer offsets and UNK behavior."""

from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parents[2] / "bert-nikud"
sys.path.insert(0, str(base_dir / "src"))

from dataset import tokenize_with_offsets
from tokenizer_utils import load_tokenizer


def main():
    tokenizer_dir = base_dir / "tokenizer" / "dictabert-large-char-menaked"
    tokenizer = load_tokenizer(str(tokenizer_dir))

    print("=" * 70)
    print("MENAKED TOKENIZER CHECK")
    print("=" * 70)
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Class: {type(tokenizer).__name__}")
    print(f"Is fast: {tokenizer.is_fast}")
    print(f"Vocab size: {len(tokenizer)}")
    print()

    test_texts = [
        "האיש רצה",
        "היום יפה",
        "שלום עולם",
        "ב|ירושלים",
        "כמה אתה חושב שזה יעלה לי",
    ]

    for text in test_texts:
        encoding = tokenize_with_offsets(text, tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        has_unk = tokenizer.unk_token_id in encoding["input_ids"]
        print(f"Text: {repr(text)}")
        print(f"  Tokens: {tokens}")
        print(f"  Has UNK: {has_unk}")
        print(f"  Offsets: {encoding['offset_mapping']}")
        print()


if __name__ == "__main__":
    main()
