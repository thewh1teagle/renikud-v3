#!/usr/bin/env python3
"""Verify setup and data loading."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bert-nikud", "src")
)

from transformers import AutoTokenizer
from dataset import load_dataset_from_file, split_dataset
import torch

print("=" * 60)
print("STEP 1: Checking GPU availability")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
print()

print("=" * 60)
print("STEP 2: Loading tokenizer")
print("=" * 60)
try:
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Test tokenization
    test_text = "האיש רצה"
    encoding = tokenizer(test_text, return_tensors="pt")
    print(
        f"✓ Tokenization test: '{test_text}' -> {len(encoding['input_ids'][0])} tokens"
    )
    print(f"  Input IDs shape: {encoding['input_ids'].shape}")
except Exception as e:
    print(f"✗ Tokenizer error: {e}")
    sys.exit(1)
print()

print("=" * 60)
print("STEP 3: Loading sample data")
print("=" * 60)
try:
    sample_file = os.path.join(os.path.dirname(__file__), "sample_data.txt")
    texts = load_dataset_from_file(sample_file)
    print(f"✓ Loaded {len(texts)} sample lines")

    # Verify NFD normalization
    import unicodedata

    for i, text in enumerate(texts[:3]):
        normalized = unicodedata.normalize("NFD", text)
        print(f"  Sample {i + 1}: {text[:50]}")
        if text != normalized:
            print(f"    (NFD diff: {normalized[:50]})")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    sys.exit(1)
print()

print("=" * 60)
print("STEP 4: Verifying dataset split")
print("=" * 60)
try:
    train_texts, eval_texts = split_dataset(texts, eval_max_lines=2, seed=42)
    print(f"✓ Train: {len(train_texts)} lines")
    print(f"✓ Eval: {len(eval_texts)} lines")
except Exception as e:
    print(f"✗ Dataset split error: {e}")
    sys.exit(1)
print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
