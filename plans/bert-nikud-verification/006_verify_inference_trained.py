#!/usr/bin/env python3
"""Test inference with the 5-minute trained model."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bert-nikud", "src")
)

import torch
from transformers import AutoTokenizer
from model import HebrewNikudModel
from decode import reconstruct_text_from_predictions

print("=" * 60)
print("INFERENCE WITH 5-MIN TRAINED MODEL")
print("=" * 60)
print()

# Load model
print("Loading model from checkpoint...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint_path = os.path.join(
    os.path.dirname(__file__), "checkpoints", "trained_5min.pt"
)

if not os.path.exists(checkpoint_path):
    print(f"✗ Checkpoint not found: {checkpoint_path}")
    print("  Please run 005_verify_5min_training.py first!")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
model = HebrewNikudModel()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print("✓ Model loaded")
print()

# Test texts - some from training, some new
test_texts = [
    # From training
    ("האיש רצה", True),
    ("הילדים שיחקו", True),
    ("היום יפה", True),
    # Slightly new (same words, different order)
    ("רצה האיש", False),
    ("יפה היום", False),
    # Mixed text
    ("האיש hello רצה", False),
]

print("=" * 60)
print("Testing inference")
print("=" * 60)

for text, seen_during_training in test_texts:
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model.predict(input_ids, attention_mask)

    nikud_text = reconstruct_text_from_predictions(
        input_ids[0],
        predictions["vowel"][0],
        predictions["dagesh"][0],
        predictions["sin"][0],
        predictions["stress"][0],
        predictions["prefix"][0],
        tokenizer,
    )

    tag = "[seen]" if seen_during_training else "[new]"
    print(f"{tag} Input:  {text}")
    print(f"    Output: {nikud_text}")

    # Check for UNK tokens
    if "[UNK]" in nikud_text:
        print("    ⚠ WARNING: Contains [UNK] tokens")
    else:
        print("    ✓ No [UNK] tokens")
    print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
