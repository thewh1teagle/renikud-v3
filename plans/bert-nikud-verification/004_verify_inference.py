#!/usr/bin/env python3
"""Verify inference pipeline."""

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
print("STEP 1: Loading model from checkpoint")
print("=" * 60)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "checkpoints", "test_checkpoint.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please run 003_verify_training.py first!")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
    model = HebrewNikudModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print("✓ Model loaded from checkpoint")
except Exception as e:
    print(f"✗ Model loading error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 2: Testing single text inference")
print("=" * 60)
try:
    test_texts = [
        "האיש רצה",
        "הילדים שיחקו",
        "היום יפה",
    ]

    for text in test_texts:
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

        print(f"  Input:  {text}")
        print(f"  Output: {nikud_text}")
        print()

    print("✓ Single inference successful")
except Exception as e:
    print(f"✗ Inference error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 3: Testing batch inference")
print("=" * 60)
try:
    batch_texts = [
        "האיש הלך",
        "האישה רצה",
        "הילדים שיחקו",
    ]

    # Tokenize batch
    encodings = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=False
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model.predict(input_ids, attention_mask)

    # Decode each text
    for i, (orig_text, inp_ids) in enumerate(zip(batch_texts, input_ids)):
        nikud_text = reconstruct_text_from_predictions(
            inp_ids,
            predictions["vowel"][i],
            predictions["dagesh"][i],
            predictions["sin"][i],
            predictions["stress"][i],
            predictions["prefix"][i],
            tokenizer,
        )
        print(f"  {i + 1}. {orig_text} -> {nikud_text}")

    print("\n✓ Batch inference successful")
except Exception as e:
    print(f"✗ Batch inference error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 4: Testing mixed text (Hebrew + English)")
print("=" * 60)
try:
    mixed_texts = [
        "האיש hello רצה",
        "This is a test: היום יפה",
        "123 המספר הוא",
    ]

    for text in mixed_texts:
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

        print(f"  Input:  {text}")
        print(f"  Output: {nikud_text}")
        print()

    print("✓ Mixed text inference successful")
except Exception as e:
    print(f"✗ Mixed text error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 5: Verifying NFD normalization")
print("=" * 60)
try:
    import unicodedata

    test_text = "האיש רצה"
    encoding = tokenizer(test_text, return_tensors="pt")
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

    # Check if it's NFD normalized
    normalized = unicodedata.normalize("NFD", nikud_text)
    is_nfd = nikud_text == normalized

    print(f"  Output: {nikud_text}")
    print(f"  NFD normalized: {is_nfd}")

    if is_nfd:
        print("✓ Output is NFD normalized")
    else:
        print("✗ Output is NOT NFD normalized")

except Exception as e:
    print(f"✗ NFD verification error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
print("Inference pipeline is functional!")
