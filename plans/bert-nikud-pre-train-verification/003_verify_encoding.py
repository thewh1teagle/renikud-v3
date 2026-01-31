"""Verify encode/decode roundtrip on actual dataset lines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

from dataset import load_dataset_from_file, prepare_training_data, VOWEL_TO_ID, ID_TO_VOWEL
from tokenizer_utils import load_tokenizer
from encode import extract_nikud_labels
from decode import reconstruct_text_from_predictions
from constants import LETTERS
import torch

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "dataset" / "val.txt"

tokenizer = load_tokenizer(str(TOKENIZER_PATH))
texts = load_dataset_from_file(str(DATASET_PATH))
print(f"Loaded {len(texts)} validation texts")

# Test extract_nikud_labels on first 20 lines
print("\n--- Testing extract_nikud_labels ---")
for idx in range(min(20, len(texts))):
    nikud_text = texts[idx]
    plain_text, labels = extract_nikud_labels(nikud_text)

    # Every label should have all 5 keys
    for i, label in enumerate(labels):
        assert set(label.keys()) == {"vowel", "dagesh", "sin", "stress", "prefix"}, f"Line {idx}, char {i}: missing keys"

    # plain_text length should match labels length
    assert len(plain_text) == len(labels), f"Line {idx}: plain_text len {len(plain_text)} != labels len {len(labels)}"

    # Hebrew letters should not have -100 labels
    for i, (ch, label) in enumerate(zip(plain_text, labels)):
        if ch in LETTERS:
            assert label["vowel"] != -100, f"Line {idx}, char {i} '{ch}': Hebrew letter has -100 vowel"
        else:
            assert label["vowel"] == -100, f"Line {idx}, char {i} '{ch}': non-Hebrew has vowel label {label['vowel']}"

print(f"extract_nikud_labels on {min(20, len(texts))} lines: PASS")

# Test prepare_training_data and roundtrip
print("\n--- Testing encode->decode roundtrip ---")
mismatches = 0
for idx in range(min(50, len(texts))):
    nikud_text = texts[idx]
    data = prepare_training_data(nikud_text, tokenizer)

    input_ids = data["input_ids"]
    offset_mapping = data["offset_mapping"]
    vowel_labels = data["vowel_labels"]
    dagesh_labels = data["dagesh_labels"]
    sin_labels = data["sin_labels"]
    stress_labels = data["stress_labels"]
    prefix_labels = data["prefix_labels"]

    # Simulate perfect predictions (use ground truth labels)
    # For binary labels, clamp -100 to 0 (model won't predict -100)
    vowel_preds = vowel_labels.clone()
    vowel_preds[vowel_preds == -100] = 0
    dagesh_preds = dagesh_labels.clone()
    dagesh_preds[dagesh_preds == -100] = 0
    sin_preds = sin_labels.clone()
    sin_preds[sin_preds == -100] = 0
    stress_preds = stress_labels.clone()
    stress_preds[stress_preds == -100] = 0
    prefix_preds = prefix_labels.clone()
    prefix_preds[prefix_preds == -100] = 0

    reconstructed = reconstruct_text_from_predictions(
        input_ids, offset_mapping,
        vowel_preds, dagesh_preds, sin_preds, stress_preds, prefix_preds,
        tokenizer,
    )

    # Compare with normalized original
    from normalize import normalize
    expected = normalize(nikud_text)

    if reconstructed != expected:
        mismatches += 1
        if mismatches <= 3:
            print(f"\n  Line {idx} MISMATCH:")
            print(f"  Expected:      {expected}")
            print(f"  Reconstructed: {reconstructed}")
            # Find first diff
            for ci, (a, b) in enumerate(zip(expected, reconstructed)):
                if a != b:
                    print(f"  First diff at char {ci}: expected U+{ord(a):04X} got U+{ord(b):04X}")
                    break

print(f"\nRoundtrip: {min(50, len(texts)) - mismatches}/{min(50, len(texts))} exact matches")
if mismatches > 0:
    print(f"WARNING: {mismatches} mismatches (may be acceptable if minor normalization diffs)")
else:
    print("Encode/decode roundtrip: PASS")
