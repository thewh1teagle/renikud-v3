"""Verify tokenizer loads correctly and handles Hebrew text as expected."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

from tokenizer_utils import load_tokenizer
from constants import LETTERS

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"

tokenizer = load_tokenizer(str(TOKENIZER_PATH))
print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
print(f"Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}")

# Verify every Hebrew letter tokenizes to a single token (not UNK)
print(f"\nChecking {len(LETTERS)} Hebrew letters...")
failures = []
for letter in LETTERS:
    enc = tokenizer(letter, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) != 1:
        failures.append(f"  '{letter}' -> {len(ids)} tokens (expected 1)")
    elif ids[0] == tokenizer.unk_token_id:
        failures.append(f"  '{letter}' -> UNK")

if failures:
    print("FAIL: Some Hebrew letters not tokenized correctly:")
    for f in failures:
        print(f)
    sys.exit(1)
print("All Hebrew letters tokenize to single non-UNK tokens: PASS")

# Verify character-level: mixed text should produce 1 token per char
test_text = "שלום hello 123"
enc = tokenizer(test_text, add_special_tokens=True, return_offsets_mapping=True)
ids = enc["input_ids"]
offsets = enc["offset_mapping"]
# Expected: [CLS] + len(test_text) chars + [SEP]
expected_len = 1 + len(test_text) + 1
print(f"\nMixed text: '{test_text}'")
print(f"Tokens: {len(ids)} (expected {expected_len})")
if len(ids) != expected_len:
    print(f"FAIL: token count mismatch")
    sys.exit(1)
print("Character-level tokenization: PASS")

# Verify offset_mapping is correct
print(f"\nOffset mapping check:")
for i in range(1, len(ids) - 1):  # skip CLS and SEP
    start, end = offsets[i]
    char = test_text[start:end]
    decoded = tokenizer.decode([ids[i]])
    if char != decoded:
        print(f"  FAIL: position {i}: offset gives '{char}' but decode gives '{decoded}'")
        sys.exit(1)
print("Offset mapping matches decoded tokens: PASS")

print("\nTokenizer verification: PASS")
