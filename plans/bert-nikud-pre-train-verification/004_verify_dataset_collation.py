"""Verify dataset loading, collation, and batch shapes for training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

from dataset import NikudDataset, load_dataset_from_file, collate_fn
from tokenizer_utils import load_tokenizer
from torch.utils.data import DataLoader

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "dataset" / "val.txt"

tokenizer = load_tokenizer(str(TOKENIZER_PATH))
texts = load_dataset_from_file(str(DATASET_PATH))[:50]  # use 50 lines for speed
print(f"Creating dataset from {len(texts)} texts...")

dataset = NikudDataset(texts, tokenizer, use_cache=False)
print(f"Dataset size: {len(dataset)}")

# Check single item
item = dataset[0]
expected_keys = {"input_ids", "attention_mask", "vowel_labels", "dagesh_labels",
                 "sin_labels", "stress_labels", "prefix_labels", "offset_mapping",
                 "plain_text", "original_text"}
assert set(item.keys()) == expected_keys, f"Item keys mismatch: {set(item.keys())} vs {expected_keys}"
print(f"Single item keys: PASS")

# Check shapes consistency
seq_len = item["input_ids"].shape[0]
assert item["attention_mask"].shape[0] == seq_len
assert item["vowel_labels"].shape[0] == seq_len
assert item["offset_mapping"].shape == (seq_len, 2)
print(f"Single item shapes (seq_len={seq_len}): PASS")

# Test collation with DataLoader
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
print(f"\nBatch keys: {list(batch.keys())}")
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch vowel_labels shape: {batch['vowel_labels'].shape}")
print(f"Batch offset_mapping shape: {batch['offset_mapping'].shape}")

bs, max_len = batch["input_ids"].shape
assert batch["attention_mask"].shape == (bs, max_len)
assert batch["vowel_labels"].shape == (bs, max_len)
assert batch["dagesh_labels"].shape == (bs, max_len)
assert batch["sin_labels"].shape == (bs, max_len)
assert batch["stress_labels"].shape == (bs, max_len)
assert batch["prefix_labels"].shape == (bs, max_len)
assert batch["offset_mapping"].shape == (bs, max_len, 2)
assert len(batch["plain_text"]) == bs
assert len(batch["original_text"]) == bs
print("Batch shapes: PASS")

# Verify padding: padded positions should have attention_mask=0 and labels=-100
import torch
for i in range(bs):
    item_len = dataset[i]["input_ids"].shape[0]
    if item_len < max_len:
        assert (batch["attention_mask"][i, item_len:] == 0).all(), "Padding attention_mask not 0"
        assert (batch["vowel_labels"][i, item_len:] == -100).all(), "Padding vowel_labels not -100"
print("Padding correctness: PASS")

print("\nDataset & collation verification: PASS")
