"""
Dataset preparation for Hebrew nikud prediction.

This module handles:
- Creating input/label pairs for training
- Hybrid encoding: vowels as multi-class (0-7), others as binary
- Loading pretokenized Arrow datasets
"""

from typing import List
import torch
from pathlib import Path
from datasets import load_from_disk
from encode import extract_nikud_labels
from constants import A_PATAH, E_TSERE, I_HIRIK, O_HOLAM, U_QUBUT, SHVA, E_VOCAL_SHVA


# Vowel encoding (multi-class)
VOWEL_NONE = 0
VOWEL_PATAH = 1
VOWEL_TSERE = 2
VOWEL_HIRIK = 3
VOWEL_HOLAM = 4
VOWEL_QUBUT = 5
VOWEL_SHVA = 6
VOWEL_VOCAL_SHVA = 7

VOWEL_TO_ID = {
    None: VOWEL_NONE,
    A_PATAH: VOWEL_PATAH,
    E_TSERE: VOWEL_TSERE,
    I_HIRIK: VOWEL_HIRIK,
    O_HOLAM: VOWEL_HOLAM,
    U_QUBUT: VOWEL_QUBUT,
    SHVA: VOWEL_SHVA,
    E_VOCAL_SHVA: VOWEL_VOCAL_SHVA,
}

ID_TO_VOWEL = {v: k for k, v in VOWEL_TO_ID.items()}


def tokenize_with_offsets(text: str, tokenizer) -> dict:
    """
    Tokenize text with offset mapping.

    Args:
        text: Plain text to tokenize
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with input_ids, attention_mask, and offset_mapping
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "offset_mapping": encoding["offset_mapping"][0],
    }


def prepare_training_data(nikud_text: str, tokenizer) -> dict:
    """
    Prepare training data from nikud'd Hebrew text.

    Args:
        nikud_text: Hebrew text with nikud marks
        tokenizer: HuggingFace tokenizer for the model

    Returns:
        Dictionary with input_ids, attention_mask, and label tensors
    """
    plain_text, labels = extract_nikud_labels(nikud_text)

    # Tokenize the plain text with offset mapping
    encoding = tokenize_with_offsets(plain_text, tokenizer)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]
    num_tokens = len(input_ids)

    # Create label tensors (-100 = ignored in loss)
    vowel_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    dagesh_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    sin_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    stress_labels = torch.full((num_tokens,), -100, dtype=torch.long)
    prefix_labels = torch.full((num_tokens,), -100, dtype=torch.long)

    # Fill in labels for actual characters (skip [CLS] at position 0)
    for i, label in enumerate(labels):
        token_idx = i + 1  # +1 to account for [CLS] token
        if token_idx < num_tokens - 1:  # -1 to avoid [SEP]
            vowel_labels[token_idx] = label["vowel"]
            dagesh_labels[token_idx] = label["dagesh"]
            sin_labels[token_idx] = label["sin"]
            stress_labels[token_idx] = label["stress"]
            prefix_labels[token_idx] = label["prefix"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "vowel_labels": vowel_labels,
        "dagesh_labels": dagesh_labels,
        "sin_labels": sin_labels,
        "stress_labels": stress_labels,
        "prefix_labels": prefix_labels,
        "offset_mapping": offset_mapping,
        "plain_text": plain_text,
        "original_text": nikud_text,
    }


def load_dataset_from_file(file_path: str) -> List[str]:
    """Load Hebrew texts from a file (one text per line)."""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def load_pretokenized(dataset_dir: str, split: str):
    """
    Load a pretokenized Arrow dataset.

    Args:
        dataset_dir: Directory containing .cache/ subdirectory
        split: Split name (e.g. "train", "val", "val_200")

    Returns:
        HuggingFace Dataset
    """
    cache_path = Path(dataset_dir) / ".cache" / split
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pretokenized dataset not found at {cache_path}. "
            f"Run: uv run python src/pretokenize.py --dataset-dir {dataset_dir}"
        )
    ds = load_from_disk(str(cache_path))
    ds.set_format("torch")
    print(f"Loaded {len(ds)} samples from {cache_path}")
    return ds


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of data dictionaries

    Returns:
        Dictionary with batched and padded tensors
    """
    # Find max length in batch
    max_len = max(item["input_ids"].shape[0] for item in batch)

    # Initialize batched tensors
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    vowel_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    dagesh_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    sin_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    stress_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    prefix_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    offset_mapping = torch.zeros(batch_size, max_len, 2, dtype=torch.long)
    plain_texts = []
    original_texts = []

    # Fill in the batch
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]

        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        vowel_labels[i, :seq_len] = item["vowel_labels"]
        dagesh_labels[i, :seq_len] = item["dagesh_labels"]
        sin_labels[i, :seq_len] = item["sin_labels"]
        stress_labels[i, :seq_len] = item["stress_labels"]
        prefix_labels[i, :seq_len] = item["prefix_labels"]
        offset_mapping[i, :seq_len] = item["offset_mapping"]

        plain_texts.append(item["plain_text"])
        original_texts.append(item["original_text"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "vowel_labels": vowel_labels,
        "dagesh_labels": dagesh_labels,
        "sin_labels": sin_labels,
        "stress_labels": stress_labels,
        "prefix_labels": prefix_labels,
        "offset_mapping": offset_mapping,
        "plain_text": plain_texts,
        "original_text": original_texts,
    }
