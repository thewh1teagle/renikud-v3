"""
Pretokenize dataset files and save as Arrow datasets for fast loading.

Usage:
    uv run python src/pretokenize.py --dataset-dir dataset
    uv run python src/pretokenize.py --dataset-dir dataset --tokenizer-path tokenizer/dictabert-large-char-menaked

This finds all .txt files in the dataset dir, processes each line with
prepare_training_data(), and saves Arrow datasets to <dir>/.cache/<name>/.
Skips files whose cache is newer than the source .txt file.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

from dataset import prepare_training_data, load_dataset_from_file
from tokenizer_utils import load_tokenizer


def pretokenize_file(txt_path: Path, cache_path: Path, tokenizer) -> None:
    """Process a single .txt file and save as Arrow dataset."""
    # Check if cache is fresh
    if cache_path.exists():
        cache_mtime = os.path.getmtime(cache_path / "dataset_info.json")
        src_mtime = os.path.getmtime(txt_path)
        if cache_mtime > src_mtime:
            print(f"Skipping {txt_path.name} (cache is up to date)")
            return

    print(f"Processing {txt_path.name}...")
    texts = load_dataset_from_file(str(txt_path))

    # Process all lines
    rows = {
        "input_ids": [],
        "attention_mask": [],
        "vowel_labels": [],
        "dagesh_labels": [],
        "sin_labels": [],
        "stress_labels": [],
        "prefix_labels": [],
        "offset_mapping": [],
        "plain_text": [],
        "original_text": [],
    }

    for text in tqdm(texts, desc=f"Processing {txt_path.name}", unit="lines"):
        data = prepare_training_data(text, tokenizer)
        rows["input_ids"].append(data["input_ids"].tolist())
        rows["attention_mask"].append(data["attention_mask"].tolist())
        rows["vowel_labels"].append(data["vowel_labels"].tolist())
        rows["dagesh_labels"].append(data["dagesh_labels"].tolist())
        rows["sin_labels"].append(data["sin_labels"].tolist())
        rows["stress_labels"].append(data["stress_labels"].tolist())
        rows["prefix_labels"].append(data["prefix_labels"].tolist())
        rows["offset_mapping"].append(data["offset_mapping"].tolist())
        rows["plain_text"].append(data["plain_text"])
        rows["original_text"].append(data["original_text"])

    ds = Dataset.from_dict(rows)
    ds.save_to_disk(str(cache_path))
    print(f"Saved {len(ds)} samples to {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretokenize dataset files")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing .txt dataset files",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/dictabert-large-char-menaked",
        help="Tokenizer path",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    cache_dir = dataset_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)

    txt_files = sorted(dataset_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {dataset_dir}")
        return

    print(f"Found {len(txt_files)} files: {[f.name for f in txt_files]}")

    for txt_path in txt_files:
        name = txt_path.stem  # e.g. "train", "val"
        cache_path = cache_dir / name
        pretokenize_file(txt_path, cache_path, tokenizer)

    print("\nDone!")


if __name__ == "__main__":
    main()
