"""
Pretokenize dataset files and save as Arrow datasets for fast loading.

Usage:
    uv run python src/pretokenize.py --input-txt dataset/train.txt --output-dir dataset/.cache/train
    uv run python src/pretokenize.py --input-txt dataset/val.txt --output-dir dataset/.cache/val

This processes a single .txt file, runs prepare_training_data() per line,
and saves the Arrow dataset to the output directory. Skips if the cache
is newer than the source .txt file.
"""

import argparse
import os
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

from datasets import Dataset
from tqdm import tqdm

from dataset import prepare_training_data, load_dataset_from_file
from tokenizer_utils import load_tokenizer


def _process_line(text, tokenizer_path):
    """Process a single line (worker function for multiprocessing)."""
    # Each worker loads its own tokenizer (not picklable)
    if not hasattr(_process_line, "_tokenizer"):
        _process_line._tokenizer = load_tokenizer(tokenizer_path)
    data = prepare_training_data(text, _process_line._tokenizer)
    return {
        "input_ids": data["input_ids"].tolist(),
        "attention_mask": data["attention_mask"].tolist(),
        "vowel_labels": data["vowel_labels"].tolist(),
        "dagesh_labels": data["dagesh_labels"].tolist(),
        "sin_labels": data["sin_labels"].tolist(),
        "stress_labels": data["stress_labels"].tolist(),
        "prefix_labels": data["prefix_labels"].tolist(),
        "offset_mapping": data["offset_mapping"].tolist(),
        "plain_text": data["plain_text"],
        "original_text": data["original_text"],
    }


def pretokenize_file(
    txt_path: Path, cache_path: Path, tokenizer_path: str, num_workers: int
) -> None:
    """Process a single .txt file and save as Arrow dataset."""
    # Check if cache is fresh
    if cache_path.exists():
        cache_info = cache_path / "dataset_info.json"
        if cache_info.exists():
            cache_mtime = os.path.getmtime(cache_info)
            src_mtime = os.path.getmtime(txt_path)
            if cache_mtime > src_mtime:
                print(f"Skipping {txt_path.name} (cache is up to date)")
                return

    print(f"Processing {txt_path.name} with {num_workers} workers...")
    texts = load_dataset_from_file(str(txt_path))

    worker_fn = partial(_process_line, tokenizer_path=tokenizer_path)

    if num_workers <= 1:
        # Single process
        tokenizer = load_tokenizer(tokenizer_path)
        results = []
        for text in tqdm(texts, desc=f"Processing {txt_path.name}", unit="lines"):
            data = prepare_training_data(text, tokenizer)
            results.append(
                {
                    "input_ids": data["input_ids"].tolist(),
                    "attention_mask": data["attention_mask"].tolist(),
                    "vowel_labels": data["vowel_labels"].tolist(),
                    "dagesh_labels": data["dagesh_labels"].tolist(),
                    "sin_labels": data["sin_labels"].tolist(),
                    "stress_labels": data["stress_labels"].tolist(),
                    "prefix_labels": data["prefix_labels"].tolist(),
                    "offset_mapping": data["offset_mapping"].tolist(),
                    "plain_text": data["plain_text"],
                    "original_text": data["original_text"],
                }
            )
    else:
        # Multiprocessing
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, texts, chunksize=256),
                    total=len(texts),
                    desc=f"Processing {txt_path.name}",
                    unit="lines",
                )
            )

    # Convert list of dicts to dict of lists
    rows = {key: [r[key] for r in results] for key in results[0]}

    ds = Dataset.from_dict(rows)
    ds.save_to_disk(str(cache_path))
    print(f"Saved {len(ds)} samples to {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretokenize dataset files")
    parser.add_argument(
        "--input-txt",
        type=str,
        required=True,
        help="Path to a single .txt dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the Arrow dataset cache",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="dicta-il/dictabert-large-char-menaked",
        help="Tokenizer path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )
    args = parser.parse_args()

    input_txt = Path(args.input_txt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_txt.exists():
        print(f"Input file not found: {input_txt}")
        return

    print(f"Processing file: {input_txt.name}")
    print(f"Output dir: {output_dir}")
    print(f"Workers: {args.workers}")

    pretokenize_file(input_txt, output_dir, args.tokenizer_path, args.workers)

    print("\nDone!")


if __name__ == "__main__":
    main()
