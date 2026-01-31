#!/usr/bin/env python3
"""
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
sudo apt install p7zip-full -y
7z x knesset_phonemes_v1.txt.7z

Script to extract lines from a tab-separated file (nikud\tphonemes), remove specific characters, and split into train/val.

uv run src/prepare_data.py --input knesset_phonemes_v1.txt --output-dir ./dataset --lines 1000000
"""
import argparse
import random
import os
import regex as re
from tqdm import tqdm
from normalize import normalize

SHVA = '\u05b0'
VOCAL_SHVA = '\u05bd'

def normalize_vocal_shva(text: str) -> str:
    """
    When a letter has both shva and meteg (vocal shva) in its diacritics,
    remove the shva and keep only the meteg as vocal shva.
    """
    def callback(match):
        letter = match.group(1)
        diac = list(match.group(2))
        if SHVA in diac and VOCAL_SHVA in diac:
            diac = [d for d in diac if d != SHVA]
        return letter + ''.join(diac)

    return re.sub(r'(\p{L})(\p{M}+)', callback, text)


def prepare_data(input_file, output_dir, num_lines, val_ratio, seed):
    """
    Extract lines from input file, clean them, and split into train/val sets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Count total lines in file
    print(f"Counting lines in {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total lines in file: {total_lines:,}")
    lines_to_process = min(num_lines, total_lines)
    print(f"Processing {lines_to_process:,} lines...")

    lines = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(tqdm(infile, total=lines_to_process, desc="Processing")):
            if i >= num_lines:
                break
            # Take only the Hebrew nikud column (before tab)
            cleaned_line = line.split('\t')[0] + '\n'
            cleaned_line = normalize(cleaned_line)
            cleaned_line = normalize_vocal_shva(cleaned_line)
            lines.append(cleaned_line)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - val_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"\n✓ Processed {len(lines):,} lines (seed={seed})")
    print(f"✓ Train: {len(train_lines):,} lines -> {train_path}")
    print(f"✓ Val:   {len(val_lines):,} lines -> {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract lines from a file and remove specific characters'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='knesset_phonemes_v1.txt',
        help='Input file path (default: knesset_phonemes_v1.txt)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Output directory (default: ./data)'
    )
    
    parser.add_argument(
        '--lines',
        type=int,
        default=1_000_000,
        help='Number of lines to extract (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )

    args = parser.parse_args()

    prepare_data(
        input_file=args.input,
        output_dir=args.output_dir,
        num_lines=args.lines,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

