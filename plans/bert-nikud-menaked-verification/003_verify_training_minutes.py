#!/usr/bin/env python3
"""Run a short training loop with menaked tokenizer and infer samples."""

from __future__ import annotations

import time
from pathlib import Path
import sys
import argparse
import torch
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parents[2] / "bert-nikud"
sys.path.insert(0, str(base_dir / "src"))

from dataset import load_dataset_from_file, split_dataset, NikudDataset, collate_fn
from model import HebrewNikudModel
from decode import reconstruct_text_from_predictions
from tokenizer_utils import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train for N minutes")
    parser.add_argument("--minutes", type=int, default=1)
    args = parser.parse_args()

    tokenizer_dir = base_dir / "tokenizer" / "dictabert-large-char-menaked"
    train_file = base_dir / "dataset" / "train.txt"

    tokenizer = load_tokenizer(str(tokenizer_dir))

    texts = load_dataset_from_file(str(train_file))
    train_texts, _ = split_dataset(texts, eval_max_lines=0, seed=42)
    train_texts = train_texts[:256]

    dataset = NikudDataset(train_texts, tokenizer, use_cache=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HebrewNikudModel().to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    start_time = time.monotonic()
    max_seconds = max(1, args.minutes) * 60
    step = 0

    print("=" * 70)
    print(f"TRAINING LOOP (~{args.minutes} MINUTES)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Train samples: {len(train_texts)}")
    print()

    while time.monotonic() - start_time < max_seconds:
        for batch in dataloader:
            step += 1
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                vowel_labels=batch["vowel_labels"],
                dagesh_labels=batch["dagesh_labels"],
                sin_labels=batch["sin_labels"],
                stress_labels=batch["stress_labels"],
                prefix_labels=batch["prefix_labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                elapsed = time.monotonic() - start_time
                print(f"Step {step:04d} | Loss {loss.item():.4f} | {elapsed:.1f}s")

            if time.monotonic() - start_time >= max_seconds:
                break

    total_time = time.monotonic() - start_time
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.minutes}min_menaked.pt"
    torch.save(model.state_dict(), ckpt_path)

    print("\nDone.")
    print(f"Steps: {step}")
    print(f"Elapsed: {total_time:.1f}s")
    print(f"Checkpoint: {ckpt_path}")

    val_file = base_dir / "dataset" / "val.txt"
    samples = []
    if val_file.exists():
        with val_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
                if len(samples) >= 20:
                    break
    else:
        samples = [
            "האיש רצה",
            "היום יפה",
            "ב|ירושלים",
            "שלום! 123",
        ]

    model.eval()
    print("\nInference:")
    print("=" * 70)
    for text in samples:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"][0]

        with torch.no_grad():
            preds = model.predict(input_ids, attention_mask)

        predicted_text = reconstruct_text_from_predictions(
            input_ids[0],
            offset_mapping,
            preds["vowel"][0],
            preds["dagesh"][0],
            preds["sin"][0],
            preds["stress"][0],
            preds["prefix"][0],
            tokenizer,
        )

        print(f"Input:  {text}")
        print(f"Output: {predicted_text}")
        print("-" * 70)


if __name__ == "__main__":
    main()
