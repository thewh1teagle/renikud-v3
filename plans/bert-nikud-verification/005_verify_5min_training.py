#!/usr/bin/env python3
"""Train for ~5 minutes with more data to get meaningful results."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bert-nikud", "src")
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import HebrewNikudModel
from dataset import NikudDataset, collate_fn
from tqdm import tqdm
import time

print("=" * 60)
print("5-MINUTE TRAINING VERIFICATION")
print("=" * 60)
print()

# Generate larger synthetic dataset by repeating the sample data
SAMPLE_TEXTS = [
    "הָאִישׁ רָצָה",
    "הָאִישָׁה הָלְכָה",
    "הַיְלָדִים שָׂחֲקוּ",
    "הַמִּלָּה הִיא טוֹבָה",
    "הַסֵּפֶר נִמְצָא",
    "הָעִיר גְּדוֹלָה",
    "הַשָּׁמַיִם כְּחוּלִים",
    "הַיּוֹם יָפֶה",
    "הַלַּיְלָה חָשֵׁךְ",
    "הָעוֹלָם יָפֶה",
    "הַכֹּל טוֹב",
    "הַבַּיִת גָּדוֹל",
    "הַיַּרְדֵּן נָהָר",
    "הָאוֹר בָּהִיר",
    "הַחֹפֶשׁ נָעִים",
    "הָאָדָם חָכָם",
    "הָאִשָּׁה יְפָה",
    "הַכֶּלֶב רָץ",
    "הַחָתוּל שׁוֹחֶה",
    "הַצִּפּוֹר עָפָה",
]

# Repeat to get more training data
MULTIPLIER = 100
train_texts = SAMPLE_TEXTS * MULTIPLIER
print(
    f"Created {len(train_texts)} training examples ({len(SAMPLE_TEXTS)} unique × {MULTIPLIER})"
)

# Small eval set
eval_texts = SAMPLE_TEXTS[:5]
print(f"Created {len(eval_texts)} evaluation examples")
print()

print("=" * 60)
print("Loading tokenizer and creating datasets")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
train_dataset = NikudDataset(train_texts, tokenizer, use_cache=False)
eval_dataset = NikudDataset(eval_texts, tokenizer, use_cache=False)

batch_size = 8
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Eval dataset: {len(eval_dataset)} samples")
print(f"✓ Batch size: {batch_size}, {len(train_loader)} batches per epoch")
print()

print("=" * 60)
print("Initializing model")
print("=" * 60)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = HebrewNikudModel(model_name="dicta-il/dictabert-large-char")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
print("✓ Model and optimizer ready")
print()

print("=" * 60)
print("Training for 5 minutes")
print("=" * 60)
TRAINING_TIME_SECONDS = 300  # 5 minutes
start_time = time.time()

model.train()
step = 0
losses = []

pbar = tqdm(total=TRAINING_TIME_SECONDS, desc="Training time", unit="s")

while (time.time() - start_time) < TRAINING_TIME_SECONDS:
    for batch in train_loader:
        if (time.time() - start_time) >= TRAINING_TIME_SECONDS:
            break

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vowel_labels = batch["vowel_labels"].to(device)
        dagesh_labels = batch["dagesh_labels"].to(device)
        sin_labels = batch["sin_labels"].to(device)
        stress_labels = batch["stress_labels"].to(device)
        prefix_labels = batch["prefix_labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids,
            attention_mask,
            vowel_labels=vowel_labels,
            dagesh_labels=dagesh_labels,
            sin_labels=sin_labels,
            stress_labels=stress_labels,
            prefix_labels=prefix_labels,
        )

        loss = outputs["loss"]
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        # Update progress
        elapsed = time.time() - start_time
        pbar.update(int(elapsed - pbar.n))
        pbar.set_postfix(
            {
                "step": step,
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{sum(losses[-100:]) / min(len(losses), 100):.4f}",
            }
        )

pbar.close()

print()
print("✓ Training completed!")
print(f"  Total steps: {step}")
print(f"  Time elapsed: {time.time() - start_time:.1f}s")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Best loss: {min(losses):.4f}")
print()

print("=" * 60)
print("Saving checkpoint")
print("=" * 60)
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "trained_5min.pt")

torch.save(model.state_dict(), checkpoint_path)
print(f"✓ Checkpoint saved: {checkpoint_path}")
print()

print("=" * 60)
print("Quick evaluation")
print("=" * 60)
model.eval()
eval_loader = DataLoader(
    eval_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
)

eval_losses = []
with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vowel_labels = batch["vowel_labels"].to(device)
        dagesh_labels = batch["dagesh_labels"].to(device)
        sin_labels = batch["sin_labels"].to(device)
        stress_labels = batch["stress_labels"].to(device)
        prefix_labels = batch["prefix_labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask,
            vowel_labels=vowel_labels,
            dagesh_labels=dagesh_labels,
            sin_labels=sin_labels,
            stress_labels=stress_labels,
            prefix_labels=prefix_labels,
        )

        eval_losses.append(outputs["loss"].item())

avg_eval_loss = sum(eval_losses) / len(eval_losses)
print(f"✓ Eval loss: {avg_eval_loss:.4f}")
print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
print("Run 006_verify_inference_trained.py to test inference!")
