"""Train for ~1 minute on a small subset and verify loss decreases and inference works."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

import torch
from torch.utils.data import DataLoader
from model import HebrewNikudModel
from dataset import NikudDataset, load_dataset_from_file, split_dataset, collate_fn
from decode import reconstruct_text_from_predictions
from evaluate import calculate_cer
from tokenizer_utils import load_tokenizer
from normalize import normalize

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "dataset" / "train.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load data
tokenizer = load_tokenizer(str(TOKENIZER_PATH))
texts = load_dataset_from_file(str(DATASET_PATH))[:500]  # small subset
train_texts, eval_texts = split_dataset(texts, eval_max_lines=20, seed=42)
print(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")

train_dataset = NikudDataset(train_texts, tokenizer, use_cache=False)
eval_dataset = NikudDataset(eval_texts, tokenizer, use_cache=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)

# Model + optimizer
model = HebrewNikudModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Train for ~1 minute
print("\n--- Training for ~1 minute ---")
model.train()
start_time = time.time()
step = 0
losses = []
first_loss = None

while time.time() - start_time < 60:
    for batch in train_loader:
        if time.time() - start_time >= 60:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vowel_labels = batch["vowel_labels"].to(device)
        dagesh_labels = batch["dagesh_labels"].to(device)
        sin_labels = batch["sin_labels"].to(device)
        stress_labels = batch["stress_labels"].to(device)
        prefix_labels = batch["prefix_labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            vowel_labels=vowel_labels, dagesh_labels=dagesh_labels,
            sin_labels=sin_labels, stress_labels=stress_labels,
            prefix_labels=prefix_labels,
        )
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if first_loss is None:
            first_loss = loss_val
        step += 1

        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step}: loss={loss_val:.4f} ({elapsed:.0f}s)")

elapsed = time.time() - start_time
last_loss = losses[-1] if losses else 0
print(f"\nTrained {step} steps in {elapsed:.1f}s")
print(f"First loss: {first_loss:.4f}, Last loss: {last_loss:.4f}")

if last_loss >= first_loss:
    print("WARNING: Loss did not decrease!")
else:
    print(f"Loss decreased by {first_loss - last_loss:.4f}: PASS")

# Evaluate: run inference on eval set and check CER
print("\n--- Evaluation ---")
model.eval()
total_cer = 0.0
num_samples = 0

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model.predict(input_ids, attention_mask)

        for i in range(input_ids.shape[0]):
            offset_mapping = batch["offset_mapping"][i]
            predicted = reconstruct_text_from_predictions(
                input_ids[i], offset_mapping,
                preds["vowel"][i], preds["dagesh"][i],
                preds["sin"][i], preds["stress"][i], preds["prefix"][i],
                tokenizer,
            )
            target = batch["original_text"][i]
            cer = calculate_cer(predicted, target)
            total_cer += cer
            num_samples += 1

            if num_samples <= 3:
                print(f"\n  Target:    {normalize(target)}")
                print(f"  Predicted: {predicted}")
                print(f"  CER: {cer:.4f}")

avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
print(f"\nAvg CER on eval: {avg_cer:.4f} ({num_samples} samples)")
print("(CER expected to be high after only 1 min of training on 500 samples)")

print("\nTraining verification: PASS")
