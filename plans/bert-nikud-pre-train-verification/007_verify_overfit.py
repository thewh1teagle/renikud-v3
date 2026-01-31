"""Train on 10 samples until overfitting to 100% match (or timeout after 5 min)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

import torch
from torch.utils.data import DataLoader
from model import HebrewNikudModel
from dataset import NikudDataset, load_dataset_from_file, collate_fn
from decode import reconstruct_text_from_predictions
from tokenizer_utils import load_tokenizer
from normalize import normalize

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "dataset" / "train.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = load_tokenizer(str(TOKENIZER_PATH))
texts = load_dataset_from_file(str(DATASET_PATH))[:10]
print(f"Overfitting on {len(texts)} samples:")
for i, t in enumerate(texts):
    print(f"  [{i}] {normalize(t)[:80]}...")

dataset = NikudDataset(texts, tokenizer, use_cache=False)
loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
batch = next(iter(loader))  # single batch, reuse every step

model = HebrewNikudModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Move batch to device once
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
vowel_labels = batch["vowel_labels"].to(device)
dagesh_labels = batch["dagesh_labels"].to(device)
sin_labels = batch["sin_labels"].to(device)
stress_labels = batch["stress_labels"].to(device)
prefix_labels = batch["prefix_labels"].to(device)
offset_mappings = batch["offset_mapping"]
targets = [normalize(t) for t in batch["original_text"]]

start_time = time.time()
max_time = 300  # 5 minutes
step = 0

while time.time() - start_time < max_time:
    model.train()
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
    step += 1

    # Check every 50 steps
    if step % 50 == 0:
        model.eval()
        with torch.no_grad():
            preds = model.predict(input_ids, attention_mask)

        matches = 0
        for i in range(len(texts)):
            predicted = reconstruct_text_from_predictions(
                input_ids[i], offset_mappings[i],
                preds["vowel"][i], preds["dagesh"][i],
                preds["sin"][i], preds["stress"][i], preds["prefix"][i],
                tokenizer,
            )
            if predicted == targets[i]:
                matches += 1

        elapsed = time.time() - start_time
        print(f"Step {step}: loss={loss.item():.6f}, matches={matches}/{len(texts)} ({elapsed:.0f}s)")

        if matches == len(texts):
            print(f"\n100% match at step {step} ({elapsed:.1f}s)")
            # Print all samples
            with torch.no_grad():
                preds = model.predict(input_ids, attention_mask)
            for i in range(len(texts)):
                predicted = reconstruct_text_from_predictions(
                    input_ids[i], offset_mappings[i],
                    preds["vowel"][i], preds["dagesh"][i],
                    preds["sin"][i], preds["stress"][i], preds["prefix"][i],
                    tokenizer,
                )
                print(f"\n  Target:    {targets[i]}")
                print(f"  Predicted: {predicted}")
                print(f"  Match: {predicted == targets[i]}")
            print("\nOverfit verification: PASS")
            sys.exit(0)

elapsed = time.time() - start_time
print(f"\nFAIL: Did not reach 100% match after {step} steps ({elapsed:.1f}s)")
# Print final state
model.eval()
with torch.no_grad():
    preds = model.predict(input_ids, attention_mask)
for i in range(len(texts)):
    predicted = reconstruct_text_from_predictions(
        input_ids[i], offset_mappings[i],
        preds["vowel"][i], preds["dagesh"][i],
        preds["sin"][i], preds["stress"][i], preds["prefix"][i],
        tokenizer,
    )
    match = predicted == targets[i]
    if not match:
        print(f"\n  [{i}] MISMATCH:")
        print(f"  Target:    {targets[i]}")
        print(f"  Predicted: {predicted}")
        for ci, (a, b) in enumerate(zip(targets[i], predicted)):
            if a != b:
                print(f"  First diff at char {ci}: expected U+{ord(a):04X} got U+{ord(b):04X}")
                break
sys.exit(1)
