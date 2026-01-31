"""Verify model loads, runs forward pass on GPU, and produces correct output shapes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "src"))

import torch
from model import HebrewNikudModel, count_parameters
from dataset import NikudDataset, load_dataset_from_file, collate_fn
from tokenizer_utils import load_tokenizer
from torch.utils.data import DataLoader

TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "tokenizer" / "dictabert-large-char-menaked"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "bert-nikud" / "dataset" / "val.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("Loading model...")
model = HebrewNikudModel()
model.to(device)
total, trainable = count_parameters(model)
print(f"Parameters: {total:,} total, {trainable:,} trainable")

# Prepare a batch
tokenizer = load_tokenizer(str(TOKENIZER_PATH))
texts = load_dataset_from_file(str(DATASET_PATH))[:8]
dataset = NikudDataset(texts, tokenizer, use_cache=False)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
vowel_labels = batch["vowel_labels"].to(device)
dagesh_labels = batch["dagesh_labels"].to(device)
sin_labels = batch["sin_labels"].to(device)
stress_labels = batch["stress_labels"].to(device)
prefix_labels = batch["prefix_labels"].to(device)

bs, seq_len = input_ids.shape
print(f"\nBatch: {bs} samples, max seq_len={seq_len}")

# Forward pass with labels (training mode)
print("Forward pass (training)...")
model.train()
outputs = model(
    input_ids=input_ids, attention_mask=attention_mask,
    vowel_labels=vowel_labels, dagesh_labels=dagesh_labels,
    sin_labels=sin_labels, stress_labels=stress_labels,
    prefix_labels=prefix_labels,
)

assert outputs["vowel_logits"].shape == (bs, seq_len, 8), f"vowel_logits shape: {outputs['vowel_logits'].shape}"
assert outputs["dagesh_logits"].shape == (bs, seq_len), f"dagesh_logits shape: {outputs['dagesh_logits'].shape}"
assert outputs["sin_logits"].shape == (bs, seq_len)
assert outputs["stress_logits"].shape == (bs, seq_len)
assert outputs["prefix_logits"].shape == (bs, seq_len)
assert "loss" in outputs
assert outputs["loss"].requires_grad
print(f"Training output shapes: PASS")
print(f"Loss: {outputs['loss'].item():.4f}")
print(f"  vowel: {outputs['vowel_loss'].item():.4f}, dagesh: {outputs['dagesh_loss'].item():.4f}, "
      f"sin: {outputs['sin_loss'].item():.4f}, stress: {outputs['stress_loss'].item():.4f}, "
      f"prefix: {outputs['prefix_loss'].item():.4f}")

# Backward pass
print("\nBackward pass...")
outputs["loss"].backward()
grad_norms = []
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norms.append((name, p.grad.norm().item()))
print(f"Parameters with gradients: {len(grad_norms)}/{trainable}")
assert len(grad_norms) > 0, "No gradients computed"
print("Backward pass: PASS")

# Predict mode (inference)
print("\nPredict mode...")
preds = model.predict(input_ids, attention_mask)
assert preds["vowel"].shape == (bs, seq_len)
assert preds["dagesh"].shape == (bs, seq_len)
assert preds["vowel"].min() >= 0 and preds["vowel"].max() <= 7
assert preds["dagesh"].min() >= 0 and preds["dagesh"].max() <= 1
print("Predict output shapes and ranges: PASS")

print("\nModel forward/backward verification: PASS")
