#!/usr/bin/env python3
"""Verify training loop with mini run."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bert-nikud", "src")
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import HebrewNikudModel
from dataset import load_dataset_from_file, split_dataset, collate_fn
from tqdm import tqdm

print("=" * 60)
print("STEP 1: Loading data for mini training")
print("=" * 60)
try:
    sample_file = os.path.join(os.path.dirname(__file__), "train_small.txt")
    texts = load_dataset_from_file(sample_file)
    train_texts, eval_texts = split_dataset(texts, eval_max_lines=5, seed=42)
    print(f"✓ Loaded {len(train_texts)} train texts, {len(eval_texts)} eval texts")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 2: Creating dataset and dataloader")
print("=" * 60)
try:
    from dataset import NikudDataset

    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
    train_dataset = NikudDataset(train_texts, tokenizer, use_cache=False)
    eval_dataset = NikudDataset(eval_texts, tokenizer, use_cache=False)

    print(f"✓ Train dataset size: {len(train_dataset)}")
    print(f"✓ Eval dataset size: {len(eval_dataset)}")

    # Test a single sample
    sample = train_dataset[0]
    print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"  Sample vowel_labels shape: {sample['vowel_labels'].shape}")

    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(
        f"✓ DataLoader created (batch_size={batch_size}, {len(train_loader)} batches)"
    )
except Exception as e:
    print(f"✗ Dataset creation error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 3: Initializing model and optimizer")
print("=" * 60)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = HebrewNikudModel(model_name="dicta-il/dictabert-large-char")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("✓ Model and optimizer initialized")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 4: Training for 5 steps")
print("=" * 60)
try:
    model.train()
    losses = []

    for step, batch in enumerate(tqdm(train_loader, total=5, desc="Training")):
        if step >= 5:
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

        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")

    print("\n✓ Training completed successfully")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss decrease: {losses[0] - losses[-1]:.4f}")
except Exception as e:
    print(f"✗ Training error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 5: Saving and loading checkpoint")
print("=" * 60)
try:
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pt")

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Load checkpoint
    model2 = HebrewNikudModel()
    model2.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model2.to(device)
    print("✓ Checkpoint loaded successfully")

    # Verify parameters match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Parameters don't match after loading!"
    print("✓ Parameters verified to match")
except Exception as e:
    print(f"✗ Checkpoint error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 6: Quick evaluation")
print("=" * 60)
try:
    model.eval()
    eval_losses = []

    eval_loader = DataLoader(
        eval_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

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
            break  # Just one batch for verification

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    print("✓ Evaluation completed")
    print(f"  Eval loss: {avg_eval_loss:.4f}")
except Exception as e:
    print(f"✗ Evaluation error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
print("Training pipeline is functional!")
