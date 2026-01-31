#!/usr/bin/env python3
"""Verify model initialization and forward pass."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bert-nikud", "src")
)

import torch
from transformers import AutoTokenizer
from model import HebrewNikudModel, count_parameters

print("=" * 60)
print("STEP 1: Initializing model")
print("=" * 60)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = HebrewNikudModel(model_name="dicta-il/dictabert-large-char")
    print("✓ Model initialized")

    total_params, trainable_params = count_parameters(model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 2: Verifying classification heads")
print("=" * 60)
expected_heads = [
    "vowel_classifier",
    "dagesh_classifier",
    "sin_classifier",
    "stress_classifier",
    "prefix_classifier",
]
for head_name in expected_heads:
    if hasattr(model, head_name):
        head = getattr(model, head_name)
        print(f"✓ {head_name}: {head.in_features} -> {head.out_features}")
    else:
        print(f"✗ Missing: {head_name}")
        sys.exit(1)
print()

print("=" * 60)
print("STEP 3: Testing forward pass (no labels)")
print("=" * 60)
try:
    tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-large-char")
    text = "האיש רצה"
    encoding = tokenizer(text, return_tensors="pt")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    print("✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Vowel logits shape: {outputs['vowel_logits'].shape}")
    print(f"  Dagesh logits shape: {outputs['dagesh_logits'].shape}")
    print(f"  Sin logits shape: {outputs['sin_logits'].shape}")
    print(f"  Stress logits shape: {outputs['stress_logits'].shape}")
    print(f"  Prefix logits shape: {outputs['prefix_logits'].shape}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 4: Testing forward pass (with labels)")
print("=" * 60)
try:
    batch_size, seq_len = input_ids.shape

    # Create dummy labels
    vowel_labels = torch.randint(0, 8, (batch_size, seq_len)).to(device)
    dagesh_labels = torch.randint(0, 2, (batch_size, seq_len)).to(device)
    sin_labels = torch.randint(0, 2, (batch_size, seq_len)).to(device)
    stress_labels = torch.randint(0, 2, (batch_size, seq_len)).to(device)
    prefix_labels = torch.randint(0, 2, (batch_size, seq_len)).to(device)

    outputs = model(
        input_ids,
        attention_mask,
        vowel_labels=vowel_labels,
        dagesh_labels=dagesh_labels,
        sin_labels=sin_labels,
        stress_labels=stress_labels,
        prefix_labels=prefix_labels,
    )

    print("✓ Forward pass with labels successful")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Vowel loss: {outputs['vowel_loss'].item():.4f}")
    print(f"  Dagesh loss: {outputs['dagesh_loss'].item():.4f}")
    print(f"  Sin loss: {outputs['sin_loss'].item():.4f}")
    print(f"  Stress loss: {outputs['stress_loss'].item():.4f}")
    print(f"  Prefix loss: {outputs['prefix_loss'].item():.4f}")
except Exception as e:
    print(f"✗ Forward pass with labels error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("STEP 5: Testing predict method")
print("=" * 60)
try:
    predictions = model.predict(input_ids, attention_mask)

    print("✓ Predictions successful")
    print(
        f"  Vowel shape: {predictions['vowel'].shape} (values: {predictions['vowel'].unique().tolist()})"
    )
    print(
        f"  Dagesh shape: {predictions['dagesh'].shape} (values: {predictions['dagesh'].unique().tolist()})"
    )
    print(
        f"  Sin shape: {predictions['sin'].shape} (values: {predictions['sin'].unique().tolist()})"
    )
    print(
        f"  Stress shape: {predictions['stress'].shape} (values: {predictions['stress'].unique().tolist()})"
    )
    print(
        f"  Prefix shape: {predictions['prefix'].shape} (values: {predictions['prefix'].unique().tolist()})"
    )
except Exception as e:
    print(f"✗ Prediction error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("VERIFICATION COMPLETE ✓")
print("=" * 60)
