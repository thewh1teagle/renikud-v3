#!/usr/bin/env python3
"""Infer with exported ONNX model using menaked tokenizer."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import onnxruntime
from tokenizers import Tokenizer

base_dir = Path(__file__).resolve().parents[2] / "bert-nikud"
sys.path.insert(0, str(base_dir / "renikud-onnx" / "src"))

from renikud_onnx.decode import decode_predictions


def load_samples(limit: int = 5) -> list[str]:
    val_file = base_dir / "dataset" / "val.txt"
    samples = []
    if val_file.exists():
        with val_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
                if len(samples) >= limit:
                    break
    return samples


def main():
    onnx_path = Path(__file__).resolve().parent / "artifacts" / "5min_menaked.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    tokenizer_path = (
        base_dir / "tokenizer" / "dictabert-large-char-menaked" / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    session = onnxruntime.InferenceSession(str(onnx_path))

    samples = load_samples(limit=5)
    print(f"ONNX: {onnx_path}")
    print(f"Tokenizer: {tokenizer_path}")

    for text in samples:
        encoding = tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        outputs = session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        predicted_text = decode_predictions(outputs, input_ids, tokenizer)

        print()
        print(f"Input:  {text}")
        print(f"Output: {predicted_text}")


if __name__ == "__main__":
    main()
