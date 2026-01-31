#!/usr/bin/env python3
"""Export menaked checkpoint to ONNX using renikud-onnx export."""

from __future__ import annotations

from pathlib import Path
import sys
import torch
from safetensors.torch import save_file

base_dir = Path(__file__).resolve().parents[2] / "bert-nikud"
export_dir = base_dir / "renikud-onnx"
sys.path.insert(0, str(export_dir))

from export import export_to_onnx


def main():
    ckpt_path = (
        Path(__file__).resolve().parents[1]
        / "bert-nikud-menaked-verification"
        / "checkpoints"
        / "5min_menaked.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = out_dir / "5min_menaked.safetensors"
    onnx_path = out_dir / "5min_menaked.onnx"

    state_dict = torch.load(ckpt_path, map_location="cpu")
    save_file(state_dict, safetensors_path)

    export_to_onnx(str(safetensors_path), str(onnx_path))

    print(f"Saved ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
