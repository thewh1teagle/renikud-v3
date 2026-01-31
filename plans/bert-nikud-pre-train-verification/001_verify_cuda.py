"""Verify CUDA is available and working for training."""

import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("FAIL: CUDA is not available")
    sys.exit(1)

print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Quick tensor test on GPU
x = torch.randn(1000, 1000, device="cuda")
y = x @ x.T
assert y.shape == (1000, 1000)
print("GPU matmul test: PASS")

print("\nCUDA verification: PASS")
