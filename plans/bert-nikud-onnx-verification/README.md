# bert-nikud ONNX Verification

## Goal
Export the latest trained checkpoint to ONNX and verify inference using
the menaked tokenizer.

## Scripts
- `001_export_onnx.py`
- `002_infer_onnx.py`

## Run
```bash
uv run --project bert-nikud/renikud-onnx --group dev plans/bert-nikud-onnx-verification/001_export_onnx.py
uv run --project bert-nikud/renikud-onnx plans/bert-nikud-onnx-verification/002_infer_onnx.py
```
