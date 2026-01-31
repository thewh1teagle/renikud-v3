# bert-nikud Menaked Tokenizer Verification

## Goal
Verify Dicta menaked tokenizer integration with bert-nikud src:
- tokenization + offsets
- label alignment
- short training run (5 minutes) + inference

## Scripts
- `001_verify_tokenization.py`
- `002_verify_alignment.py`
- `003_verify_training_minutes.py`

## Run
```bash
uv run --project bert-nikud plans/bert-nikud-menaked-verification/001_verify_tokenization.py
uv run --project bert-nikud plans/bert-nikud-menaked-verification/002_verify_alignment.py
uv run --project bert-nikud plans/bert-nikud-menaked-verification/003_verify_training_minutes.py --minutes 1
```
