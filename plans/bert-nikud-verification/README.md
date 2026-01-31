# bert-nikud Verification Plans

Verification scripts to test the bert-nikud Hebrew nikud prediction model.

## Overview

This directory contains verification plans and scripts to ensure the bert-nikud project works correctly. Each verification is self-contained and tests a specific aspect of the pipeline.

## Verification Scripts

| Script | Purpose |
|--------|---------|
| `001_verify_setup.py` | Environment setup, dependencies, tokenizer, data loading |
| `002_verify_model.py` | Model initialization, forward pass, prediction |
| `003_verify_training.py` | Training loop with mini run (5 steps) |
| `004_verify_inference.py` | Inference pipeline, batch processing, mixed text |

## Running All Verifications

```bash
# From bert-nikud directory
cd /home/yakov/Documents/audio/renikud-v3/bert-nikud

# Create sample data (10 lines)
cat > ../plans/bert-nikud-verification/sample_data.txt << 'EOF'
הָאִישׁ רָצָה
הָאִישָׁה הָלְכָה
הַיְלָדִים שָׂחֲקוּ
הַמִּלָּה הִיא טוֹבָה
הַסֵּפֶר נִמְצָא
הָעִיר גְּדוֹלָה
הַשָּׁמַיִם כְּחוּלִים
הַיּוֹם יָפֶה
הַלַּיְלָה חָשֵׁךְ
הָעוֹלָם יָפֶה
EOF

# Run each verification in order
python ../plans/bert-nikud-verification/001_verify_setup.py
python ../plans/bert-nikud-verification/002_verify_model.py
python ../plans/bert-nikud-verification/003_verify_training.py
python ../plans/bert-nikud-verification/004_verify_inference.py
```

## Individual Verification

Each script can be run independently (following dependencies):

```bash
# Setup verification
python 001_verify_setup.py

# Model verification (requires 001)
python 002_verify_model.py

# Training verification (requires 001, 002)
python 003_verify_training.py

# Inference verification (requires 001, 002, 003)
python 004_verify_inference.py
```

## Expected Duration

- 001: ~30 seconds
- 002: ~1 minute
- 003: ~2-3 minutes
- 004: ~1 minute

## Notes

- All scripts use the bert-nikud source code by adding it to sys.path
- Scripts create temporary files in the verification directory
- GPU is automatically detected and used if available
- Training verification only runs for 5 steps (not full training)
