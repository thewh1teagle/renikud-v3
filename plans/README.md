# Verification Plans Overview

This directory contains verification plans for sub-projects in renikud-v3.

## Available Plans

### bert-nikud-verification
Verifies the BERT-based Hebrew nikud prediction model.

**Location**: `bert-nikud-verification/`

**Steps**:
1. Setup and data loading verification
2. Model initialization verification
3. Training verification (mini-run, 5 steps)
4. Inference verification

**Quick Start**:
```bash
cd bert-nikud
python ../plans/bert-nikud-verification/001_verify_setup.py
python ../plans/bert-nikud-verification/002_verify_model.py
python ../plans/bert-nikud-verification/003_verify_training.py
python ../plans/bert-nikud-verification/004_verify_inference.py
```

See `bert-nikud-verification/README.md` for details.
