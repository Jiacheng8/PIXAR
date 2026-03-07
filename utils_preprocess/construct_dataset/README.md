# Dataset Construction

This directory contains scripts for building the PIXAR dataset from raw image pairs at a chosen pixel-difference threshold τ.

## Files

| File | Description |
|---|---|
| `2_construct_dataset.py` | Core processing script — mask-only supervision (no text descriptions) |
| `2_construct_dataset_text.py` | Core processing script — with text descriptions from a CSV |
| `generate_v2.sh` | Batch runner for **training set** using `2_construct_dataset.py` |
| `generate_v2-text.sh` | Batch runner for **training set** using `2_construct_dataset_text.py` |
| `generate_v2-text-val.sh` | Batch runner for **test set** (single generative source) using `2_construct_dataset_text.py` |
| `generate_v2-text-val-*.sh` | Per-source test set runners (gemini, gemini3, gpt, flux2, qwen, seedream) |

---

## Step 1 — Build the training set

Open `generate_v2.sh` (mask-only) or `generate_v2-text.sh` (with text descriptions) and set the config block at the top:

```bash
DATASET_DIR="/path/to/raw_outputs"        # downloaded raw image pairs
OUT_DIR="/path/to/output/train/ours"      # where to write the processed dataset
TAOS=(0.05)                               # one or more τ values, e.g. (0.01 0.05 0.1)

# only for generate_v2-text.sh:
DESCRIPTIONS_CSV="/path/to/descriptions.csv"
```

Then run from this directory:

```bash
cd utils_preprocess/construct_dataset

# mask-only labels
bash generate_v2.sh

# labels + text descriptions
bash generate_v2-text.sh
```

The script processes all training sub-splits (`replacement`, `removal`, `addition`, `color`, `motion`, `material`, `background`) and an optional validation mock split, logging output to `logs/`.

---

## Step 2 — Build the test set

Use `generate_v2-text-val.sh` for a single generative source, or the per-source scripts for specific models.

### Single source

Edit the `TYPE` variable at the top of `generate_v2-text-val.sh`:

```bash
TYPE="qwen"   # one of: gemini | gemini3 | gpt | flux2 | qwen | seedream
TAOS=(0.05)
DATASET_DIR="/path/to/raw_outputs"
OUT_DIR="/path/to/output/test/${TYPE}"
DESCRIPTIONS_CSV="/path/to/descriptions.csv"
```

Then run:

```bash
bash generate_v2-text-val.sh
```

### All sources at once

Use the per-source scripts in sequence:

```bash
bash generate_v2-text-val-qwen.sh
bash generate_v2-text-val-gpt.sh
bash generate_v2-text-val-gemini.sh
bash generate_v2-text-val-gemini3.sh
bash generate_v2-text-val-flux2.sh
bash generate_v2-text-val-seedream.sh
```

---

## τ selection guide

| τ | Effect |
|---|---|
| 0.01 | Captures micro-edits and subtle pixel changes |
| 0.05 | Default — balanced sensitivity (recommended) |
| 0.1  | High-confidence semantic changes only |
| 0.2  | Conservative — only large, obvious edits |

Multiple values can be processed in one run by setting `TAOS=(0.01 0.05 0.1)`.

---

## Output structure

```
OUT_DIR/
├── train/
│   ├── real/
│   ├── full_synthetic/
│   ├── tampered/
│   ├── masks/           # hard binary masks
│   ├── soft_masks/      # pixel-difference maps M_τ at chosen τ
│   └── metadata/        # JSON: {"cls": [...], "text": "..."}
└── validation/
    └── (same structure)
```
