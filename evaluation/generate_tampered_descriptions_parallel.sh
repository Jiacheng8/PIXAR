#!/bin/bash
# ============================================================
# generate_tampered_descriptions_parallel.py — Multi-GPU
# text description generation for tampered images only.
#
# Usage:
#   bash evaluation/generate_tampered_descriptions_parallel.sh
#
# Before running, update VERSION, TYPE, and GPUS below.
# ============================================================
MAIN_DIR="/data/ironman/jiacheng/final_Omni_Data"
TYPE="full_0.05"
VERSION="finetune_PIXAR-7B_ours_seg-only_text0.1"
GPUS="0,1,2,3,4,5"             # ← 改成你想用的卡

VERSION_DIR="${MAIN_DIR}/ck/${VERSION}"
DATASET_DIR="${MAIN_DIR}/test/${TYPE}"
VISION_PRETRAINED="${MAIN_DIR}/ck/sam_vit_h_4b8939.pth"
OUTPUT_DIR="./evaluation/logs/${VERSION}_${TYPE}_tampered_descriptions"

# ---------- Settings ----------
PRECISION="bf16"
SPLIT="validation"
MAX_NEW_TOKENS=32
BATCH_SIZE=1       # samples per model.generate() call, lower if OOM
NUM_WORKERS=4      # DataLoader prefetch workers

# ---------- Run ----------
mkdir -p "${OUTPUT_DIR}"

python generate_tampered_descriptions_parallel.py \
  --version "${VERSION_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --split "${SPLIT}" \
  --precision "${PRECISION}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --batch_size "${BATCH_SIZE}" \
  --seg_prompt_mode seg_only \
  --num_workers "${NUM_WORKERS}" \
  --gpus "${GPUS}" \
  --use_mm_start_end \
  --train_mask_decoder \
  2>&1 | tee "${OUTPUT_DIR}/generate.log"
