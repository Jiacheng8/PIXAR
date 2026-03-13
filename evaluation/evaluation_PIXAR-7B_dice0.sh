#!/bin/bash
# ============================================================
# PIXAR test.py evaluation script
#
# Usage:
#   bash evaluation1.sh
#
# Before running, update the paths below to match your setup.
# ============================================================

# ---------- Paths (modify these) ----------
MAIN_DIR="/data/ironman/jiacheng/final_Omni_Data"
TYPE="full_0.05"  # gpt_0.05 | ours_0.05 | gemini3_0.05
VERSION="finetune_PIXAR-7B_ours_seg-only_dice0.0"
GPU="2,3"
SEG_PROMPT_MODE="seg_only"          # seg_only | text_only | fuse

VERSION_DIR="${MAIN_DIR}/ck/${VERSION}"
DATASET_DIR="${MAIN_DIR}/test/${TYPE}"
VISION_PRETRAINED="${MAIN_DIR}/ck/sam_vit_h_4b8939.pth"
OUTPUT_DIR="./evaluation/logs/${VERSION}_${TYPE}"

# ---------- Settings ----------
PRECISION="bf16"
SPLIT="validation"
OBJ_THRESHOLD=0.5
MAX_NEW_TOKENS=128

# ---------- Run ----------
mkdir -p "${OUTPUT_DIR}"

python test_parallel.py \
  --version "${VERSION_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --split "${SPLIT}" \
  --precision "${PRECISION}" \
  --output_dir "${OUTPUT_DIR}" \
  --seg_prompt_mode "${SEG_PROMPT_MODE}" \
  --obj_threshold "${OBJ_THRESHOLD}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --gpus "${GPU}" \
  --save_generated_text \
  --use_mm_start_end \
  --train_mask_decoder \
  --load_in_8bit \
  2>&1 | tee "${OUTPUT_DIR}/test.log"