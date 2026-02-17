#!/bin/bash
# ============================================================
# SIDA test.py evaluation script
#
# Usage:
#   bash test.sh
#
# Before running, update the paths below to match your setup.
# ============================================================

# ---------- Paths (modify these) ----------
VERSION="finetune_SIDA-7B_ours-text_only"
GPU=5
SEG_PROMPT_MODE="text_only"          # seg_only | text_only | fuse

VERSION_DIR="/data/ironman/jiacheng/final_Omni_Data/ck/${VERSION}"
DATASET_DIR="/data/ironman/jiacheng/final_Omni_Data/test/ours_0.05"
VISION_PRETRAINED="/data/ironman/jiacheng/final_Omni_Data/ck/sam_vit_h_4b8939.pth"
OUTPUT_DIR="./evaluation/${VERSION}"

# ---------- Settings ----------
PRECISION="bf16"
SPLIT="validation"
SEG_PROMPT_MODE="fuse"          # seg_only | text_only | fuse
OBJ_THRESHOLD=0.5
MAX_NEW_TOKENS=128

# ---------- Run ----------
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
  --version "${VERSION_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --split "${SPLIT}" \
  --precision "${PRECISION}" \
  --output_dir "${OUTPUT_DIR}" \
  --seg_prompt_mode "${SEG_PROMPT_MODE}" \
  --obj_threshold "${OBJ_THRESHOLD}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --save_generated_text \
  --use_mm_start_end \
  --train_mask_decoder \
  --save_generated_text \
  --text_output_file ./evaluation/${VERSION}/generated_text.jsonl \
  2>&1 | tee "${OUTPUT_DIR}/test.log"
