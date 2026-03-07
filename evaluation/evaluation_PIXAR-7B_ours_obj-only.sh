#!/bin/bash
# ============================================================
# test_obj_only.py evaluation script
#
# Evaluates only the OBJ token's 81-class classification accuracy.
# No segmentation, no text generation.
#
# Usage:
#   bash evaluation/evaluation_PIXAR-7B_ours_obj-only.sh
#
# Two OBJ eval modes:
#   USE_MODEL_CLS=false  (default) : evaluate on all GT-tampered samples
#   USE_MODEL_CLS=true             : evaluate only when model predicts tampered
# ============================================================

MAIN_DIR="/data/ironman/jiacheng/final_Omni_Data"
TYPE="full_0.05"
VERSION="finetune_PIXAR-7B_ours_fuse"
GPU=1
SEG_PROMPT_MODE="fuse"      # seg_only | text_only | fuse
USE_MODEL_CLS=false         # true: end-to-end mode (model decides tampered first)

VERSION_DIR="${MAIN_DIR}/ck/${VERSION}"
DATASET_DIR="${MAIN_DIR}/test/${TYPE}"
VISION_PRETRAINED="${MAIN_DIR}/ck/sam_vit_h_4b8939.pth"
OUTPUT_DIR="./evaluation/logs/${VERSION}_${TYPE}_obj-only"

# ---------- Settings ----------
PRECISION="bf16"
SPLIT="validation"
OBJ_THRESHOLD=0.5
MAX_NEW_TOKENS=128

# ---------- Run ----------
mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS=""
if [ "${USE_MODEL_CLS}" = "true" ]; then
    EXTRA_FLAGS="--use_model_cls"
    OUTPUT_DIR="${OUTPUT_DIR}_model-cls"
    mkdir -p "${OUTPUT_DIR}"
fi

CUDA_VISIBLE_DEVICES="${GPU}" python test_obj_only.py \
  --version "${VERSION_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --split "${SPLIT}" \
  --precision "${PRECISION}" \
  --output_dir "${OUTPUT_DIR}" \
  --seg_prompt_mode "${SEG_PROMPT_MODE}" \
  --obj_threshold "${OBJ_THRESHOLD}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --use_mm_start_end \
  --train_mask_decoder \
  ${EXTRA_FLAGS} \
  2>&1 | tee "${OUTPUT_DIR}/test.log"
