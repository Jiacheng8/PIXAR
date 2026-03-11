#!/bin/bash
# ============================================================
# PIXAR test.py evaluation script — loops over 6 datasets
#
# Usage:
#   bash evaluation_PIXAR-7B.sh
#
# Before running, update the paths below to match your setup.
# ============================================================

# ---------- Paths (modify these) ----------
MAIN_DIR="/data/ironman/jiacheng/final_Omni_Data"
VERSION="finetune_PIXAR-7B_ours_seg-only_text3.0"
GPU="1"
SEG_PROMPT_MODE="seg_only"          # seg_only | text_only | fuse

VERSION_DIR="${MAIN_DIR}/ck/${VERSION}"
VISION_PRETRAINED="${MAIN_DIR}/ck/sam_vit_h_4b8939.pth"

# ---------- Settings ----------
PRECISION="bf16"
SPLIT="validation"
OBJ_THRESHOLD=0.5
MAX_NEW_TOKENS=128

# ---------- Datasets ----------
TYPES=(
    "gpt_0.05"
    "seedream_0.05"
    "gemini3_0.05"
    "flux2_0.05"
    "gemini_0.05"
    "qwen_0.05"
)

# ---------- Run ----------
TOTAL=${#TYPES[@]}
for i in "${!TYPES[@]}"; do
    TYPE="${TYPES[$i]}"
    IDX=$((i + 1))

    DATASET_DIR="${MAIN_DIR}/test/${TYPE}"
    OUTPUT_DIR="./evaluation/logs/${VERSION}_${TYPE}"

    echo "========================================================"
    echo "[${IDX}/${TOTAL}] Dataset: ${TYPE}"
    echo "  DATASET_DIR : ${DATASET_DIR}"
    echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
    echo "========================================================"

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

    EXIT_CODE=${PIPESTATUS[0]}
    if [ "${EXIT_CODE}" -ne 0 ]; then
        echo "[WARN] Dataset ${TYPE} exited with code ${EXIT_CODE}. Continuing..."
    fi
done

echo "========================================================"
echo "All ${TOTAL} datasets finished."
echo "========================================================"
