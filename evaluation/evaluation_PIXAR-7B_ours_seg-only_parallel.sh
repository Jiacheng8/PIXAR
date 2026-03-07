
#!/bin/bash
# ============================================================
# test_parallel.py evaluation script (multi-GPU)
#
# Usage:
#   bash evaluation/evaluation_PIXAR-7B_ours_fuse_parallel.sh
#
# Before running, update GPUS and TYPE below.
# ============================================================
MAIN_DIR="/data/ironman/jiacheng/final_Omni_Data"
TYPE="full_0.05"
VERSION="finetune_PIXAR-13B_ours_seg-only"
# GPUS="0,1|2,3|4,5"             # 3 workers，每个 worker 用 2 张卡
# GPUS="0,1|2,3"             # 2 workers，每个 worker 用 2 张卡
GPUS="0|1|2|3"             # 4 workers，每个 worker 用 1 张卡（需配合 --load_in_8bit）
SEG_PROMPT_MODE="seg_only"     # seg_only | text_only | fuse

VERSION_DIR="${MAIN_DIR}/ck/${VERSION}"
DATASET_DIR="${MAIN_DIR}/test/${TYPE}"
VISION_PRETRAINED="${MAIN_DIR}/ck/sam_vit_h_4b8939.pth"
OUTPUT_DIR="./evaluation/logs/${VERSION}_${TYPE}_parallel"

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
  --gpus "${GPUS}" \
  --save_generated_text \
  --use_mm_start_end \
  --train_mask_decoder \
  --load_in_8bit \
  2>&1 | tee "${OUTPUT_DIR}/test.log"
