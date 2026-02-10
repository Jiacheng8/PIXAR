# #!/usr/bin/env bash
# # set -euo pipefail

# cd /home/jiacheng/Omni_detection/utils/construct_dataset

# train_w_anno_ids=(
#   coco_train_inter_replacement_1
#   coco_train_inter_replacement_2
#   coco_train_removal_1
#   coco_train_replacement_1
#   coco_train_replacement_2
# )

# train_wo_anno_ids=(
#   coco_train_addition
#   coco_train_background
#   coco_train_color
#   coco_train_motion
#   coco_train_material
# )


# val_w_anno_ids=(
#   coco_val_inter_replacement_1
#   coco_val_inter_replacement_2
#   coco_val_removal_1
#   coco_val_replacement_1
#   coco_val_replacement_2
# )

# val_wo_anno_ids=(
#   coco_val_addition
#   coco_val_background
#   coco_val_color
#   coco_val_motion
#   coco_val_material
# )

# # test data construction
# taos=(0.05 0.1 0.2)

# for t in "${taos[@]}"; do
#   # generate set w/ anno
#   for id in "${val_w_anno_ids[@]}"; do
#     echo "Running id=${id}, tao=${t}"
#     python 2_construct_dataset_w-anno.py \
#       --id "${id}" \
#       --tao "${t}" \
#       --dataset-dir /data/ironman/jiacheng/final_Omni_Data/raw_outputs \
#       --output-dir "/data/ironman/jiacheng/final_Omni_Data/final_bench/evaluation" \
#       --dest-type validation
#   done

#   # generate set w/o anno
#   for id in "${val_wo_anno_ids[@]}"; do
#     echo "Running id=${id}, tao=${t}"
#     python 2_construct_dataset_wo-anno.py \
#       --id "${id}" \
#       --tao "${t}" \
#       --dataset-dir /data/ironman/jiacheng/final_Omni_Data/raw_outputs \
#       --output-dir "/data/ironman/jiacheng/final_Omni_Data/final_bench/evaluation" \
#       --dest-type validation
#   done
# done


#!/usr/bin/env bash
set -u  # 不用 -e，避免中途一个失败直接把整批停掉（更适合跑大批量）
set -o pipefail

cd /home/jiacheng/Omni_detection/utils/construct_dataset || exit 1

# -------------------------
# Config
# -------------------------
DATASET_DIR="/data/ironman/jiacheng/final_Omni_Data/raw_outputs"
OUT_DIR="/data/ironman/jiacheng/final_Omni_Data/train/ours"
DEST_TYPE="train"

TAOS=(0.05)

train_w_anno_ids=(
  coco_train_inter_replacement_1
  coco_train_inter_replacement_2
  coco_train_replacement_1
  coco_train_replacement_2
)

train_w_anno_bg_ids=(
  coco_train_removal_1
)

train_wo_anno_ids=(
  coco_train_addition
  coco_train_color
  coco_train_motion
  coco_train_material
)

train_wo_anno_bg_ids=(
  coco_train_background
)

# -------------------------
# Logging helpers
# -------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/construct_${DEST_TYPE}_${RUN_ID}.log"

ts() { date +"%F %T"; }

h1() {
  echo -e "\n========================================" | tee -a "$LOG_FILE"
  echo -e "🚀 $1" | tee -a "$LOG_FILE"
  echo -e "========================================" | tee -a "$LOG_FILE"
}

h2() {
  echo -e "\n----------------------------------------" | tee -a "$LOG_FILE"
  echo -e "📌 $1" | tee -a "$LOG_FILE"
  echo -e "----------------------------------------" | tee -a "$LOG_FILE"
}

log() {
  # usage: log "message"
  echo "[$(ts)] $1" | tee -a "$LOG_FILE"
}

# -------------------------
# Runner
# -------------------------
OK=0
FAIL=0

run_one () {
  local script="$1"
  local id="$2"
  local tao="$3"

  log "▶️  Start: script=${script} | id=${id} | tao=${tao} | dest=${DEST_TYPE}"
  local start_ts end_ts dur

  start_ts=$(date +%s)

  # stdout+stderr 全进 log，同时在终端显示
  python "$script" \
    --id "$id" \
    --tao "$tao" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUT_DIR" \
    --dest-type "$DEST_TYPE" \
    2>&1 | tee -a "$LOG_FILE"

  local rc=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))

  if [[ $rc -eq 0 ]]; then
    OK=$((OK + 1))
    log "✅ Done: id=${id} tao=${tao} (${dur}s)"
  else
    FAIL=$((FAIL + 1))
    log "❌ Failed(rc=${rc}): id=${id} tao=${tao} (${dur}s)"
    # 打印最后 30 行，方便快速定位（只输出到终端 & log）
    log "🧾 Tail(30) for failure: id=${id} tao=${tao}"
    tail -n 30 "$LOG_FILE" | sed 's/^/    /' | tee -a "$LOG_FILE"
  fi
}

# -------------------------
# Main
# -------------------------
h1 "Construct Dataset Batch (run_id=${RUN_ID})"
log "📂 workdir=$(pwd)"
log "📥 dataset_dir=${DATASET_DIR}"
log "📦 output_dir=${OUT_DIR}"
log "🧩 dest_type=${DEST_TYPE}"
log "🧪 taos=${TAOS[*]}"
log "📝 log_file=${LOG_FILE}"

for tao in "${TAOS[@]}"; do
  h2 "TAO=${tao} | Train w/ anno"
  for id in "${train_w_anno_ids[@]}"; do
    run_one "2_construct_dataset_w-anno.py" "$id" "$tao"
  done

  h2 "TAO=${tao} | Train w/o anno"
  for id in "${train_wo_anno_ids[@]}"; do
    run_one "2_construct_dataset_wo-anno.py" "$id" "$tao"
  done

  h2 "TAO=${tao} | Train w/ anno bg"
  for id in "${train_w_anno_bg_ids[@]}"; do
    run_one "2_construct_dataset_w-anno_bg.py" "$id" "$tao"
  done

  h2 "TAO=${tao} | Train w/o anno bg"
  for id in "${train_wo_anno_bg_ids[@]}"; do
    run_one "2_construct_dataset_wo-anno_bg.py" "$id" "$tao"
  done
done

h1 "Summary"
log "📊 Total: $((OK + FAIL)) | ✅ OK=${OK} | ❌ FAIL=${FAIL}"
log "✅ Done. Log saved to: ${LOG_FILE}"

# 如果你想：失败就让脚本最后返回非 0（用于 CI 或上层监控）
if [[ $FAIL -ne 0 ]]; then
  exit 1
fi