# 用 DeepSpeed 在单卡 GPU 上对 PIXAR-7B 进行微调训练（含分类 / 分割 / OBJ 多任务）。
# 使用说明：
# 1) GPU 选择：
#    --include localhost:2 表示只使用第 2 张 GPU；如果想换卡，改成 localhost:0/1/3 等。
# 2) 端口设置：
#    --master_port 需保证当前机器未被占用；冲突时报错就换一个端口号。
# 3) 模型与数据：
#    --version           指向 PIXAR 的初始权重目录（HF 格式）
#    --dataset_dir       训练数据根目录（需包含 train/validation 划分或相应结构）
#    --val_dataset       验证集路径（用于周期性验证与保存最优 checkpoint）
#    --vision_pretrained SAM ViT-H 权重路径（用于分割模块）
# 4) 训练配置：
#    --batch_size        每卡 micro-batch size（总 batch = batch_size × grad_accumulation_steps）
#    --epochs            训练轮数
#    --steps_per_epoch   每个 epoch 的训练步数（与数据量/采样策略相关）
#    --lr                学习率（bf16 下建议 1e-4 或更小）
#    --dice_loss_weight  Dice loss 在分割任务中的权重
#    --precision         计算精度；bf16 需要硬件支持
# 5) 日志与输出：
#    --exp_name          实验名称（用于 runs 目录下区分不同实验）
#    --log_base_dir      TensorBoard 与 checkpoint 的保存根目录

################################################################################
# Key parameters — edit here
################################################################################
BASE_DIR="/data/ironman/jiacheng/final_Omni_Data"

GPU="localhost:1"
PORT=12448
VERSION="${BASE_DIR}/ck/PIXAR-7B"
DATASET_DIR="${BASE_DIR}/train/ours_0.05"
VAL_DATASET="${BASE_DIR}/train/ours_0.05/validation"
VISION_PRETRAINED="${BASE_DIR}/ck/sam_vit_h_4b8939.pth"
LOG_BASE_DIR="${BASE_DIR}/runs"
EXP_NAME="finetune_PIXAR-7B_ours_fuse_text0.1"

BATCH_SIZE=2
EPOCHS=20
STEPS_PER_EPOCH=1000
LR=0.0001
PRECISION="bf16"

DICE_LOSS_WEIGHT=1.0
OBJ_LOSS_WEIGHT=0.5
TEXT_LOSS_WEIGHT=0.1
SEG_PROMPT_MODE="fuse"
MASK_TYPE="ours"          # "ours" -> gt_soft_mask, "others" -> gt_mask
################################################################################

mkdir -p ./finetune/logs

deepspeed --include ${GPU} --master_port=${PORT} train_PIXAR.py \
  --version="${VERSION}" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${VISION_PRETRAINED}" \
  --val_dataset="${VAL_DATASET}" \
  --batch_size=${BATCH_SIZE} \
  --exp_name="${EXP_NAME}" \
  --epochs=${EPOCHS} \
  --dice_loss_weight ${DICE_LOSS_WEIGHT} \
  --obj_loss_weight ${OBJ_LOSS_WEIGHT} \
  --mask_type "${MASK_TYPE}" \
  --seg_prompt_mode "${SEG_PROMPT_MODE}" \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --precision="${PRECISION}" \
  --lr=${LR} \
  --text_loss_weight ${TEXT_LOSS_WEIGHT} \
  --no_eval \
  --log_base_dir="${LOG_BASE_DIR}" > ./finetune/logs/${EXP_NAME}.log 2>&1 &
