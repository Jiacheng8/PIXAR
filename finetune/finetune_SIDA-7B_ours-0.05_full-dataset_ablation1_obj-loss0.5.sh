# 用 DeepSpeed 在单卡 GPU2 上对 SIDA-7B 进行微调训练（含分类 / 分割 / OBJ 多任务）。
# 使用说明：
# 1) GPU 选择：
#    --include localhost:2 表示只使用第 2 张 GPU；如果想换卡，改成 localhost:0/1/3 等。
# 2) 端口设置：
#    --master_port 需保证当前机器未被占用；冲突时报错就换一个端口号。
# 3) 模型与数据：
#    --version           指向 SIDA 的初始权重目录（HF 格式）
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

mkdir -p ./finetune/logs


deepspeed --include localhost:0 --master_port=12346 train_SIDA.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/SIDA-7B" \
  --dataset_dir='/data/ironman/jiacheng/final_Omni_Data/train/ours_0.05' \
  --vision_pretrained="/data/ironman/jiacheng/final_Omni_Data/ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/data/ironman/jiacheng/final_Omni_Data/train/ours_0.05/validation" \
  --batch_size=2 \
  --exp_name="finetune_SIDA-7B_ours-0.05_full-dataset_ablation1_obj-loss0.5" \
  --epochs=100 \
  --dice_loss_weight 1.0 \
  --obj_loss_weight 0.5 \
  --steps_per_epoch=1000 \
  --precision="bf16" \
  --lr=0.0001 \
  --log_base_dir='/data/ironman/jiacheng/final_Omni_Data/runs' > ./finetune/logs/finetune_SIDA-7B_ours-0.05_full-dataset_ablation1_obj-loss0.5.log 2>&1 &
