# 用 DeepSpeed 在单卡 GPU0 上跑 test.py 做“只评测不训练”的验证集评测。
# 使用方法：
# 1) 确认四个路径都存在：--version(模型目录)、--dataset_dir(数据集目录)、--vision_pretrained(SAM权重)、以及日志输出目录 ./logs/
# 2) 需要指定一张可用 GPU：--include localhost:0 表示只用 0 号卡；如果想用 1 号卡改成 localhost:1
# 3) master_port 需要是空闲端口；冲突就换一个（例如 24997 / 25000）
# 4) precision 选择 bf16；bf16 需要硬件支持（如 RTX4090和RTXA6000都可以）
# 5) 结果会写入 ./logs/*.log（stdout+stderr 都重定向进去），跑完后直接看该 log 即可

deepspeed --include localhost:0 --master_port=24996 test.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_SIDA-7B_masks_mask-only" \
  --dataset_dir='/data/ironman/jiacheng/final_Omni_Data/final_bench/evaluation_0.05' \
  --vision_pretrained="/data/ironman/jiacheng/final_Omni_Data/ck/sam_vit_h_4b8939.pth" \
  --test_dataset="validation" \
  --precision='bf16' \
  --exp_name="evaluation-SIDA-7B_masks_mask-only" \
  --test_only \
  > ./logs/evaluation-SIDA-7B_masks_mask-only.log 2>&1
  
