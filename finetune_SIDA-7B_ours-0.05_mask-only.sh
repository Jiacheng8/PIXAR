deepspeed --include localhost:3 --master_port=12348 train_SIDA.py \
  --version="/data/ironman/jiacheng/Omni_data/ck/SIDA-7B" \
  --dataset_dir='/data/ironman/jiacheng/Omni_data/final_dataset/ours_w-mask_0.05' \
  --vision_pretrained="/data/ironman/jiacheng/Omni_data/ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/data/ironman/jiacheng/Omni_data/final_dataset/ours_w-mask_0.05/validation" \
  --batch_size=2 \
  --exp_name="finetune_SIDA-7B_ours-0.05_mask-only" \
  --epochs=10 \
  --dice_loss_weight 1.0 \
  --steps_per_epoch=1000 \
  --precision="bf16" \
  --lr=0.0001 \
  --log_base_dir='/data/ironman/jiacheng/Omni_data/runs' \
  2> >(grep -v "libpng warning" >&2)
