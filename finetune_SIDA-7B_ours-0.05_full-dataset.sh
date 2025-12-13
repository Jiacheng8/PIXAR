deepspeed --include localhost:0 --master_port=12345 train_SIDA.py \
  --version="/data/ironman/jiacheng/Omni_data/ck/finetune_SIDA-7B_ours-0.05_full-dataset" \
  --dataset_dir='/data/ironman/jiacheng/Omni_data/final_dataset/ours_0.05' \
  --vision_pretrained="/data/ironman/jiacheng/Omni_data/ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/data/ironman/jiacheng/Omni_data/final_dataset/ours_0.05/validation" \
  --batch_size=2 \
  --exp_name="evaluate-SIDA-7B_ours-0.05_full-dataset" \
  --epochs=10 \
  --dice_loss_weight 1.0 \
  --steps_per_epoch=1000 \
  --precision="bf16" \
  --lr=0.0001 \
  --log_base_dir='/data/ironman/jiacheng/Omni_data/runs' \
  2> >(grep -v "libpng warning" >&2)
