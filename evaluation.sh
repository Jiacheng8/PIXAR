deepspeed --include localhost:0 --master_port=24997 test.py \
  --version="/data/ironman/jiacheng/Omni_data/ck/finetune_LISA-7B_masks_mask-only" \
  --dataset_dir='/data/ironman/jiacheng/Omni_data/final_dataset/evaluation_0.05' \
  --vision_pretrained="/data/ironman/jiacheng/Omni_data/ck/sam_vit_h_4b8939.pth" \
  --test_dataset="validation" \
  --precision='bf16' \
  --exp_name="evaluation-LISA-7B_masks_mask-only" \
  --test_only \
  > ./logs/evaluation-LISA-7B_masks_mask-only.log 2>&1
  
