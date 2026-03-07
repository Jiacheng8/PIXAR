export CUDA_VISIBLE_DEVICES=5
python visualization.py \
  --dataset_dir /data/ironman/jiacheng/Omni_data/final_dataset/subset_visualization_sub\
  --version /data/ironman/jiacheng/Omni_data/ck/finetune_PIXAR-7B_masks_mask-only\
  --vision_pretrained="/data/ironman/jiacheng/Omni_data/ck/sam_vit_h_4b8939.pth" \
  --vis_root /data/ironman/jiacheng/Omni_data/vis_output_pixar \
  --split validation \
  --precision bf16 \
  --batch_size 1