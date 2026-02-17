#!/bin/bash

# Example script for running SIDA Interactive Chat
# Usage: bash chat_example.sh
export CUDA_VISIBLE_DEVICES=0
python chat.py \
    --version /data/ironman/jiacheng/final_Omni_Data/ck/finetune_SIDA-7B_ours-0.05_seg\
    --vis_save_path ./vis_output \
    --precision bf16 \
    --image_size 1024 \
    --model_max_length 512 \
    --vision-tower openai/clip-vit-large-patch14 \
    --conv_type llava_v1 \
    --max_new_tokens 512 \
    --num_obj_classes 81 \
    --use_mm_start_end
