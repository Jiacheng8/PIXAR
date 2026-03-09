#!/bin/bash

# Example script for running PIXAR Interactive Chat
# Usage: bash chat_example.sh
export CUDA_VISIBLE_DEVICES=4
python chat.py \
    --version path/to/PIXAR-7B\
    --vis_save_path ./vis_output \
    --precision bf16 \
    --image_size 1024 \
    --model_max_length 512 \
    --vision-tower openai/clip-vit-large-patch14 \
    --conv_type llava_v1 \
    --max_new_tokens 512 \
    --num_obj_classes 81 \
    --seg_prompt_mode seg_only \
    --use_mm_start_end \
    --generate_text_in_seg_only 
