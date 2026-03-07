CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/PIXAR-7B" \
  --weight="/data/ironman/jiacheng/final_Omni_Data/runs/finetune_PIXAR-7B_ours_seg-only_text3.0/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_PIXAR-7B_ours_seg-only_text3.0"