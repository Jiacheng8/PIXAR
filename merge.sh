CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/SIDA-7B" \
  --weight="/data/ironman/jiacheng/final_Omni_Data/runs/finetune_SIDA-7B_ours-text_only/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_SIDA-7B_ours-text_only"