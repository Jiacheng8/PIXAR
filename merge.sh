CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/SIDA-7B" \
  --weight="/data/ironman/jiacheng/final_Omni_Data/runs/finetune_PIXAR-7B_tao0.1/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_PIXAR-7B_tao0.1"