CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/workspace/base/ck/PIXAR-13B" \
  --weight="/workspace/base/runs/finetune_PIXAR-7B_ours_seg-only_text4.0/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_PIXAR-7B_ours_seg-only_text4.0"