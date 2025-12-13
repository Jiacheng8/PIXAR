CUDA_VISIBLE_DEVICES="2" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/ironman/jiacheng/Omni_data/ck/SIDA-7B" \
  --weight="/data/ironman/jiacheng/Omni_data/runs/finetune_SIDA-7B_ours-0.05_full-dataset_ablation1_obj-loss0.5/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/Omni_data/ck/finetune_SIDA-7B_ours-0.05_full-dataset_ablation2_obj-loss0.5"