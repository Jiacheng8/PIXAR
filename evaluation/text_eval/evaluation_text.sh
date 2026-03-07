version="finetune_PIXAR-7B_ours_seg-only_text0.1_full_0.05_tampered_descriptions"
gpu=0
mkdir -p ./logs/${version}
export CUDA_VISIBLE_DEVICES=${gpu}
python compute_css.py \
    --json_path /home/jiacheng/Omni_detection/PIXAR/evaluation/logs/${version}/generated_texts.json \
    --output_path ./logs/${version}/css_scores.json