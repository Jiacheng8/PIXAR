<div align="center">
<!-- <img src="./images/SIDA.png" alt="Image Alt Text" width="150" height="150"> -->
<h3> From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering </h3>
</div>


Xinyi Shang\*, Yi Tang\*, Jiacheng Cui\*, Ahmed Elhagry, Salwa K. Al Khatib
Sondos Mahmoud Bsharat, Jiacheng Liu, Xiaohan Zhao, Jing-Hao Xue, Hao Li, Salman Khan, Zhiqiang Shen<sup>†</sup>

---

## Overview

**PIXAR** implements **SIDA** (Segment and Identify Detection Architecture), a multi-task Vision-Language Model for image tampering detection. Given an input image, the model simultaneously performs:

1. **Image-level Classification** — Classifies images as *Real* (0), *Full Synthetic* (1), or *Tampered* (2)
2. **Pixel-level Segmentation** — Localizes tampered regions with binary masks (for tampered images)
3. **Object-level Recognition** — Identifies which object categories (81 COCO classes) were tampered
4. **Text Description** — Generates natural language descriptions of the tampering

The model is built on top of LLaVA and integrates SAM (Segment Anything Model) ViT-H as the vision encoder, CLIP for visual-language alignment, and LLaMA-2 as the language backbone. It uses LoRA for parameter-efficient fine-tuning and DeepSpeed ZeRO-2 for distributed training.

## Project Structure

```
PIXAR/
├── model/
│   ├── SIDA.py                  # Core model: SIDAForCausalLM
│   ├── SIDA_description.py      # Model variant with text description support
│   ├── llava/                   # LLaVA backbone
│   └── segment_anything/        # SAM encoder
├── finetune/
│   ├── finetune_SIDA-7B_*.sh    # Training scripts (various configs)
│   └── logs/
├── evaluation/
│   ├── evaluation1.sh           # Evaluation scripts
│   ├── evaluation2.sh
│   └── evaluation3.sh
├── utils/
│   ├── SID_Set.py               # Dataset class (CustomDataset)
│   ├── utils.py                 # AverageMeter, IoU computation
│   ├── batch_sampler.py         # Custom distributed batch sampler
│   ├── conversation.py          # Conversation template handling
│   └── data_processing.py       # Preprocessing utilities
├── utils_preprocess/            # Dataset construction scripts
├── train_SIDA.py                # Main training script
├── test.py                      # Main evaluation script
├── chat.py                      # Interactive inference
├── visualization.py             # Per-instance visualization (TP/FP/FN maps)
├── merge_lora_weights_and_save_hf_model.py  # LoRA weight merging
├── merge.sh                     # Merge script
├── filter.py                    # Post-processing and filtering
├── fix/                         # Environment fix scripts
├── requirements.txt
└── README.md
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 1.13.1 (CUDA 11.7)
- Transformers 4.31.0
- DeepSpeed 0.14.0
- PEFT 0.4.0 (LoRA)
- SAM (Segment Anything)
- OpenAI CLIP

### 2. Fix Environment

After installation, run the fix scripts to resolve any remaining issues:

```bash
bash fix/fix.sh
```

If `fix.sh` does not work properly, run the fallback:

```bash
bash fix/fix_again.sh
```

### 3. Pretrained Weights

You will need the following pretrained weights:

| Component | Description |
|-----------|-------------|
| SIDA-7B base model | HuggingFace-format base model directory |
| SAM ViT-H | `sam_vit_h_4b8939.pth` |
| CLIP | `openai/clip-vit-large-patch14` (auto-downloaded) |

## Dataset Format

The dataset should be organized as follows:

```
dataset_dir/
├── train/
│   ├── real/               # Authentic images
│   ├── full_synthetic/     # AI-generated images
│   ├── tampered/           # Tampered images
│   ├── masks/              # Hard binary masks (for tampered images)
│   ├── soft_masks/         # Soft/gradient masks (for tampered images)
│   └── metadata/           # JSON metadata per tampered image
└── validation/
    └── (same structure)
```

Each metadata JSON file corresponds to a tampered image and contains:

```json
{
  "cls": ["person", "car"],
  "text": "A person standing next to a car has been digitally inserted into the scene."
}
```

- `cls`: List of COCO object categories that were tampered
- `text`: Natural language description of the tampering

## Training

Training uses DeepSpeed with LoRA fine-tuning. Multiple configurations are provided for different dataset scales and ablation studies.

### Run Training

```bash
deepspeed --include localhost:0 --master_port=12345 train_SIDA.py \
  --version <path_to_base_model> \
  --dataset_dir <path_to_dataset> \
  --vision_pretrained <path_to_sam_vit_h.pth> \
  --val_dataset <path_to_validation_set> \
  --batch_size 2 \
  --epochs 10 \
  --steps_per_epoch 1000 \
  --lr 1e-4 \
  --dice_loss_weight 1.0 \
  --precision bf16 \
  --exp_name "my_experiment" \
  --log_base_dir ./runs
```

Or use the provided shell scripts:

```bash
bash finetune/finetune_SIDA-7B_ours-0.05_full-dataset.sh

bash finetune/finetune_SIDA-7B_ours-0.05_mask-only.sh
```

### Training Hyperparameters

| Parameter | Default |
|-----------|---------|
| Batch size | 2 per GPU |
| Learning rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Precision | bf16 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | q_proj, v_proj |
| Dice loss weight | 1.0 |

Training logs and checkpoints are saved to the `--log_base_dir` directory. Monitor progress with TensorBoard:

```bash
tensorboard --logdir ./runs
```

## Model Merging

After training, merge LoRA adapter weights into the base model for standalone deployment:

```bash
python merge_lora_weights_and_save_hf_model.py \
  --version <path_to_base_model> \
  --weight <path_to_lora_checkpoint/pytorch_model.bin> \
  --save_path <path_to_save_merged_model>
```

Or use the provided script:

```bash
bash merge.sh
```

## Evaluation

### Run Evaluation

```bash
deepspeed --include localhost:0 --master_port=24996 test.py \
  --version <path_to_model_checkpoint> \
  --dataset_dir <path_to_test_dataset> \
  --vision_pretrained <path_to_sam_vit_h.pth> \
  --test_dataset validation \
  --precision bf16 \
  --exp_name "my_evaluation" \
  --test_only \
  > ./logs/my_evaluation.log 2>&1
```

Or use the provided scripts:

```bash
bash evaluation/evaluation1.sh
```

### Evaluation Metrics

The evaluation reports comprehensive metrics across all tasks:

**Classification:**
- Overall accuracy, per-class precision / recall / F1
- Confusion matrix

**Segmentation:**
- gIoU (global Intersection over Union)
- cIoU (class-level IoU)
- Pixel accuracy, precision, recall, F1
- Pixel-level ROC-AUC

**Object Recognition:**
- Micro / Macro averaged precision, recall, F1
- Subset accuracy (exact match)
- Top-1 and Top-5 accuracy


## Interactive Inference

Use `chat.py` for interactive command-line inference on individual images:

```bash
python chat.py \
  --version <path_to_model_checkpoint> \
  --precision bf16 \
  --vision-tower openai/clip-vit-large-patch14
```

## Model Architecture

```
Input Image
    │
    ├──► SAM ViT-H Encoder ──► Image Embeddings ──► Mask Decoder ──► Segmentation Mask
    │                                  ▲
    ├──► CLIP Encoder ──► Visual Tokens │
    │         │                        │
    │         ▼                        │
    │    LLaMA-2 LLM                   │
    │    (with LoRA)                   │
    │         │                        │
    │         ├──► [CLS] token ──► cls_head ──► Classification (3-way)
    │         │         │                          │
    │         │         └── Attention Layer ────────┘ (integrates cls features into segmentation)
    │         ├──► [OBJ] token ──► obj_head ──► Object Recognition (81 classes)
    │         ├──► [SEG] token ──► Projection ──► Mask Decoder
    │         └──► [END] token ──► End of sequence
```

Special tokens:
- `[CLS]`: Triggers image-level classification
- `[SEG]`: Triggers mask segmentation
- `[OBJ]`: Triggers object category recognition
- `[END]`: Marks end of the structured output

## Citation

If you find this work useful, please cite:

```bibtex

```
