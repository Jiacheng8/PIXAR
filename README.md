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

### Forward Pass Overview

The forward pass (`model_forward`) proceeds in five stages:

#### Stage 1 — Dual Image Encoding

The input image is encoded by **two separate encoders** in parallel:

- **SAM ViT-H** (`get_visual_embs`): encodes the raw high-resolution image into `image_embeddings` used later by the SAM mask decoder. This branch is always frozen (no gradient).
- **CLIP ViT-L/14**: encodes image visual tokens that are injected into the LLM token sequence. Used by the LLaVA backbone for vision-language alignment.

#### Stage 2 — LLM Forward Pass (LLaVA / LLaMA-2 + LoRA)

The CLIP visual tokens are concatenated with the structured text input containing three special tokens: `[CLS]`, `[OBJ]`, and `[SEG]`. The full sequence is passed through LLaMA-2 (fine-tuned with LoRA) to produce:

- `output_hidden_states[-1]`: the last-layer hidden state tensor of shape `[B, T_expanded, H_dim]`, where `T_expanded = T_input + num_image_tokens - 1`.
- `text_loss`: the standard causal language modeling loss over the generated tampering description.

#### Stage 3 — Special Token Extraction

The positions of `[CLS]`, `[OBJ]`, and `[SEG]` in `input_ids` are located. Their corresponding hidden vectors are extracted from the last hidden state with an `image_offset` correction (to account for the extra image tokens inserted by LLaVA):

```
cls_vec = hs[b, cls_pos + image_offset]   # → Classification
obj_vec = hs[b, obj_pos + image_offset]   # → Object recognition
seg_vec = hs[b, seg_pos + image_offset]   # → Segmentation prompt
```

#### Stage 4 — Task Heads (Conditional on Class)

All three heads run on every batch, but loss is only accumulated for the relevant subset:

**Classification head** (all samples):
```
cls_logits = cls_head(cls_vec)   # [B, 3]: real / fully synthetic / tampered
cls_loss   = CrossEntropyLoss(cls_logits, cls_labels)
```

**Object recognition head** (tampered samples only, `cls_label == 2`):
```
obj_logits = obj_head(obj_vec)   # [B, 81 COCO classes]
obj_loss   = BCEWithLogitsLoss(obj_logits[tampered], obj_labels[tampered],
                               pos_weight=dynamic_or_fixed)
```
> The object head is always included in the computation graph (via `obj_logits * 0.0`) to prevent DeepSpeed ZeRO all-reduce deadlocks across ranks.

**Segmentation** (tampered samples only):

For each tampered sample, a **gated fusion** mechanism combines the `[SEG]` token embedding with a text context embedding derived from the tokens generated *after* `[SEG]`:

```
seg_emb  = seg_proj(seg_vec)                          # [out_dim]
text_emb = text_proj(mean(hs[seg_pos+1 : seq_end]))   # [out_dim]

# Three ablation modes:
#   seg_only  → fused = seg_emb
#   text_only → fused = text_emb
#   fuse      → gate  = sigmoid(gate_mlp([seg_emb, text_emb]))
#                fused = gate * seg_emb + (1 - gate) * text_emb
```

The fused prompt embedding is passed to the **SAM prompt encoder** and then the **SAM mask decoder**:

```
sparse_emb, dense_emb = prompt_encoder(text_embeds=fused)
low_res_masks, _      = mask_decoder(image_embeddings, sparse_emb, dense_emb)
pred_mask             = postprocess_masks(low_res_masks, resize, original_size)
```

Mask loss = weighted BCE (sigmoid cross-entropy) + Dice loss against ground-truth (soft or hard) masks.

#### Stage 5 — Total Loss

```
loss = mask_loss_weight  * (bce_loss + dice_loss)
     + cls_loss_weight   * cls_loss
     + obj_loss_weight   * obj_loss
     + text_loss_weight  * text_loss
```

### Architecture Diagram

```
Input Image
    │
    ├──► SAM ViT-H Encoder (frozen) ──────────────────────────────────────────────────────┐
    │         └──► image_embeddings                                                        │
    │                                                                                      ▼
    ├──► CLIP ViT-L/14 ──► visual tokens                                        SAM Mask Decoder
    │         │                   │                                                        │
    │         └─────────────────► LLaMA-2 (LoRA)  ◄── [CLS] [OBJ] [SEG] text tokens      │
    │                                   │                                                  │
    │                                   ├──► hs[[CLS]+offset] ──► cls_head ──► 3-way classification
    │                                   │
    │                                   ├──► hs[[OBJ]+offset] ──► obj_head ──► 81-class multi-label
    │                                   │                           (tampered only)
    │                                   │
    │                                   └──► hs[[SEG]+offset] ──► seg_proj ──► seg_emb ──┐
    │                                        hs[[SEG]+1:end]  ──► text_proj ──► text_emb ┤
    │                                                                  gate_mlp ──► gate  │
    │                                                         fused = gate*seg + (1-gate)*text
    │                                                                    │
    │                                              SAM Prompt Encoder ◄─┘
    │                                                        │
    └────────────────────────────────────────────────────────┴──► Segmentation Mask
                                                                  (tampered only)
```

### Special Tokens

| Token | Role |
|-------|------|
| `[CLS]` | Its hidden state drives 3-way image classification (real / fully synthetic / tampered) |
| `[OBJ]` | Its hidden state drives multi-label object recognition (81 COCO classes); used only for tampered images |
| `[SEG]` | Its hidden state, fused with the generated description, forms the SAM segmentation prompt; used only for tampered images |

## Citation

If you find this work useful, please cite:

```bibtex

```
