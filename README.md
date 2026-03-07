<div align="center">

<h3><img src="./assets/logo.svg" alt="PIXAR Logo" height="50" style="vertical-align:middle;">&ensp;<em>From Masks to Pixels and Meaning:<br>A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering</em></h3>

<p>
  <a href=""><img src="https://img.shields.io/badge/Paper-PDF-blue?style=flat-square&logo=adobeacrobatreader" alt="Paper"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Benchmark-420K%2B%20pairs-orange?style=flat-square" alt="Benchmark">
  &nbsp;
  <img src="https://img.shields.io/badge/Tasks-Classification%20%7C%20Localization%20%7C%20Description-green?style=flat-square" alt="Tasks">
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="License">
</p>

<p>
<strong>Xinyi Shang</strong>*&ensp;
<strong>Yi Tang</strong>*&ensp;
<strong>Jiacheng Cui</strong>*&ensp;
<strong>Ahmed Elhagry</strong>&ensp;
<strong>Salwa K. Al Khatib</strong><br>
<strong>Sondos Mahmoud Bsharat</strong>&ensp;
<strong>Jiacheng Liu</strong>&ensp;
<strong>Xiaohan Zhao</strong>&ensp;
<strong>Jing-Hao Xue</strong>&ensp;
<strong>Hao Li</strong>&ensp;
<strong>Salman Khan</strong>&ensp;
<strong>Zhiqiang Shen</strong>†
</p>

<p><sub>* Equal contribution &nbsp;|&nbsp; † Corresponding author &nbsp;|&nbsp; Preprint</sub></p>

<br>

<img src="./assets/motivation.png" alt="PIXAR Motivation" width="850">
<p><sub><em>Mask-based labels misalign with true edit signals (top). Our pixel-difference labels are precisely aligned with the generative footprint (bottom).</em></sub></p>

<br>

> **TL;DR**: We expose a fundamental flaw in mask-based tampering benchmarks, and introduce **PIXAR**: a 420K+ benchmark with pixel-faithful labels, 8 manipulation types, and a VLM detector that simultaneously localizes, classifies, and describes tampered regions, achieving **2.7× IoU improvement** over prior SOTA.

</div>

---

## 🔥 News
- **[2026-03]** 📦 Code and model weights and PIXAR benchmark (420K+ pairs) are released.
- **[2026-02]** 🎉 Paper accepted to **CVPR 2026 Findings** (withdrawn for resubmission).
---

## Overview

Existing tampering benchmarks rely on coarse object masks as ground truth, which severely misalign with the true edit signal: many pixels inside a mask are untouched, while subtle yet consequential edits outside the mask are treated as natural. We reformulate VLM image tampering from coarse region labels to a **pixel-grounded, meaning- and language-aware** task.

**PIXAR** is a large-scale benchmark and training framework with three core contributions:

1. **A new taxonomy** spanning 8 edit primitives (replace / remove / splice / inpaint / attribute / colorization, etc.) linked to the semantic class of the tampered object.
2. **A new benchmark** of over 420K training image pairs and a carefully balanced 40K test set, each with per-pixel tamper maps, semantic category labels, and natural language descriptions.
3. **A new training framework and metrics** that quantify pixel-level correctness with localization, assess confidence on true edit intensity, and measure tamper meaning understanding via semantics-aware classification and natural language descriptions.

---

## Motivation

<div align="center">
<img src="./assets/motivation.png" alt="Motivation" width="800">
<p><em><b>Pitfalls of current benchmarks.</b> (a) Unrealistic samples. (b)–(d) Widely adopted mask-based labels contain large misaligned regions. Our pixel-difference label is precisely aligned with the true generative pixels.</em></p>
</div>

Current SOTA benchmarks (e.g., SID-Set) annotate "where the edit is" using coarse object masks. In practice, the edit signal is neither spatially nor metrically uniform — many pixels inside the mask remain unchanged while visually consequential adjustments extend outside. This causes mask-only evaluation to conflate unedited pixels with tamper evidence, distorting both detector training and measurement.

We address this by computing a **per-pixel difference map** D(**x**,y) = |I_orig(**x**,y) − I_gen(**x**,y)| and thresholding it at a tunable τ to obtain binary supervision masks **M**_τ. Small τ captures micro-edits; larger τ retains only high-confidence semantic changes.

<div align="center">
<img src="./assets/vis_tau_1.png" alt="Tau visualization" width="750">
<p><em><b>Pixel-level labels under different τ.</b> Smaller τ emphasizes pixel-intensity changes; larger τ emphasizes semantic changes.</em></p>
</div>

---

## PIXAR Benchmark

### Scale and Structure

| Split | Size | Labels |
|---|---|---|
| Training | 420K+ image pairs | Pixel-level M_τ, semantic class, text description |
| Test | 40K image pairs (balanced) | Pixel-level M_τ, semantic class, text description |

Each entry is a **quadruple**: (i) real source image, (ii) tampered counterpart, (iii) binary pixel-level label map **M**_τ at τ = 0.05 by default, (iv) raw per-pixel difference map for deriving alternative labels at other τ values. Accompanying metadata records tampering type, fidelity score, tampered size, and tampering number.

### 8 Tampering Types

The benchmark covers 8 manipulation strategies instantiated via both open-source and closed-source generative models (Flux.2, Gemini 2.5, Gemini 3, GPT-image-1.5, Qwen-Image, Seedream 4.5):

- Replacement (intra-class / inter-class)
- Object removal
- Object addition
- Material change
- Color change
- Attribute modification
- Splice / Inpaint

### Four-Stage Generation Pipeline

1. **Image Generation** — 8 tampering types applied via VLMs with structured instructions.
2. **Tampering Effectiveness Checks** — Global rectification (RANSAC homography alignment) + edit magnitude and semantic correctness filtering.
3. **Image Fidelity Assessment** — Automated scoring by Qwen3 (threshold ≥ 9/10) followed by human expert review (10 annotators, realism threshold ≥ 4/5).
4. **Label Construction** — Per-pixel difference map thresholded at τ, with pixel-semantic consistency and spatial concentration checks to discard unreliable labels.

### Comparison with Existing Benchmarks

| Dataset | Year | Multi-Object | Fidelity Check | Ground Truth |
|---|---|---|---|---|
| ArtiFact | 2023 | ✗ | ✗ | — |
| SIDBench | 2024 | ✗ | ✗ | Mask |
| M3Dsynth | 2024 | ✗ | ✗ | Mask |
| SemiTruths | 2024 | ✗ | ✗ | Mask |
| SID-Set | 2025 | ✗ | ✗ | Mask |
| **PIXAR (Ours)** | **2026** | **✓** | **✓** | **Pixel & Semantics** |

---

## Method

<div align="center">
<img src="./assets/method.png" alt="PIXAR Method" width="800">
<p><em><b>PIXAR training framework.</b> The model jointly trains a CLS head for image-level detection, an OBJ head for semantic classification, a SEG head for pixel-level localization, and a text decoder for tamper description generation.</em></p>
</div>

The PIXAR detector fθ takes an image and a user prompt, and simultaneously produces:
- **(i)** a per-pixel tamper logit map **S** ∈ ℝ^(H×W) → predicted mask **M̂**
- **(ii)** a multi-label semantic logit vector **z** ∈ ℝ^|C| → object category predictions **ŷ**
- **(iii)** a natural language description of the specific tampering artifact

### Architecture

Built on LLaVA + LLaMA-2 with LoRA fine-tuning, integrated with SAM ViT-H for pixel-level decoding and CLIP ViT-L/14 for visual-language alignment.

Three special tokens anchor the multi-task heads in the token sequence:

| Token | Role |
|---|---|
| `[CLS]` | Hidden state → 3-way classification (real / tampered) via `FC_cls` |
| `[OBJ]` | Hidden state → multi-label object recognition (81 COCO classes) via `FC_obj` |
| `[SEG]` | Hidden state fused with generated text → SAM prompt for pixel localization via `FC_seg` |

### Training Objective

```
L_total = λ_sem · L_sem  +  λ_bce · L_bce  +  λ_dice · L_dice  +  λ_text · L_text  +  λ_cls · L_cls
```

Default weights: λ_sem = 0.1, λ_dice = 1.0, λ_text = 2.0.

### Segmentation Prompt Modes

The `[SEG]` token embedding can be fused with the generated text description in three ablation modes:

| Mode | Fused prompt |
|---|---|
| `seg_only` | seg_emb only |
| `text_only` | text_emb only |
| `fuse` | gate · seg_emb + (1 − gate) · text_emb, gate = σ(MLP([seg_emb, text_emb])) |

---

## Results

<div align="center">
<img src="./assets/exp.png" alt="Experimental Results" width="750">
</div>

> PIXAR-7B achieves a near-doubling of localization accuracy over SIDA-7B (IoU 6.9 → 18.5). PIXAR-13B further sets new SOTA across all metrics.

## Project Structure

```
PIXAR/
├── assets/                          # Paper figures and PDF
│   ├── method.png
│   ├── motivation.png
│   ├── vis_tau_1.png
│   └── exp.png
├── model/
│   ├── PIXAR.py                     # Core model: PIXARForCausalLM
│   ├── llava/                       # LLaVA backbone
│   └── segment_anything/            # SAM encoder
├── finetune/
│   ├── finetune_PIXAR-7B_*.sh       # Training scripts 
├── evaluation/
│   ├── evaluation_PIXAR-7B_*.sh     # Evaluation scripts
│   ├── text_eval/
│   │   └── compute_css.py           # Cosine Semantic Similarity scoring
│   └── README.md                    # Evaluation guide
├── utils/
│   ├── PIXAR_Set.py                 # Dataset class (CustomDataset)
│   ├── utils.py                     # AverageMeter, IoU computation
│   └── batch_sampler.py             # Custom distributed batch sampler
├── utils_preprocess/                # Dataset construction scripts
├── download-data/                   # Data download scripts
│   ├── download.sh
│   ├── files.txt
│   └── README.md
├── train_PIXAR.py                   # Main training script
├── test_parallel.py                 # Multi-GPU parallel evaluation
├── chat.py                          # Interactive inference
├── merge_lora_weights_and_save_hf_model.py
├── merge.sh
├── visualization.py
└── filter.py
```

---

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.10

### 2. Fix Environment

```bash
bash fix/fix.sh
```

if it is not working, please run fix_again.sh

### 3. Pretrained Weights

| Component | Description |
|---|---|
| PIXAR-7B base model | HuggingFace-format LLaVA-LLaMA-2 base |
| SAM ViT-H | `sam_vit_h_4b8939.pth` |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` (auto-downloaded) |

### 4. Download Data

See [`download-data/README.md`](./download-data/README.md) for instructions on downloading the PIXAR dataset via rclone.

---

## Dataset Format

```
dataset_dir/
├── train/
│   ├── real/               # Authentic images
│   ├── full_synthetic/     # AI-generated images (fully synthetic)
│   ├── tampered/           # Tampered images
│   ├── masks/              # Hard binary masks
│   ├── soft_masks/         # Pixel-difference maps M_τ (τ = 0.05)
│   └── metadata/           # JSON metadata per tampered image
└── validation/
    └── (same structure)
```

Each metadata JSON:

```json
{
  "cls": ["person", "car"],
  "text": "A person standing next to a car has been digitally inserted into the scene."
}
```

---

## Training

### Quick Start

```bash
deepspeed --include localhost:0 --master_port=12345 train_PIXAR.py \
  --version <path_to_base_model> \
  --dataset_dir <path_to_dataset> \
  --vision_pretrained <path_to_sam_vit_h.pth> \
  --val_dataset <path_to_validation_set> \
  --batch_size 2 \
  --epochs 10 \
  --steps_per_epoch 1000 \
  --lr 1e-4 \
  --dice_loss_weight 1.0 \
  --seg_prompt_mode fuse \
  --precision bf16 \
  --exp_name "pixar_experiment" \
  --log_base_dir ./runs
```

Or use the provided scripts:

```bash
bash finetune/finetune_PIXAR-7B_ours_seg-only.sh
```

### Key Hyperparameters

| Parameter | Default |
|---|---|
| Batch size | 2 per GPU |
| Learning rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Precision | bf16 |
| LoRA rank | 8, alpha 16, dropout 0.05 |
| LoRA targets | q_proj, v_proj |
| Dice loss weight λ_dice | 1.0 |
| Semantic loss weight λ_sem | 0.1 |
| Text loss weight λ_text | 2.0 |
| Pixel threshold τ | 0.05 |

---

## Model Merging

Merge LoRA adapter weights into the base model for standalone deployment:

```bash
python merge_lora_weights_and_save_hf_model.py \
  --version <path_to_base_model> \
  --weight <path_to_lora_checkpoint/pytorch_model.bin> \
  --save_path <path_to_save_merged_model>
```

Or:

```bash
bash merge.sh
```

---

## Evaluation

See [`evaluation/README.md`](./evaluation/README.md) for the full evaluation guide. A brief summary:

### Multi-GPU Parallel Evaluation

```bash
python test_parallel.py \
  --version <path_to_merged_model> \
  --dataset_dir <path_to_test_dataset> \
  --vision_pretrained <path_to_sam_vit_h.pth> \
  --gpus 0,1,2,3 \
  --output_dir ./evaluation/logs/my_experiment \
  --seg_prompt_mode fuse \
  --precision bf16 \
  --save_generated_text
```

### Text Quality (CSS) Evaluation

```bash
cd evaluation/text_eval
python compute_css.py \
  --json_path ../logs/my_experiment/generated_texts.json \
  --output_path ./logs/my_experiment/css_scores.json
```

---

## Interactive Inference

```bash
python chat.py \
  --version <path_to_merged_model> \
  --precision bf16 \
  --seg_prompt_mode fuse \
  --vision-tower openai/clip-vit-large-patch14
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{shang2026pixar,
  title     = {From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering},
  author    = {Shang, Xinyi and Tang, Yi and Cui, Jiacheng and Elhagry, Ahmed and Al Khatib, Salwa K. and Bsharat, Sondos Mahmoud and Liu, Jiacheng and Zhao, Xiaohan and Xue, Jing-Hao and Li, Hao and Khan, Salman and Shen, Zhiqiang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
