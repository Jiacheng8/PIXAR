"""
SIDA Test Script (Merged Model)

This script evaluates a merged (base + finetune) SIDA model.
No LoRA wrapping or checkpoint loading needed — the model is loaded directly.
Uses the evaluate() two-stage approach:
  1. Inject [CLS] [OBJ] [SEG] as assistant prefix
  2. Generate text description
  3. Forward pass to extract hidden states for cls/obj/seg heads

Usage:
    python test.py --version /path/to/merged_model --precision fp16
"""

import argparse
import json
import os
import sys
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.SID_Set import CustomDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                         AverageMeter, Summary, intersectionAndUnionGPU)

import warnings
warnings.filterwarnings("ignore")


def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Testing (Merged Model)")
    parser.add_argument("--version", required=True, type=str,
                        help="Path to merged model (base + finetune weights)")
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument("--output_dir", default="./test_output", type=str)
    parser.add_argument("--workers", default=4, type=int)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])

    # OBJ head
    parser.add_argument("--num_obj_classes", type=int, default=81)
    parser.add_argument("--obj_threshold", type=float, default=0.5)

    # Text generation
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_generated_text", action="store_true", default=False)
    parser.add_argument("--text_output_file", type=str, default="generated_texts.json")

    # Segmentation prompt mode
    parser.add_argument("--seg_prompt_mode", type=str, default="fuse",
                        choices=["seg_only", "text_only", "fuse"])

    # Sampling
    parser.add_argument("--sample_ratio", type=float, default=None,
                        help="Fraction of test set to evaluate (e.g. 0.1 for 10%%)")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Tokenizer -----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Token indices (already saved in tokenizer)
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # ----- Model (merged, load directly) -----
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_prompt_mode": args.seg_prompt_mode,
    }

    model = SIDAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    model.resize_token_embeddings(len(tokenizer))

    # Move to GPU and set eval
    model = model.cuda()
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    print(f"Model loaded from: {args.version}")
    print(f"Precision: {args.precision}")
    print(f"Seg prompt mode: {args.seg_prompt_mode}")

    # ----- Dataset -----
    test_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split=args.split,
        precision=args.precision,
        image_size=args.image_size,
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # ----- Default prompt -----
    default_prompt = (
        "Can you identify whether this image is real, fully synthetic, or tampered? "
        "If it is tampered, please (1) classify which object was modified and "
        "(2) output a mask for the modified regions."
    )

    # ----- Metrics accumulators -----
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cpu')
    correct = 0
    total = 0

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # OBJ accumulators
    obj_tp_total = 0.0; obj_fp_total = 0.0; obj_fn_total = 0.0
    obj_exact_match_total = 0; obj_rows_total = 0
    obj_tp_per_class = None; obj_fp_per_class = None; obj_fn_per_class = None
    obj_hit1_total = 0; obj_hit5_total = 0; obj_hit_den_total = 0

    # Pixel-level TP/FP/FN
    pix_TP = 0; pix_FP = 0; pix_FN = 0

    # AUC histogram (constant memory)
    BINS = 512
    pos_hist = torch.zeros(BINS, device='cuda', dtype=torch.float64)
    neg_hist = torch.zeros(BINS, device='cuda', dtype=torch.float64)

    # Text generation storage
    generated_texts = []

    # Sampling
    indices = list(range(len(test_dataset)))
    if args.sample_ratio is not None:
        num_samples = max(1, int(len(indices) * args.sample_ratio))
        indices = sorted(random.sample(indices, num_samples))
        print(f"Sampling {num_samples}/{len(test_dataset)} examples...")

    # ----- Evaluation Loop -----
    for sample_idx in tqdm.tqdm(indices, desc="Testing"):
        # Get dataset item
        item = test_dataset[sample_idx]
        (image_path, image, image_clip, conversations, mask, soft_mask,
         labels, cls_labels, resize, _, _, _, has_text, obj_label_vec) = item

        # Build conversation with [CLS] [OBJ] [SEG] prefix
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + default_prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "[CLS] [OBJ] [SEG] ")
        full_prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # Prepare images
        image_clip = image_clip.unsqueeze(0).cuda()
        image = image.unsqueeze(0).cuda()

        if args.precision == "fp16":
            image_clip = image_clip.half()
            image = image.half()
        elif args.precision == "bf16":
            image_clip = image_clip.bfloat16()
            image = image.bfloat16()

        resize_list = [resize]
        original_size_list = [labels.shape[-2:]]  # H, W of the original label

        # Run evaluate()
        with torch.no_grad():
            output_ids, pred_masks, obj_preds, cls_info = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                cls_label=cls_labels,
            )

        # Decode generated text
        input_token_len = input_ids.shape[1]
        new_tokens = output_ids[0][input_token_len:]
        new_tokens = new_tokens[new_tokens != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        text_output = text_output.replace("\n", " ").replace("  ", " ").strip()

        # Ground-truth text description
        if cls_labels == 0:
            gt_text_description = "This image is real."
        elif cls_labels == 1:
            gt_text_description = "This image is fully synthetic."
        else:
            conv_str = conversations[0]
            seg_marker = "[SEG] "
            seg_pos = conv_str.find(seg_marker)
            if seg_pos >= 0:
                gt_text_description = conv_str[seg_pos + len(seg_marker):].split("</s>")[0].strip()
            else:
                gt_text_description = ""

        # Store generated text
        generated_texts.append({
            "image_path": image_path,
            "generated_text": text_output,
            "gt_text_description": gt_text_description,
            "ground_truth_label": int(cls_labels),
            "predicted_class": cls_info["predicted_class"],
            "predicted_label": cls_info["label"],
        })

        if sample_idx < 5:
            print(f"\n[Sample {sample_idx}] {image_path}")
            print(f"  GT: {cls_labels}, Pred: {cls_info['label']}")
            print(f"  Text: {text_output[:120]}...")

        # ------ Classification metrics ------
        predicted_class = cls_info["predicted_class"]
        preds = torch.tensor([predicted_class], device='cuda')
        gt_cls = torch.tensor([cls_labels], device='cuda')

        correct += (preds == gt_cls).sum().item()
        total += 1

        # Confusion matrix (CPU)
        confusion_matrix[int(cls_labels), predicted_class] += 1

        # ------ Segmentation (tampered only) ------
        if cls_labels == 2:
            # soft_mask: [1, H, W], pred_masks: list of [1, H, W]
            gt_mask = soft_mask.int().cuda()
            pred_mask_bin = (pred_masks[0] > 0).int().cuda()

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(gt_mask, pred_mask_bin):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / gt_mask.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=gt_mask.shape[0])

            # Pixel-level metrics & AUC histogram
            with torch.no_grad():
                pm = pred_masks[0].float().cuda()
                if (pm.min() < 0) or (pm.max() > 1.0):
                    pred_scores = torch.sigmoid(pm)
                else:
                    pred_scores = pm.clamp(0, 1)

            pred_bin = (pred_scores >= 0.5).to(torch.int32)

            for mask_i, score_i, bin_i in zip(gt_mask, pred_scores, pred_bin):
                m = mask_i.flatten().to(torch.uint8)
                p = bin_i.flatten().to(torch.uint8)
                s = score_i.flatten().to(torch.float32)

                TP = (p.eq(1) & m.eq(1)).sum().item()
                FP = (p.eq(1) & m.eq(0)).sum().item()
                FN = (p.eq(0) & m.eq(1)).sum().item()
                pix_TP += TP; pix_FP += FP; pix_FN += FN

                # Histogram counting (constant memory)
                s_clamped = s.clamp_(0, 1)
                bins = torch.clamp((s_clamped * (BINS - 1)).long(), 0, BINS - 1)
                m_bool = (m > 0)
                if m_bool.any():
                    pos_hist.index_add_(0, bins[m_bool],
                                       torch.ones_like(bins[m_bool], dtype=torch.float64))
                if (~m_bool).any():
                    neg_hist.index_add_(0, bins[~m_bool],
                                       torch.ones_like(bins[~m_bool], dtype=torch.float64))

        # ------ OBJ Multi-label (tampered only) ------
        if cls_labels == 2:
            gt = obj_label_vec.unsqueeze(0).cuda()  # [1, K]
            probs_obj = obj_preds.unsqueeze(0) if obj_preds.dim() == 1 else obj_preds  # [1, K]
            pred = (probs_obj >= args.obj_threshold).to(gt.dtype)

            gt_bool = (gt > 0).to(torch.bool)
            valid_rows = gt_bool.any(dim=1)
            n_valid = int(valid_rows.sum().item())
            if n_valid > 0:
                K = gt.shape[1]; k5 = min(5, K)
                topk_idx = probs_obj.topk(k5, dim=1).indices
                top1_idx = topk_idx[:, :1]
                hit1 = (gt_bool.gather(1, top1_idx)).any(dim=1)
                topk_mask = torch.zeros_like(gt_bool)
                topk_mask.scatter_(1, topk_idx, True)
                hit5 = (topk_mask & gt_bool).any(dim=1)
                obj_hit1_total += int(hit1[valid_rows].sum().item())
                obj_hit5_total += int(hit5[valid_rows].sum().item())
                obj_hit_den_total += n_valid

            if obj_tp_per_class is None:
                K = gt.shape[1]; device = gt.device
                obj_tp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                obj_fp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                obj_fn_per_class = torch.zeros(K, device=device, dtype=torch.float64)

            tp = (pred * gt).sum().double()
            fp = (pred * (1 - gt)).sum().double()
            fn = ((1 - pred) * gt).sum().double()
            obj_tp_total += tp.item(); obj_fp_total += fp.item(); obj_fn_total += fn.item()

            exact_match = (pred == gt).all(dim=1).sum().item()
            obj_exact_match_total += exact_match
            obj_rows_total += gt.shape[0]

            obj_tp_per_class += (pred * gt).sum(dim=0).double()
            obj_fp_per_class += (pred * (1 - gt)).sum(dim=0).double()
            obj_fn_per_class += ((1 - pred) * gt).sum(dim=0).double()

    # -------- Compute final metrics --------
    # Pixel P/R/F1
    pixel_precision = pix_TP / (pix_TP + pix_FP + 1e-12) if (pix_TP + pix_FP) > 0 else 0.0
    pixel_recall    = pix_TP / (pix_TP + pix_FN + 1e-12) if (pix_TP + pix_FN) > 0 else 0.0
    pixel_f1        = (2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-12)
                       if (pixel_precision + pixel_recall) > 0 else 0.0)

    # AUC from histogram
    if (pos_hist.sum() + neg_hist.sum()) > 0:
        pos_cum = torch.cumsum(pos_hist.flip(0), dim=0)
        neg_cum = torch.cumsum(neg_hist.flip(0), dim=0)
        tp_h = pos_cum; fp_h = neg_cum
        P = pos_cum[-1]; N = neg_cum[-1]
        fn_h = P - tp_h; tn_h = N - fp_h

        precision_h = tp_h / (tp_h + fp_h + 1e-12)
        recall_h    = tp_h / (tp_h + fn_h + 1e-12)

        dr = recall_h[:-1] - recall_h[1:]
        pixel_pr_auc = torch.sum(precision_h[1:] * dr).item()

        fpr = fp_h / (fp_h + tn_h + 1e-12)
        tpr = recall_h
        df = fpr[1:] - fpr[:-1]
        pixel_roc_auc = torch.sum((tpr[1:] + tpr[:-1]) * 0.5 * df).item()
    else:
        pixel_pr_auc = 0.0
        pixel_roc_auc = 0.0

    # OBJ metrics
    obj_micro_prec = obj_tp_total / (obj_tp_total + obj_fp_total + 1e-12) if (obj_tp_total + obj_fp_total) > 0 else 0.0
    obj_micro_rec  = obj_tp_total / (obj_tp_total + obj_fn_total + 1e-12) if (obj_tp_total + obj_fn_total) > 0 else 0.0
    obj_micro_f1   = (2 * obj_micro_prec * obj_micro_rec / (obj_micro_prec + obj_micro_rec + 1e-12)
                      if (obj_micro_prec + obj_micro_rec) > 0 else 0.0)
    obj_subset_acc = (obj_exact_match_total / obj_rows_total) if obj_rows_total > 0 else 0.0
    obj_top1 = (obj_hit1_total / obj_hit_den_total * 100.0) if obj_hit_den_total > 0 else 0.0
    obj_top5 = (obj_hit5_total / obj_hit_den_total * 100.0) if obj_hit_den_total > 0 else 0.0

    if obj_tp_per_class is not None:
        prec_c = obj_tp_per_class / (obj_tp_per_class + obj_fp_per_class + 1e-12)
        rec_c  = obj_tp_per_class / (obj_tp_per_class + obj_fn_per_class + 1e-12)
        f1_c   = (2 * prec_c * rec_c / (prec_c + rec_c + 1e-12))
        obj_macro_prec = float(prec_c.mean().item())
        obj_macro_rec  = float(rec_c.mean().item())
        obj_macro_f1   = float(f1_c.mean().item())
    else:
        obj_macro_prec = obj_macro_rec = obj_macro_f1 = 0.0

    # IoU
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    # Classification
    accuracy = correct / total * 100.0 if total > 0 else 0.0
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    cm = confusion_matrix
    for i in range(num_classes):
        tp_i = cm[i, i]
        fp_i = cm[:, i].sum() - tp_i
        fn_i = cm[i, :].sum() - tp_i
        total_class_samples = cm[i, :].sum()
        class_accuracy = float(tp_i / total_class_samples) if total_class_samples > 0 else 0.0
        prec_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        rec_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i = float(2 * (prec_i * rec_i) / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy, 'precision': prec_i, 'recall': rec_i, 'f1': f1_i
        }

    iou = ciou
    f1_score = (2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10)
                if (iou + accuracy / 100) > 0 else 0.0)

    # -------- Print results --------
    test_type = "Full" if args.sample_ratio is None else f"Sampled ({args.sample_ratio*100:.0f}%)"
    print(f"\n{'='*70}")
    print(f"{test_type} Test Results ({total} samples)")
    print(f"{'='*70}")

    print(f"\nClassification Accuracy: {accuracy:.4f}%")
    print("\nPer-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"{'':20}", end="")
    for name in class_names:
        print(f"{name:>15}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:15.0f}", end="")
        print()

    print(f"\nSegmentation Metrics (tampered only):")
    print(f"  gIoU: {giou:.4f}")
    print(f"  cIoU: {ciou:.4f}")
    print(f"  Pixel Precision: {pixel_precision:.4f}")
    print(f"  Pixel Recall:    {pixel_recall:.4f}")
    print(f"  Pixel F1:        {pixel_f1:.4f}")
    print(f"  Pixel ROC-AUC:   {pixel_roc_auc:.4f}")

    print(f"\n[OBJ] Multi-Label Metrics (tampered only):")
    print(f"  Threshold: {args.obj_threshold:.2f}")
    print(f"  Micro  - P: {obj_micro_prec:.4f}, R: {obj_micro_rec:.4f}, F1: {obj_micro_f1:.4f}")
    print(f"  Macro  - P: {obj_macro_prec:.4f}, R: {obj_macro_rec:.4f}, F1: {obj_macro_f1:.4f}")
    print(f"  Subset Acc: {obj_subset_acc:.4f}")
    print(f"  Top-1 Acc:  {obj_top1:.4f}%")
    print(f"  Top-5 Acc:  {obj_top5:.4f}%")

    print(f"\nCombined F1: {f1_score:.4f}")

    # Save generated texts
    if args.save_generated_text and generated_texts:
        output_path = os.path.join(args.output_dir, args.text_output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generated_texts, f, indent=2, ensure_ascii=False)
        print(f"\nGenerated texts saved to: {output_path}")

    # Save metrics summary
    metrics_summary = {
        "accuracy": accuracy,
        "giou": float(giou),
        "ciou": float(ciou),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_f1": pixel_f1,
        "pixel_roc_auc": pixel_roc_auc,
        "obj_micro_f1": obj_micro_f1,
        "obj_macro_f1": obj_macro_f1,
        "obj_top1": obj_top1,
        "obj_top5": obj_top5,
        "per_class_metrics": per_class_metrics,
        "total_samples": total,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return accuracy, giou, ciou, per_class_metrics


if __name__ == "__main__":
    main(sys.argv[1:])
