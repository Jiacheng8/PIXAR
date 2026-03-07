import argparse
import os
import re
import sys
import numpy as np
from PIL import Image
from functools import partial
import torch
import torch.nn.functional as F
import tqdm
import transformers

from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from utils.PIXAR_Set import collate_fn, CustomDataset
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda

import matplotlib.pyplot as plt


# ------------------------- utils for visualization -------------------------

def load_or_placeholder(path, size=(256, 256), text="real not found"):
    """读取图片；不存在就生成一张灰底占位图并写字。"""
    try:
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB").resize(size)
    except Exception:
        pass
    img = Image.new("RGB", size, (240, 240, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, size[1] // 2 - 8), text, fill=(80, 80, 80))
    return img

def to_pil_mask(arr_bool, size=(256, 256)):
    """把 bool/0-1 mask 变为灰度 PIL 图（0/255），并按 size 缩放。"""
    if isinstance(arr_bool, torch.Tensor):
        arr_bool = arr_bool.detach().cpu().numpy()
    arr_u8 = (arr_bool.astype(np.uint8) * 255)
    return Image.fromarray(arr_u8).resize(size)

def save_quad_figure(real_img, tp_img, fp_img, fn_img, out_path):
    """2x2 排版：左上 real、右上 TP、左下 FP、右下 FN。"""
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
    titles = ["real", "TP", "FP", "FN"]
    imgs = [real_img, tp_img, fp_img, fn_img]
    cmaps = [None, "gray", "gray", "gray"]

    for ax, im, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        ax.imshow(im if cmap is None else im, cmap=cmap)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def to_u8(mask_bool):
    return (mask_bool.astype(np.uint8) * 255)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ------------------------- arg parsing & model/dataset -------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="Instance-level TP/FP/FN visualization (no training)")
    # model / tokenizer
    p.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    p.add_argument("--precision", default="fp16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    p.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H")
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--use_mm_start_end", action="store_true", default=True)
    p.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])

    # dataset / loader
    p.add_argument("--dataset_dir", default="./dataset")
    p.add_argument("--split", default="validation", choices=["train", "validation", "val", "test"])
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=0, help=">0 to limit processed batches")

    # visualization root
    p.add_argument("--vis_root", default="./vis_output")

    # heads kept for model signature compat
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--ce_loss_weight", type=float, default=1.0)
    p.add_argument("--dice_loss_weight", type=float, default=1.0)
    p.add_argument("--bce_loss_weight", type=float, default=1.0)
    p.add_argument("--cls_loss_weight", type=float, default=1.0)
    p.add_argument("--mask_loss_weight", type=float, default=1.0)
    p.add_argument("--train_mask_decoder", action="store_true", default=True)
    p.add_argument("--num_obj_classes", type=int, default=81)
    p.add_argument("--obj_loss_weight", type=float, default=1.0)

    return p.parse_args(argv)

def build_tokenizer_and_model(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CLS]"); tokenizer.add_tokens("[SEG]"); tokenizer.add_tokens("[OBJ]")
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_args = dict(
        train_mask_decoder=args.train_mask_decoder,
        out_dim=args.out_dim,
        cls_loss_weight=args.cls_loss_weight,
        mask_loss_weight=args.mask_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        cls_token_idx=args.cls_token_idx,
        seg_token_idx=args.seg_token_idx,
        obj_token_idx=args.obj_token_idx,
        num_obj_classes=args.num_obj_classes,
        obj_loss_weight=args.obj_loss_weight,
        vision_pretrained=args.vision_pretrained,
        vision_tower=args.vision_tower,
        use_mm_start_end=args.use_mm_start_end,
    )
    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    return tokenizer, model, device

def build_loader(args, tokenizer):
    dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split="validation" if args.split == "val" else args.split,
        precision=args.precision,
        image_size=args.image_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=0,
            cls_token_idx=args.cls_token_idx,
            obj_token_idx=args.obj_token_idx,
        ),
    )
    return dataset, loader


# ------------------------- helpers -------------------------

def extract_instance_name_and_id(path_like: str, fallback_name: str):
    """
    从诸如 /.../tampered_0a02a15af775.png 中提取：
    - instance_name: tampered_0a02a15af775
    - instance_id:   0a02a15af775
    """
    name = os.path.splitext(os.path.basename(path_like))[0] if path_like else fallback_name
    m = re.match(r"^(tampered_)([^\.]+)$", name)
    if m:
        instance_name = name
        instance_id = m.group(2)
        return instance_name, instance_id
    m2 = re.match(r"^(tampered_)([^\.]+)", name)
    if m2:
        instance_name = f"tampered_{m2.group(2)}"
        instance_id = m2.group(2)
        return instance_name, instance_id
    return name, name.replace("tampered_", "")

def find_real_image(args, instance_id: str):
    """
    在 args.dataset_dir/validation/real/ 下查找 original_<id>.(png|jpg|jpeg)
    """
    base = os.path.join(args.dataset_dir, "validation", "real")
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        cand = os.path.join(base, f"original_{instance_id}.{ext}")
        if os.path.exists(cand):
            return cand
    return None

def guess_tampered_path(input_dict):
    """尽力从 batch 的 input_dict 中拿到当前样本的图像路径。"""
    for k in ["img_paths", "paths", "image_paths"]:
        if k in input_dict and len(input_dict[k]) > 0:
            return input_dict[k][0]
    return None


# ------------------------- main visualization -------------------------

def visualize(args):
    tokenizer, model, device = build_tokenizer_and_model(args)
    dataset, loader = build_loader(args, tokenizer)
    ensure_dir(args.vis_root)

    print(f"[Info] Split: {args.split} | Samples: {len(dataset)}")
    print(f"[Info] Saving per-instance visualizations to: {args.vis_root}")

    n_batches = len(loader)
    if args.max_batches > 0:
        n_batches = min(n_batches, args.max_batches)

    with torch.no_grad():
        for bidx, input_dict in enumerate(tqdm.tqdm(loader, total=n_batches)):
            if bidx >= n_batches:
                break

            # precision & device
            input_dict = dict_to_cuda(input_dict)

            # === 可选：跳过 Real（label==0）/ 仅看 Tampered（label==2） ===
            cls_labels = input_dict.get("cls_labels", None)
            if cls_labels is not None:
                label = int(cls_labels[0])
                # 0=Real, 1=Full Synthetic, 2=Tampered
                if label == 0:
                    continue
                elif label != 2:
                    continue

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            input_dict["inference"] = True
            out = model(**input_dict)

            tampered_path = guess_tampered_path(input_dict)
            fallback_name = f"tampered_batch{bidx}"
            instance_base_name, instance_id = extract_instance_name_and_id(tampered_path, fallback_name)

            gt_list = out["gt_soft_masks"][0].int()          # (N, H, W)
            pred_list = (out["pred_masks"][0] > 0).int()     # (N, H, W) 或 (1, H, W)

            if pred_list.dim() == 2:
                pred_list = pred_list.unsqueeze(0)  # -> (1,H,W)

            N_gt = gt_list.shape[0]
            N_pr = pred_list.shape[0]
            N = min(N_gt, N_pr)

            for i in range(N):
                gt_i = (gt_list[i] > 0)
                pr_i = (pred_list[i] > 0)

                inst_folder_name = f"{instance_base_name}"
                inst_dir = os.path.join(args.vis_root, inst_folder_name)

                gt_np = gt_i.detach().cpu().numpy().astype(bool)
                pr_np = pr_i.detach().cpu().numpy().astype(bool)
                tp = pr_np & gt_np
                fp = pr_np & (~gt_np)
                fn = (~pr_np) & gt_np

                TP = int(tp.sum())
                FP = int(fp.sum())
                FN = int(fn.sum())
                denom = TP + FP + FN
                pixel_acc = (TP / denom) if denom > 0 else 0.0

                if pixel_acc <= 0:
                    continue
                ensure_dir(inst_dir)
                

                # 保存 tp/fp/fn 二值图（0/255）
                Image.fromarray(to_u8(tp)).save(os.path.join(inst_dir, "tp.png"))
                Image.fromarray(to_u8(fp)).save(os.path.join(inst_dir, "fp.png"))
                Image.fromarray(to_u8(fn)).save(os.path.join(inst_dir, "fn.png"))

                # ---- 保存图像与掩码 ----
                vis_size = (256, 256)

                tampered_img_pil = load_or_placeholder(
                    tampered_path, size=vis_size, text="tampered not found"
                )
                tampered_img_pil.save(os.path.join(inst_dir, "tampered.png"))

                soft_mask_pil = to_pil_mask(gt_i, size=vis_size)
                soft_mask_pil.save(os.path.join(inst_dir, "soft_mask.png"))

                real_path = find_real_image(args, instance_id)
                real_img_pil = load_or_placeholder(real_path, size=vis_size, text="real not found")
                real_img_pil.save(os.path.join(inst_dir, "real.png"))

                tp_pil = to_pil_mask(tp, size=vis_size)
                fp_pil = to_pil_mask(fp, size=vis_size)
                fn_pil = to_pil_mask(fn, size=vis_size)
                quad_path = os.path.join(inst_dir, "quad.png")
                save_quad_figure(real_img_pil, tp_pil, fp_pil, fn_pil, quad_path)

                # 保存 acc.txt
                with open(os.path.join(inst_dir, "acc.txt"), "w", encoding="utf-8") as f:
                    f.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\n")
                    f.write(f"PixelAcc (TP/(TP+FP+FN)): {pixel_acc:.6f}\n")


    print("[Done] Instance-level visualizations saved.")

def main(argv):
    args = parse_args(argv)
    os.makedirs(args.vis_root, exist_ok=True)
    visualize(args)

if __name__ == "__main__":
    main(sys.argv[1:])
