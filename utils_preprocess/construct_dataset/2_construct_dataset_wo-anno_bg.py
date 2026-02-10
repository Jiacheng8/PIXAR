#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import json
import torch
from PIL import Image
import torchvision.transforms as transforms  # 保留不使用也可删
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import count
import threading
import hashlib


def sync_ai_image_size(ai_image_path: str, real_image_path: str) -> bool:
    """
    读取 ai_image 与 real_image；若尺寸不同，则将 ai_image resize 到 real_image 尺寸并覆盖保存。
    返回值：是否发生了 resize（True/False）
    """
    ai = cv2.imread(ai_image_path, cv2.IMREAD_UNCHANGED)
    real = cv2.imread(real_image_path, cv2.IMREAD_UNCHANGED)

    if ai is None:
        raise FileNotFoundError(f"无法读取 ai_image: {ai_image_path}")
    if real is None:
        raise FileNotFoundError(f"无法读取 real_image: {real_image_path}")

    h_ai, w_ai = ai.shape[:2]
    h_real, w_real = real.shape[:2]

    if (h_ai, w_ai) == (h_real, w_real):
        return False

    interp = cv2.INTER_CUBIC if (h_ai < h_real or w_ai < w_real) else cv2.INTER_AREA
    ai_resized = cv2.resize(ai, (w_real, h_real), interpolation=interp)

    ok = cv2.imwrite(ai_image_path, ai_resized)
    if not ok:
        raise IOError(f"保存失败：{ai_image_path}")

    return True


def compute_diff_maps(
    real_image_path: str,
    generated_image_path: str,
    threshold: float
) -> torch.Tensor:
    """
    计算像素级平均 RGB 差分并与阈值比较，返回二值 soft_map（0/1 的 torch.Tensor，HxW）。
    为提速，内部用 OpenCV+NumPy 实现图像读取与差分，不再走 PIL/ToTensor。
    """
    # 读图（BGR, uint8）
    real_bgr = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
    gen_bgr = cv2.imread(generated_image_path, cv2.IMREAD_COLOR)
    if real_bgr is None or gen_bgr is None:
        raise FileNotFoundError(f"读取图像失败: {real_image_path} 或 {generated_image_path}")

    # 尺寸必须一致（你外层已做 sync，这里断言一下）
    if real_bgr.shape[:2] != gen_bgr.shape[:2]:
        raise AssertionError("Input images must have the same dimensions (请先调用 sync_ai_image_size).")

    # BGR -> RGB，归一化到 [0,1]
    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gen_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # |real - gen| 的通道均值
    mean_diff = np.abs(real_rgb - gen_rgb).mean(axis=2)  # HxW, float32 in [0,1]

    # 阈值化得到二值图（0/1）
    soft_bin = (mean_diff > float(threshold)).astype(np.uint8)  # 0/1

    # 转回 torch.Tensor（与原函数返回兼容）
    soft_map = torch.from_numpy(soft_bin).to(torch.float32)
    return soft_map


def save_soft_map(soft_map: torch.Tensor, save_path: str):
    # 兼容：soft_map 为 0/1 的张量 -> 0/255 保存
    img = (soft_map * 255).byte().cpu().numpy()
    pil_img = Image.fromarray(img)
    pil_img.save(save_path)


def hash_string(s: str, length: int = 16) -> str:
    """
    对字符串做 SHA1，并截断为指定长度的 hex 字符串。
    """
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()[:length]


def hash_file(path: str, length: int = 16) -> str:
    """
    对文件内容做 SHA1，并截断为指定长度。
    用于 real 图 ID，使其与图像内容绑定，跨 run/跨数据源都稳定。
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:length]


def main():
    ap = argparse.ArgumentParser(description="Construct dataset from provided parameters.")
    ap.add_argument("--id", required=True, help="Path to the dataset file.")
    ap.add_argument(
        "--dataset-dir",
        default="/workspace/dataset/raw_outputs_training",
        help="Path to the dataset directory.",
    )
    ap.add_argument(
        "--output-dir",
        default="/workspace/dataset/demo",
        help="Directory to save filtered dataset.",
    )
    ap.add_argument("--tao", help="Threshold value for ground truth.", type=float, default=0.1)
    ap.add_argument(
        "--dest-type",
        default="train",
        help="Destination type: train/test/validation.",
        type=str,
        choices=["train", "test", "validation"],
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads (default: 2 * CPU cores, capped at 32).",
    )

    args = ap.parse_args()
    args.output_dir = f"{args.output_dir}_{args.tao}"
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.dataset_dir, args.id)
    json_file = os.path.join(args.output_dir, "mapping.json")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
        print(f"Created new mapping file: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        reverse_mapping = json.load(f)

    # 目录结构
    for split in ("train", "test", "validation"):
        os.makedirs(os.path.join(args.output_dir, split, "masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "tampered"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "soft_masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "real"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "full_synthetic"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "metadata"), exist_ok=True)

    # real 统一仍然放在 train/real 里（保持你原来的逻辑）
    real_root = os.path.join(args.output_dir, "train", "real")

    # 只保留目录名
    all_names = sorted(
        [
            name
            for name in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, name))
        ]
    )

    print(f"Found {len(all_names)} samples in dataset: {dataset_path}")

    # 线程锁：保护 reverse_mapping 更新
    mapping_lock = threading.Lock()

    # 线程池大小
    if args.num_workers is not None and args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = os.cpu_count() or 4
        num_workers = min(32, num_workers * 2)

    print(f"Using {num_workers} worker threads.")

    dest_type = args.dest_type  # 为了闭包里方便用

    def process_one(name: str) -> int:
        """
        处理单个样本目录。
        返回 1 表示处理成功，0 表示跳过（例如缺文件）。
        """
        file_path = os.path.join(dataset_path, name)

        entry = f"{name}"

        orig_mask_path = os.path.join(file_path, "mask.png")
        real_image_path = os.path.join(file_path, "original.png")
        ai_image_path = os.path.join(file_path, "generated.png")
        cls_info_path = os.path.join(file_path, "replace_info.json")

        # 先对齐尺寸（generated -> original 尺寸）
        try:
            _ = sync_ai_image_size(ai_image_path, real_image_path)
        except FileNotFoundError:
            # 用 tqdm.write 避免打乱进度条
            tqdm.write(f"Skipping {entry} due to missing image.")
            return 0

        if not (
            os.path.exists(ai_image_path)
            and os.path.exists(real_image_path)
            and os.path.exists(cls_info_path)
        ):
            tqdm.write(f"Skipping {entry} due to missing original image or metadata.")
            return 0

        # 计算 soft map
        try:
            soft_map = compute_diff_maps(real_image_path, ai_image_path, threshold=args.tao)
        except Exception as e:
            tqdm.write(f"Error computing diff for {entry}: {e}")
            return 0

        # ---------- 生成稳定的唯一 ID ----------

        # tampered ID：基于 (dataset-id + 目录名) 的哈希
        tampered_id = hash_string(f"{args.id}/{name}", length=16)

        # real ID：基于 original.png 文件内容的哈希
        try:
            real_id = hash_file(real_image_path, length=16)
        except Exception as e:
            tqdm.write(f"Error hashing real image for {entry}: {e}")
            return 0

        tampered_filename = f"tampered_{tampered_id}.png"
        tampered_mask_filename = f"tampered_{tampered_id}_mask.png"
        tampered_meta_filename = f"tampered_{tampered_id}_cls.json"

        real_filename = f"original_{real_id}.png"

        # 目标路径
        dst_ai_image = os.path.join(args.output_dir, dest_type, "tampered", tampered_filename)
        dst_soft_map = os.path.join(
            args.output_dir, dest_type, "soft_masks", tampered_mask_filename
        )
        dst_orig_mask = os.path.join(
            args.output_dir, dest_type, "masks", tampered_mask_filename
        )
        dst_meta = os.path.join(args.output_dir, dest_type, "metadata", tampered_meta_filename)

        dst_real_image = os.path.join(args.output_dir, dest_type, "real", real_filename)

        # 写 metadata
        try:
            with open(cls_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cls_info = data.get("replacement_categories", [])
            with open(dst_meta, "w", encoding="utf-8") as wf:
                json.dump({"cls": cls_info}, wf, ensure_ascii=False, indent=2)
        except Exception as e:
            tqdm.write(f"Error writing metadata for {entry}: {e}")
            return 0

        # 保存 tampered 图（覆盖写没问题）
        try:
            shutil.copy(ai_image_path, dst_ai_image)
        except Exception as e:
            tqdm.write(f"Error copying tampered image for {entry}: {e}")
            return 0

        # 保存 soft map
        try:
            save_soft_map(soft_map, dst_soft_map)
        except Exception as e:
            tqdm.write(f"Error saving soft map for {entry}: {e}")
            return 0

        # 保存原 mask（若无，则用 soft map）
        try:
            if not os.path.exists(orig_mask_path):
                save_soft_map(soft_map, dst_orig_mask)
            else:
                shutil.copy(orig_mask_path, dst_orig_mask)
        except Exception as e:
            tqdm.write(f"Error saving mask for {entry}: {e}")
            return 0

        # 更新反向映射：这里保持与你原来一样，只存 entry 字符串
        with mapping_lock:
            reverse_mapping[tampered_filename] = {
                "entry": entry,
                "real": real_filename,
                "type":args.id
            }

        # real 图：如果已经存在同名文件（内容 hash 相同），就不重复拷贝
        try:
            if not os.path.exists(dst_real_image):
                shutil.copy(real_image_path, dst_real_image)
        except Exception as e:
            tqdm.write(f"Error copying real image for {entry}: {e}")
            # real 拷贝失败不影响 tampered/soft_mask 的存在，这里不返回 0

        return 1

    # 多线程 + tqdm 进度条
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(process_one, all_names),
                total=len(all_names),
                desc="Processing samples",
                unit="img",
            )
        )

    # 写回 mapping.json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
