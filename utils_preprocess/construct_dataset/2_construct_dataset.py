#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高效数据构建脚本（逻辑等价 + 并行 + tqdm）
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
import cv2
import numpy as np
from tqdm import tqdm


def ensure_dirs(base: Path):
    """创建输出目录结构"""
    for split in ("train", "test", "validation"):
        for sub in ("masks", "tampered", "soft_masks", "real", "full_synthetic", "metadata"):
            (base / split / sub).mkdir(parents=True, exist_ok=True)


def compute_soft_map_strict(real_bgr: np.ndarray, ai_bgr: np.ndarray, tao: float):
    """严格等价于 ToTensor 风格的 RGB 平均差 + 阈值"""
    hR, wR = real_bgr.shape[:2]
    hA, wA = ai_bgr.shape[:2]
    if (hA, wA) != (hR, wR):
        interp = cv2.INTER_CUBIC if (hA < hR or wA < wR) else cv2.INTER_AREA
        ai_bgr = cv2.resize(ai_bgr, (wR, hR), interpolation=interp)

    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ai_rgb   = cv2.cvtColor(ai_bgr,   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    diff = np.abs(real_rgb - ai_rgb)
    mean_diff = diff.mean(axis=2)
    soft = (mean_diff > float(tao)).astype(np.uint8) * 255
    return soft, ai_bgr


def process_one_entry(task, cfg):
    """处理单个数据条目"""
    try:
        real = cv2.imread(task["real"], cv2.IMREAD_COLOR)
        ai   = cv2.imread(task["ai"],   cv2.IMREAD_COLOR)
        if real is None or ai is None:
            return False, f"skip(no image): {task['entry']}"

        soft_mask, ai_sync = compute_soft_map_strict(real, ai, cfg["tao"])

        if cfg["overwrite_ai"] and ai_sync is not ai:
            cv2.imwrite(task["ai"], ai_sync)

        temper_idx = cfg["temper_idx_func"]()
        split = cfg["dest_type"]
        base  = Path(cfg["output_dir"]) / split

        dst_ai_image = base / "tampered"   / f"tampered_{temper_idx:05d}.png"
        dst_soft_map = base / "soft_masks" / f"tampered_{temper_idx:05d}_mask.png"
        dst_orig_mask= base / "masks"      / f"tampered_{temper_idx:05d}_mask.png"
        dst_meta     = base / "metadata"   / f"tampered_{temper_idx:05d}_cls.json"
        dst_real = base / "real" / f"original_{real_idx:05d}.png"
        

        ai_img = cv2.imread(task["ai"], cv2.IMREAD_COLOR)
        cv2.imwrite(str(dst_ai_image), ai_img)

        cv2.imwrite(str(dst_soft_map), soft_mask)
        if not os.path.exists(task["orig_mask"]):
            cv2.imwrite(str(dst_orig_mask), soft_mask)
        else:
            mask = cv2.imread(task["orig_mask"], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(dst_orig_mask), mask)


        with open(task["cls_info"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        cls_info = meta.get("replacement_categories", [])
        with open(dst_meta, "w", encoding="utf-8") as f:
            json.dump({"cls": cls_info}, f, ensure_ascii=False, indent=2)

        if cfg["mark_saved_real"](task["image_root"]):
            real_idx = cfg["real_idx_func"]()
            real_img = cv2.imread(task["real"], cv2.IMREAD_COLOR)
            cv2.imwrite(str(dst_real), real_img)

        cfg["revmap"][f"tampered_{temper_idx:05d}.png"] = task["entry"]
        return True, f"ok: {task['entry']}"
    except Exception as e:
        return False, f"err: {task['entry']} -> {e}"


def main():
    ap = argparse.ArgumentParser(description="Fast dataset builder (OpenCV+NumPy, parallel, tqdm)")
    ap.add_argument("--id", required=True, help="dataset id under --dataset-dir")
    ap.add_argument("--dataset-dir", default="/workspace/dataset/raw_outputs_training")
    ap.add_argument("--output-dir",  default="/workspace/dataset/demo")
    ap.add_argument("--tao", type=float, default=0.1)
    ap.add_argument("--dest-type", choices=["train","test","validation"], default="train")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    ap.add_argument("--overwrite-ai", action="store_true", help="overwrite generated.png after size sync")
    args = ap.parse_args()

    out_root = Path(f"{args.output_dir}_{args.tao}")
    ensure_dirs(out_root)

    json_file = out_root / "mapping.json"
    if not json_file.exists():
        json_file.write_text("{}", encoding="utf-8")
        print(f"Created new mapping file: {json_file}")
    reverse_mapping = json.loads(json_file.read_text(encoding="utf-8"))

    tampered_dir = out_root / args.dest_type / "tampered"
    real_dir     = out_root / args.dest_type / "real"
    tampered_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    _t = len([p for p in tampered_dir.iterdir() if p.is_file()])
    _r = len([p for p in real_dir.iterdir() if p.is_file()])

    t_lock = Lock()
    r_lock = Lock()
    def next_temper_idx():
        nonlocal _t
        with t_lock:
            idx = _t; _t += 1
            return idx
    def next_real_idx():
        nonlocal _r
        with r_lock:
            idx = _r; _r += 1
            return idx

    saved_real_flag = {}
    s_lock = Lock()
    def mark_saved_real(image_root: str):
        with s_lock:
            if image_root in saved_real_flag:
                return False
            saved_real_flag[image_root] = True
            return True

    dataset_path = Path(args.dataset_dir) / args.id
    tasks = []
    for name in sorted(os.listdir(dataset_path)):
        dir_path = dataset_path / name
        if not dir_path.is_dir():
            continue
        for sub in os.listdir(dir_path):
            file_path = dir_path / sub
            if not file_path.is_dir():
                continue
            real = file_path / "original.png"
            ai   = file_path / "generated.png"
            cls  = file_path / "replace_info.json"
            if not (real.exists() and ai.exists() and cls.exists()):
                continue
            entry = f"{name}/{sub}"
            tasks.append({
                "dir_path": str(file_path),
                "image_root": str(dir_path),
                "orig_mask": str(file_path / "mask.png"),
                "real": str(real),
                "ai":   str(ai),
                "cls_info": str(cls),
                "entry": entry,
            })

    cfg = {
        "output_dir": str(out_root),
        "dest_type": args.dest_type,
        "tao": args.tao,
        "overwrite_ai": args.overwrite_ai,
        "temper_idx_func": next_temper_idx,
        "real_idx_func": next_real_idx,
        "mark_saved_real": mark_saved_real,
        "revmap": reverse_mapping,
    }

    workers = max(1, args.workers)
    ok_cnt = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_entry, t, cfg) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            ok, _ = fut.result()
            if ok:
                ok_cnt += 1

    json_file.write_text(json.dumps(reverse_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Done. processed={ok_cnt}/{len(tasks)}; output={out_root}; workers={workers}")


if __name__ == "__main__":
    main()
