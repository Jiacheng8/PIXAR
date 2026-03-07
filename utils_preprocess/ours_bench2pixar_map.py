import os
import shutil
import csv
import json
import argparse

# === 参数部分（按需修改） ===
parser = argparse.ArgumentParser(description="PIXAR Model Training")
parser.add_argument("--name", default="incr_20251022_220942", type=str)
args = parser.parse_args()

source_root = f"/workspace/dataset/raw_outputs/{args.name}"   # 源目录

target_tampered = f"/workspace/dataset/{args.name}/test/tampered"
target_masks = f"/workspace/dataset/{args.name}/test/masks"
target_real = f"/workspace/dataset/{args.name}/test/real"
out_dir = f"/workspace/dataset/{args.name}/test"              # 映射文件输出目录

os.makedirs(f"/workspace/dataset/{args.name}/test/full_synthetic", exist_ok=True)
os.makedirs(f"/workspace/dataset/{args.name}/train/full_synthetic", exist_ok=True)
os.makedirs(f"/workspace/dataset/{args.name}/train/masks", exist_ok=True)
os.makedirs(f"/workspace/dataset/{args.name}/train/tampered", exist_ok=True)
os.makedirs(f"/workspace/dataset/{args.name}/train/real", exist_ok=True)


# 是否执行拷贝（设为 False 时只生成映射）
COPY_FILES = True

# 创建目标路径（如不存在）
os.makedirs(target_tampered, exist_ok=True)
os.makedirs(target_masks, exist_ok=True)
os.makedirs(target_real, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# 映射文件路径
csv_path = os.path.join(out_dir, "mapping.csv")
json_path = os.path.join(out_dir, "mapping.json")
rev_json_path = os.path.join(out_dir, "reverse_mapping.json")

counter = 1       # tampered/mask 的编号从 1 开始
real_counter = 1  # real 的编号从 1 开始

# 用于写出映射
rows = []
mapping = {}         # key: "image_id/anno_id" -> "tampered_0000{id}.png"
reverse_mapping = {} # key: "tampered_0000{id}.png" -> "image_id/anno_id"

# 遍历所有 image_id（你的代码里是 class_name）
for image_id in sorted(os.listdir(source_root)):
    image_path = os.path.join(source_root, image_id)
    if not os.path.isdir(image_path):
        continue

    saved_real_for_image = False

    # 遍历 ann_xxxx 子目录（anno_id）
    for anno_id in sorted(os.listdir(image_path)):
        ann_path = os.path.join(image_path, anno_id)
        if not os.path.isdir(ann_path):
            continue

        gen_path = os.path.join(ann_path, "generated.png")
        mask_path = os.path.join(ann_path, "mask.png")
        # mask_path = os.path.join(ann_path, "ground_truth_hard_map.png")
        real_path = os.path.join(ann_path, "original.png")

        if not os.path.exists(gen_path) or not os.path.exists(mask_path):
            print(f"⚠️ 跳过 {ann_path}（generated.png 或 ground_truth_hard_map.png 缺失）")
            continue

        # 生成目标文件名（tampered/mask 共用同一个编号）
        filename_idx = f"{counter:05d}"
        out_gen = os.path.join(target_tampered, f"tampered_{filename_idx}.png")
        out_mask = os.path.join(target_masks,   f"tampered_{filename_idx}_mask.png")

        # real 只在该 image_id 的第一个样本保存一次（保持你原逻辑）
        if not saved_real_for_image and os.path.exists(real_path):
            real_idx = f"{real_counter:05d}"
            real_gen = os.path.join(target_real, f"real_{real_idx}.png")
            if COPY_FILES:
                shutil.copy(real_path, real_gen)
            saved_real_for_image = True
            real_counter += 1
        else:
            real_gen = ""  # 没保存时留空，CSV里也能看到

        # 拷贝 tampered 和 mask
        if COPY_FILES:
            shutil.copy(gen_path, out_gen)
            shutil.copy(mask_path, out_mask)

        # 记录映射：image_id/anno_id  <-> tampered_0000{id}.png
        key = f"{image_id}/{anno_id}"
        tampered_name = f"tampered_{filename_idx}.png"
        mask_name = f"tampered_{filename_idx}_mask.png"

        mapping[key] = tampered_name
        reverse_mapping[tampered_name] = key

        # 记录到 CSV 行（相对路径，便于迁移）
        rows.append({
            "image_id": image_id,
            "anno_id": anno_id,
            "pair_key": key,
            "tampered": os.path.relpath(out_gen, out_dir),
            "mask": os.path.relpath(out_mask, out_dir),
            "real": os.path.relpath(real_gen, out_dir) if real_gen else ""
        })

        # print(f"✅ {key} -> {tampered_name}")
        counter += 1

# 写出 CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "anno_id", "pair_key", "tampered", "mask", "real"])
    writer.writeheader()
    writer.writerows(rows)

# 写出 JSON（正向与反向）
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

with open(rev_json_path, "w", encoding="utf-8") as f:
    json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)

print("🎯 所有文件处理完成！")
print(f"🗂 映射已生成：\n- CSV: {csv_path}\n- JSON: {json_path}\n- 反向JSON: {rev_json_path}")
