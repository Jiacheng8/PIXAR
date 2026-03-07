#!/usr/bin/env python3
"""
将 descriptions_prev.csv 中 workspace 列以 coco_val_ 开头的条目迁移到 descriptions.csv
"""
import csv

SRC = "/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions_prev.csv"
DST = "/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions.csv"

# 读取目标文件中已有的 relative_path，避免重复写入
print("读取目标文件已有条目...")
existing = set()
with open(DST, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        existing.add(row["relative_path"])
print(f"目标文件已有 {len(existing)} 条记录")

# 从源文件筛选 workspace 以 coco_val_ 开头且不重复的条目
to_append = []
with open(SRC, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["workspace"].startswith("coco_val_") and row["relative_path"] not in existing:
            to_append.append(row)

print(f"待写入 {len(to_append)} 条新记录（已跳过重复）")

if not to_append:
    print("无需写入，退出。")
    exit(0)

# 追加写入目标文件
with open(DST, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "ann_id", "workspace", "relative_path", "descriptions"])
    writer.writerows(to_append)

print("迁移完成。")
