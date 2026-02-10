import os
import shutil

# === 参数部分 ===
source_root = "/workspace/dataset/benchmark_output"  # 源目录
target_tampered = "/workspace/dataset/ours/test/tampered"
target_masks = "/workspace/dataset/ours/test/masks"
target_real = "/workspace/dataset/ours/test/real"

# 创建目标路径（如不存在）
os.makedirs(target_tampered, exist_ok=True)
os.makedirs(target_masks, exist_ok=True)
os.makedirs(target_real, exist_ok=True)

counter = 1  # 从1开始编号
real_counter = 1
# 遍历所有 class 文件夹
for class_name in os.listdir(source_root):
    class_path = os.path.join(source_root, class_name)
    if not os.path.isdir(class_path):
        continue
    saved = False
    # 遍历 ann_xxxx 子文件夹
    for ann_name in sorted(os.listdir(class_path)):
        ann_path = os.path.join(class_path, ann_name)
        if not os.path.isdir(ann_path):
            continue

        gen_path = os.path.join(ann_path, "generated.png")
        mask_path = os.path.join(ann_path, "ground_truth_hard_map.png")
        real_path = os.path.join(ann_path, "original.png")

        if not os.path.exists(gen_path) or not os.path.exists(mask_path):
            print(f"⚠️ 跳过 {ann_path}（文件缺失）")
            continue

        # 生成目标文件名
        filename_idx = f"{counter:05d}"
        real_idx = f"{real_counter:05d}"
        out_gen = os.path.join(target_tampered, f"tampered_{filename_idx}.png")
        out_mask = os.path.join(target_masks, f"tampered_{filename_idx}_mask.png")
        real_gen = os.path.join(target_real, f"real_{real_idx}.png")
        # 复制文件
        # shutil.copy(gen_path, out_gen)
        # shutil.copy(mask_path, out_mask)
        if not saved:
            shutil.copy(real_path, real_gen)
            saved = True
            real_counter += 1
            

        print(f"✅ {ann_name} -> {out_gen}")
        counter += 1

print("🎯 所有文件处理完成！")
