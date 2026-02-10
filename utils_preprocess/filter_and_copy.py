import os
import shutil
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Copy matched subfolders listed in a txt file to a new directory.")
    parser.add_argument("--txt", required=True, help="包含合格路径的 txt 文件，每行如 000000580418/ann_76348")
    parser.add_argument("--src_root", required=True, help="源数据根目录")
    parser.add_argument("--dst_root", required=True, help="目标输出根目录")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    with open(args.txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    copied = 0
    missing = []

    for rel_path in lines:
        src_path = src_root / rel_path
        if src_path.exists() and src_path.is_dir():
            dst_path = dst_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # 复制整个文件夹（覆盖同名）
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            copied += 1
        else:
            missing.append(rel_path)

    print(f"✅ 已复制 {copied} 个文件夹到 {dst_root}")
    if missing:
        print(f"⚠️ 以下 {len(missing)} 个路径未找到：")
        for m in missing[:10]:
            print("  ", m)
        if len(missing) > 10:
            print("  ...（省略其余）")

if __name__ == "__main__":
    main()
