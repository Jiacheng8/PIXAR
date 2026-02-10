import os

# 你要扫描的路径
path = r"/workspace/dataset/raw_outputs"

# 输出文件名
output_file = "/workspace/utils/construct_dataset/temp_log/folder_list.txt"

# 获取所有文件夹（不递归）
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

# 写入到 txt 文件
with open(output_file, "w", encoding="utf-8") as f:
    for folder in folders:
        f.write(folder + "\n")

print(f"已将 {len(folders)} 个文件夹名写入 {output_file}")
