import json
import argparse
from pathlib import Path

def load_bad_list(path_txt: Path):
    bad = set()
    with path_txt.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                bad.add(name)
    return bad

def extract_original_name(value: str, only_before_slash: bool = True) -> str:
    """从类似 '000000009590/ann_84206' 中提取原始图像名。
    默认取斜杠前（'000000009590'）。如果没有斜杠，返回原样。
    """
    if only_before_slash and "/" in value:
        return value.split("/", 1)[0]
    return value

def main():
    ap = argparse.ArgumentParser(description="Filter bad tampered files and export original image names.")
    ap.add_argument("--bad_txt", required=True, help="不合格名单 txt（每行一个 tampered_xxxx.png）")
    ap.add_argument("--mapping_json", required=True, help="映射 json（{tampered: 'img/ann_xxx', ...}）")
    ap.add_argument("--out_txt", required=True, help="输出的 txt（每行一个原始图像名）")
    ap.add_argument("--full_value", action="store_true",
                    help="默认只输出斜杠前的原始图像名；加此开关则输出整个值（如 '000000009590/ann_84206'）")
    args = ap.parse_args()

    bad = load_bad_list(Path(args.bad_txt))

    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # 你的格式就是 dict

    ok_originals = []
    for tampered, value in mapping.items():
        if tampered not in bad:
            name = extract_original_name(value, only_before_slash=not args.full_value)
            ok_originals.append(name)

    # 去重并保持顺序
    seen = set()
    dedup = []
    for x in ok_originals:
        if x not in seen:
            seen.add(x)
            dedup.append(x)

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for n in dedup:
            f.write(n + "\n")

    print(f"完成！筛出 {len(dedup)} 条写入 {out_path}")

if __name__ == "__main__":
    main()
