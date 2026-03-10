#!/usr/bin/env python3
"""
Rebuild descriptions.csv from mapping.json + metadata files.
"""
import os
import json
import csv
import argparse
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping-json", required=True, help="Path to mapping.json")
    ap.add_argument("--data-root", required=True,
                    help="Root dir containing train/test/validation subdirs (each with metadata/)")
    ap.add_argument("--output-csv", required=True, help="Output descriptions.csv path")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "validation"],
                    help="Which splits to process (default: train test validation)")
    args = ap.parse_args()

    # Load mapping
    print(f"Loading {args.mapping_json} ...")
    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"  {len(mapping)} entries")

    # Build tampered_id -> meta dict from mapping
    # key: tampered_id (e.g. "8aed609d73858635")
    # value: {entry, type, bg}
    id_to_info = {}
    for fname, info in mapping.items():
        # fname = "tampered_XXXXXXXXXXXXXXXX.png"
        tampered_id = fname.removeprefix("tampered_").removesuffix(".png")
        id_to_info[tampered_id] = info

    # Collect all metadata files from all splits
    splits = args.splits
    rows = []
    seen = set()  # deduplicate by (workspace, image_id, ann_id)

    for split in splits:
        meta_dir = os.path.join(args.data_root, split, "metadata")
        if not os.path.isdir(meta_dir):
            continue
        files = [f for f in os.listdir(meta_dir) if f.endswith("_cls.json")]
        print(f"  {split}: {len(files)} metadata files")

        for fname in tqdm(files, desc=split):
            # fname = "tampered_XXXXXXXXXXXXXXXX_cls.json"
            tampered_id = fname.removeprefix("tampered_").removesuffix("_cls.json")

            info = id_to_info.get(tampered_id)
            if info is None:
                print(f"  Warning: {tampered_id} not found in mapping, skipping")
                continue

            entry = info["entry"]       # "name/ann_id"  or  "name"
            workspace = info["type"]
            # match get_description logic: strip "qwen_" prefix for lookup
            if "qwen" in workspace:
                workspace = workspace[5:]

            parts = entry.split("/")
            if len(parts) == 2:
                image_id, raw_ann_id = parts
                # strip "ann_" prefix
                ann_id = raw_ann_id.removeprefix("ann_") if raw_ann_id.startswith("ann_") else raw_ann_id
            else:
                image_id = parts[0]
                ann_id = "-1"

            # Read text from metadata
            meta_path = os.path.join(meta_dir, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                text = meta.get("text", "")
            except Exception as e:
                print(f"  Warning: failed to read {meta_path}: {e}")
                text = ""

            key = (workspace, image_id, ann_id)
            if key in seen:
                continue
            seen.add(key)

            # relative_path: reconstruct
            if ann_id != "-1":
                relative_path = f"{workspace}/{image_id}/ann_{ann_id}/generated.png"
            else:
                relative_path = f"{workspace}/{image_id}/generated.png"

            rows.append({
                "image_id": image_id,
                "ann_id": ann_id,
                "workspace": workspace,
                "relative_path": relative_path,
                "descriptions": text,
            })

    print(f"\nTotal unique rows: {len(rows)}")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "ann_id", "workspace", "relative_path", "descriptions"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
