#!/usr/bin/env python3
"""
Compare two metadata directories: check if every JSON file has the same content.
"""
import os
import json
import argparse
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", default="/data/ironman/jiacheng/final_Omni_Data/train/ours_0.05/train/metadata", help="First metadata directory")
    ap.add_argument("dir_b", default="/data/thor/jiacheng/omni_backup/train/ours_0.1/train/metadata", help="Second metadata directory")
    args = ap.parse_args()

    files_a = set(f for f in os.listdir(args.dir_a) if f.endswith(".json"))
    files_b = set(f for f in os.listdir(args.dir_b) if f.endswith(".json"))

    only_in_a = files_a - files_b
    only_in_b = files_b - files_a
    common = files_a & files_b

    print(f"Dir A: {len(files_a)} files  ({args.dir_a})")
    print(f"Dir B: {len(files_b)} files  ({args.dir_b})")
    print(f"Only in A: {len(only_in_a)}")
    print(f"Only in B: {len(only_in_b)}")
    print(f"Common:    {len(common)}")

    if only_in_a:
        print("\n[Only in A] first 10:")
        for f in sorted(only_in_a)[:10]:
            print(f"  {f}")

    if only_in_b:
        print("\n[Only in B] first 10:")
        for f in sorted(only_in_b)[:10]:
            print(f"  {f}")

    # Compare common files
    diff_count = 0
    diff_examples = []
    for fname in tqdm(sorted(common), desc="Comparing"):
        try:
            a = load_json(os.path.join(args.dir_a, fname))
            b = load_json(os.path.join(args.dir_b, fname))
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            diff_count += 1
            continue

        if a != b:
            diff_count += 1
            if len(diff_examples) < 5:
                diff_examples.append((fname, a, b))

    print(f"\nDiffering files: {diff_count} / {len(common)}")

    if diff_examples:
        print("\n[Diff examples]")
        for fname, a, b in diff_examples:
            print(f"\n  {fname}")
            print(f"    A: {a}")
            print(f"    B: {b}")

    if diff_count == 0 and not only_in_a and not only_in_b:
        print("\nAll files are identical.")
    elif diff_count == 0:
        print("\nAll common files are identical (but file sets differ).")


if __name__ == "__main__":
    main()
