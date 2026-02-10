#!/usr/bin/env python3
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

ROOT = Path("/data/ironman/jiacheng/final_Omni_Data/raw_outputs")
DRY_RUN = False

# 65GB 通常 IO 是瓶颈：线程别太多，建议 4~12 之间试
WORKERS = 8

# 分批提交，避免 futures 堆积；建议 1000~5000
CHUNK = 2000

def clean(p: Path) -> int:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    if not DRY_RUN:
        # 写回：会清掉 iCCP/ICC 等元数据
        ok = cv2.imwrite(str(p), img)
        return 1 if ok else 0
    return 1

print("🔍 Scanning + Cleaning...", flush=True)

scanned = 0
cleaned = 0

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = []

    scan_bar = tqdm(desc="Scanning PNG", unit="file", file=sys.stderr)
    proc_bar = tqdm(desc="Cleaning PNG", unit="file", file=sys.stderr)

    for p in ROOT.rglob("*.png"):
        if not p.is_file():
            continue

        scanned += 1
        scan_bar.update(1)

        futures.append(ex.submit(clean, p))

        # 分批回收结果，控制内存
        if len(futures) >= CHUNK:
            for fut in as_completed(futures):
                cleaned += fut.result()
                proc_bar.update(1)
            futures.clear()

    # 收尾回收
    for fut in as_completed(futures):
        cleaned += fut.result()
        proc_bar.update(1)

    scan_bar.close()
    proc_bar.close()

print(f"✅ Done. scanned={scanned}, cleaned={cleaned}, dry_run={DRY_RUN}")
