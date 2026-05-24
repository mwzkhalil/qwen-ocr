#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Value
from datasets import Image as HFImage

from augment import augment_image


def env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var '{key}' is not set.")
    return val


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


HF_TOKEN         = env("HF_TOKEN")
LOCAL_DATASET_DIR = Path(env("LOCAL_DATASET_DIR", "/workspace/hf_dataset"))
SHARD_SIZE       = env_int("SHARD_SIZE", 1000)
MAX_WORKERS      = env_int("MAX_WORKERS", 16)
MAX_SAMPLES_PER_DS = env_int("MAX_SAMPLES_PER_DS", -1)
VAL_FRAC         = env_float("VAL_FRAC", 0.05)

os.environ["HF_TOKEN"] = HF_TOKEN

FEATURES = Features({
    "image":      HFImage(),
    "text":       Value("string"),
    "class_name": Value("string"),
    "source":     Value("string"),
})


def normalise_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def download_image(url: str, retries: int = 3, timeout: int = 15) -> Optional[object]:
    import requests
    from PIL import Image
    import io
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    return None


def save_shards(rows: list, split_dir: Path, split_name: str) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    n_shards = max(1, (total + SHARD_SIZE - 1) // SHARD_SIZE)

    for shard_idx in range(n_shards):
        chunk = rows[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
        shard_path = split_dir / f"shard-{shard_idx+1:05d}-of-{n_shards:05d}.parquet"
        ds = Dataset.from_list(chunk, features=FEATURES)
        ds.to_parquet(str(shard_path))

    print(f"  [{split_name}] {n_shards} shards saved → {split_dir}")
    return n_shards


def build_row(image, text: str, class_name: str, source: str, is_validation: bool, severity: str) -> Optional[dict]:
    text = normalise_text(text)
    if not text:
        return None
    if image is None:
        return None
    aug_severity = "none" if is_validation else severity
    try:
        img = augment_image(image, aug_severity)
    except Exception:
        img = image
    return {"image": img, "text": text, "class_name": class_name, "source": source}


def load_persian_arabic(test_mode: bool = False):
    from datasets import load_dataset
    print("[persian-arabic] Loading mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Medium …")
    ds = load_dataset("mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Medium", token=HF_TOKEN)

    train_rows = []
    val_rows   = []

    for split_name, out_list, is_val in [("Train", train_rows, False), ("Test", val_rows, True)]:
        split = ds[split_name]
        limit = 3 if test_mode else (MAX_SAMPLES_PER_DS if MAX_SAMPLES_PER_DS > 0 else len(split))
        count = 0
        for row in split:
            if count >= limit:
                break
            r = build_row(
                image=row["Image"],
                text=row["Text"],
                class_name="persian-arabic",
                source="persian-arabic-medium",
                is_validation=is_val,
                severity="light",
            )
            if r:
                out_list.append(r)
                count += 1
        print(f"  [persian-arabic] {split_name}: {len(out_list)} rows collected")

    return train_rows, val_rows


def load_qaari(test_mode: bool = False):
    from datasets import load_dataset
    print("[qaari] Loading oddadmix/qaari-0.1-ocr-urdu-news-dataset-small …")
    ds = load_dataset("oddadmix/qaari-0.1-ocr-urdu-news-dataset-small", token=HF_TOKEN)

    train_rows = []
    val_rows   = []

    for split_name, out_list, is_val in [("train", train_rows, False), ("validation", val_rows, True)]:
        split = ds[split_name]
        limit = 3 if test_mode else (MAX_SAMPLES_PER_DS if MAX_SAMPLES_PER_DS > 0 else len(split))
        urls_texts = []
        for i, row in enumerate(split):
            if i >= limit:
                break
            text = normalise_text(row.get("text", ""))
            if not text:
                continue
            urls_texts.append((row["image"], text))

        print(f"  [qaari] {split_name}: downloading {len(urls_texts)} images with {MAX_WORKERS} workers …")

        def _fetch(item):
            url, text = item
            img = download_image(url)
            return img, text

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch, item): item for item in urls_texts}
            for fut in as_completed(futures):
                img, text = fut.result()
                if img is None or not text:
                    continue
                r = build_row(
                    image=img,
                    text=text,
                    class_name="urdu-news",
                    source="qaari-urdu-news",
                    is_validation=is_val,
                    severity="medium",
                )
                if r:
                    results.append(r)

        out_list.extend(results)
        print(f"  [qaari] {split_name}: {len(out_list)} rows after download")

    return train_rows, val_rows


def load_urdu_nastaliq(test_mode: bool = False):
    from datasets import load_dataset
    print("[urdu-nastaliq] Loading PuristanLabs1/urdu-ocr-1M nastaliq (streaming) …")

    train_rows = []
    val_rows   = []

    for split_name, out_list, is_val in [("train", train_rows, False), ("val", val_rows, True)]:
        ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", streaming=True, token=HF_TOKEN)
        split = ds[split_name]
        limit = 3 if test_mode else (MAX_SAMPLES_PER_DS if MAX_SAMPLES_PER_DS > 0 else float("inf"))
        count = 0
        for row in split:
            if count >= limit:
                break
            r = build_row(
                image=row["image"],
                text=row["text"],
                class_name="urdu-nastaliq",
                source="urdu-ocr-1M-nastaliq",
                is_validation=is_val,
                severity="heavy",
            )
            if r:
                out_list.append(r)
                count += 1
            if count % 10000 == 0 and count > 0:
                print(f"  [urdu-nastaliq] {split_name}: streamed {count} rows …")
        print(f"  [urdu-nastaliq] {split_name}: {len(out_list)} rows collected")

    return train_rows, val_rows


def mode_test():
    print("=" * 72)
    print("TEST MODE — streaming peek (3 rows each, no data saved)")
    print("=" * 72)
    load_persian_arabic(test_mode=True)
    load_qaari(test_mode=True)
    load_urdu_nastaliq(test_mode=True)
    print("\nTest mode complete. All datasets accessible.")


def mode_stats():
    from collections import Counter
    print("=" * 72)
    print("STATS — reading already-saved shards")
    print("=" * 72)
    for split in ("train", "validation"):
        split_dir = LOCAL_DATASET_DIR / split
        parquets = sorted(split_dir.glob("*.parquet"))
        if not parquets:
            print(f"  No shards found under {split_dir}")
            continue
        from datasets import load_dataset as _ld
        ds = _ld("parquet", data_files={split: str(split_dir / "*.parquet")})
        rows = ds[split]
        sources = Counter(rows["source"])
        total = len(rows)
        print(f"\n{split.upper()} — {total:,} rows, {len(parquets)} shards")
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {src:<30} {cnt:>8,}   ({cnt/total*100:.1f}%)")


def print_statistics(train_rows, val_rows, n_train_shards, n_val_shards):
    from collections import Counter
    import sys

    train_sources = Counter(r["source"] for r in train_rows)
    val_sources   = Counter(r["source"] for r in val_rows)
    total_train   = len(train_rows)
    total_val     = len(val_rows)

    estimated_bytes = (total_train + total_val) * 15000
    estimated_gb    = estimated_bytes / 1e9

    sep = "=" * 80
    print(f"\n{sep}")
    print("DATASET PREPARATION COMPLETE")
    print(sep)
    print(f"Total train rows : {total_train:>12,}")
    print(f"Total val rows   : {total_val:>12,}")
    print()
    print("Train breakdown by source:")
    for src, cnt in sorted(train_sources.items(), key=lambda x: -x[1]):
        print(f"  {src:<30} {cnt:>8,}   ({cnt/max(total_train,1)*100:.1f}%)")
    print()
    print("Val breakdown by source:")
    for src, cnt in sorted(val_sources.items(), key=lambda x: -x[1]):
        print(f"  {src:<30} {cnt:>8,}   ({cnt/max(total_val,1)*100:.1f}%)")
    print()
    print("Augmentation applied:")
    print("  persian-arabic-medium  → severity: light")
    print("  urdu-ocr-1M-nastaliq   → severity: heavy")
    print("  qaari-urdu-news        → severity: medium")
    print("  (all validation rows)  → severity: none")
    print()
    print("Shards saved:")
    print(f"  train/      : {n_train_shards:,} shards  @ {SHARD_SIZE:,} rows each")
    print(f"  validation/ : {n_val_shards:,} shards")
    print()
    print(f"Estimated disk size: ~{estimated_gb:.1f} GB")
    print(sep)


def mode_all():
    print("=" * 72)
    print("PREPARE MODE — download + augment + shard")
    print("=" * 72)

    pa_train, pa_val     = load_persian_arabic()
    qa_train, qa_val     = load_qaari()
    un_train, un_val     = load_urdu_nastaliq()

    train_rows = pa_train + qa_train + un_train
    val_rows   = pa_val   + qa_val   + un_val

    print(f"\n[merge] Total train rows: {len(train_rows):,}")
    print(f"[merge] Total val rows  : {len(val_rows):,}")

    print("\n[shard] Saving train shards …")
    n_train = save_shards(train_rows, LOCAL_DATASET_DIR / "train", "train")

    print("[shard] Saving validation shards …")
    n_val = save_shards(val_rows, LOCAL_DATASET_DIR / "validation", "validation")

    print_statistics(train_rows, val_rows, n_train, n_val)


def main():
    parser = argparse.ArgumentParser(description="Urdu OCR dataset preparation pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "test", "stats"],
        default="all",
        help="all=download+augment+shard, test=peek 3 rows, stats=report on saved shards",
    )
    args = parser.parse_args()

    if args.mode == "test":
        mode_test()
    elif args.mode == "stats":
        mode_stats()
    else:
        mode_all()


if __name__ == "__main__":
    main()
