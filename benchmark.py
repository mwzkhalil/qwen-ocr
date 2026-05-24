#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm


def env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var '{key}' is not set.")
    return val


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_bool(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes")


OCR_PROMPT   = env("OCR_PROMPT", "اس تصویر سے تمام متن نکالیں۔ صرف متن لکھیں۔")
LOAD_IN_4BIT = env_bool("LOAD_IN_4BIT", "false")
HF_TOKEN     = env("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN


def normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def cer(pred: str, gt: str) -> float:
    gt = normalise(gt)
    pred = normalise(pred)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    gt_words   = normalise(gt).split()
    pred_words = normalise(pred).split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return edit_distance(pred_words, gt_words) / len(gt_words)


def _edit_distance_words(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def wer_from_split(pred: str, gt: str) -> float:
    gt_words   = normalise(gt).split()
    pred_words = normalise(pred).split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit_distance_words(pred_words, gt_words) / len(gt_words)


def bleu4(pred: str, gt: str) -> float:
    pred_tokens = list(normalise(pred))
    gt_tokens   = list(normalise(gt))
    score = 0.0
    for n in range(1, 5):
        pred_ngrams: Dict[tuple, int] = {}
        gt_ngrams:   Dict[tuple, int] = {}
        for i in range(len(pred_tokens) - n + 1):
            ng = tuple(pred_tokens[i:i + n])
            pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1
        for i in range(len(gt_tokens) - n + 1):
            ng = tuple(gt_tokens[i:i + n])
            gt_ngrams[ng] = gt_ngrams.get(ng, 0) + 1
        matches = sum(min(cnt, gt_ngrams.get(ng, 0)) for ng, cnt in pred_ngrams.items())
        denom   = max(len(pred_tokens) - n + 1, 1)
        score  += math.log(max(matches / denom, 1e-10)) / 4.0
    bp = math.exp(min(0.0, 1 - len(gt_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(score)


def normalised_edit_distance(pred: str, gt: str) -> float:
    pred = normalise(pred)
    gt   = normalise(gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(pred, gt) / max_len


def load_dataset_split(dataset_dir: str, split: str, n_samples: int):
    from datasets import load_dataset as _ld
    split_dir = Path(dataset_dir) / split
    parquets  = sorted(split_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet shards found under {split_dir}")
    ds = _ld("parquet", data_files={split: str(split_dir / "*.parquet")})
    rows = ds[split]
    if n_samples > 0:
        rows = rows.select(range(min(n_samples, len(rows))))
    return rows


def run_inference_batch(model, tokenizer, images, prompt: str) -> List[str]:
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    results = []

    for image in images:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=1.0,
                do_sample=False,
            )

        out_ids = out_ids[0][len(inputs["input_ids"][0]):]
        pred = tokenizer.decode(out_ids, skip_special_tokens=True)
        results.append(pred)

    return results


def compute_metrics(predictions: List[str], ground_truths: List[str], sources: List[str]):
    global_cer   = []
    global_wer   = []
    global_bleu  = []
    global_ned   = []
    global_exact = []

    per_source_cer   = defaultdict(list)
    per_source_wer   = defaultdict(list)
    per_source_exact = defaultdict(list)

    per_sample = []

    for pred, gt, src in zip(predictions, ground_truths, sources):
        c  = cer(pred, gt)
        w  = wer_from_split(pred, gt)
        b  = bleu4(pred, gt)
        n  = normalised_edit_distance(pred, gt)
        ex = int(normalise(pred) == normalise(gt))

        global_cer.append(c)
        global_wer.append(w)
        global_bleu.append(b)
        global_ned.append(n)
        global_exact.append(ex)

        per_source_cer[src].append(c)
        per_source_wer[src].append(w)
        per_source_exact[src].append(ex)

        per_sample.append({
            "source": src,
            "ground_truth": gt,
            "prediction": pred,
            "cer": round(c, 4),
            "wer": round(w, 4),
            "bleu4": round(b, 4),
            "normalised_edit_distance": round(n, 4),
            "exact_match": ex,
        })

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    overall = {
        "n_samples":              len(predictions),
        "CER":                    round(avg(global_cer), 4),
        "WER":                    round(avg(global_wer), 4),
        "character_accuracy":     round(1.0 - avg(global_cer), 4),
        "exact_match":            round(avg(global_exact), 4),
        "BLEU4":                  round(avg(global_bleu), 4),
        "normalised_edit_distance": round(avg(global_ned), 4),
    }

    per_source = {}
    for src in sorted(per_source_cer):
        per_source[src] = {
            "n_samples":   len(per_source_cer[src]),
            "CER":         round(avg(per_source_cer[src]), 4),
            "WER":         round(avg(per_source_wer[src]), 4),
            "exact_match": round(avg(per_source_exact[src]), 4),
        }

    return overall, per_source, per_sample


def print_results(overall: dict, per_source: dict):
    sep   = "=" * 80
    line  = "-" * 80

    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)
    print(f"  Samples evaluated : {overall['n_samples']:,}")
    print(f"  CER               : {overall['CER']*100:.2f}%   (lower is better)")
    print(f"  1-CER (Char Acc.) : {overall['character_accuracy']*100:.2f}%")
    print(f"  WER               : {overall['WER']*100:.2f}%   (lower is better)")
    print(f"  Exact Match       : {overall['exact_match']*100:.2f}%")
    print(f"  BLEU-4            : {overall['BLEU4']:.4f}")
    print(f"  Norm. Edit Dist.  : {overall['normalised_edit_distance']:.4f}")
    print()
    print("Per-source breakdown:")
    print(f"  {'Source':<32} {'N':>6}  {'CER':>7}  {'WER':>7}  {'Exact':>7}")
    print(f"  {line[:70]}")
    for src, m in sorted(per_source.items()):
        print(f"  {src:<32} {m['n_samples']:>6}  {m['CER']*100:>6.2f}%  {m['WER']*100:>6.2f}%  {m['exact_match']*100:>6.2f}%")
    print()
    print("Reference benchmarks (published Urdu/Arabic OCR results):")
    print(f"  {'System':<35} {'CER':>12}  {'WER':>12}")
    print(f"  {line[:64]}")
    refs = [
        ("Google Vision API (Urdu)",   "8–15%",  "18–30%"),
        ("Tesseract 5 (Urdu)",          "25–40%", "45–65%  ← poor on Nastaliq"),
        ("EasyOCR (Arabic/Urdu)",       "12–22%", "25–40%"),
        ("SOTA academic (UTRNet)",       "3–6%",   "8–15%   ← printed Nastaliq only"),
        ("Production target",           "< 5%",   "< 12%"),
    ]
    for name, c, w in refs:
        print(f"  {name:<35} {c:>12}  {w}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Urdu OCR benchmark evaluation")
    parser.add_argument("--model_path",   required=True,  help="Path to LoRA or merged model")
    parser.add_argument("--dataset_dir",  default=None,   help="Path to sharded Parquet dataset (overrides LOCAL_DATASET_DIR)")
    parser.add_argument("--split",        default="validation", choices=["train", "validation"])
    parser.add_argument("--n_samples",    type=int, default=500, help="Rows to evaluate (-1 = all)")
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--output_file",  default="benchmark_results.json")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or env("LOCAL_DATASET_DIR", "/workspace/hf_dataset")

    print(f"[benchmark] Model       : {args.model_path}")
    print(f"[benchmark] Dataset dir : {dataset_dir}")
    print(f"[benchmark] Split       : {args.split}")
    print(f"[benchmark] Samples     : {args.n_samples}")

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_path,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)

    print(f"[benchmark] Loading dataset split …")
    rows = load_dataset_split(dataset_dir, args.split, args.n_samples)
    n    = len(rows)
    print(f"[benchmark] Evaluating {n:,} samples …")

    predictions   = []
    ground_truths = []
    sources       = []

    batch_size = args.batch_size

    for start in tqdm(range(0, n, batch_size), desc="Inference"):
        end   = min(start + batch_size, n)
        batch = rows.select(range(start, end))

        images  = batch["image"]
        texts   = batch["text"]
        srcs    = batch["source"]

        while True:
            try:
                preds = run_inference_batch(model, tokenizer, images, OCR_PROMPT)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    preds = [""] * len(images)
                    break
                batch_size = max(1, batch_size // 2)
                print(f"\n[benchmark] CUDA OOM — reducing batch_size to {batch_size}")

        predictions.extend(preds)
        ground_truths.extend(texts)
        sources.extend(srcs)

    overall, per_source, per_sample = compute_metrics(predictions, ground_truths, sources)

    print_results(overall, per_source)

    output = {
        "model_path":   args.model_path,
        "dataset_dir":  dataset_dir,
        "split":        args.split,
        "n_samples":    n,
        "overall":      overall,
        "per_source":   per_source,
        "per_sample":   per_sample,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[benchmark] Results saved → {args.output_file}")


if __name__ == "__main__":
    main()
