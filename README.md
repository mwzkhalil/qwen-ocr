# Urdu OCR Fine-tuning Pipeline

Fine-tunes `Qwen3.5-0.8B` via Unsloth + TRL on ~1.3 M rows of Urdu/Persian-Arabic OCR data, then evaluates against industry-standard metrics.

## Architecture

```
prepare_dataset.py  →  sharded Parquet  →  train.py  →  LoRA weights
                                                      →  merged 16-bit
                                                      →  GGUF (q8 / q4_k_m)
                    benchmark.py reads the same shards + model checkpoint
```

### Datasets used

| Dataset | Rows | Language | Augmentation |
|---|---|---|---|
| `mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Medium` | ~791 k | Persian/Arabic | light |
| `PuristanLabs1/urdu-ocr-1M` subset `nastaliq` | ~500 k | Urdu synthetic | heavy |
| `oddadmix/qaari-0.1-ocr-urdu-news-dataset-small` | ~36 k | Urdu real scans | medium |

---

## Setup

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    trl>=0.12.0 datasets>=2.19.0 transformers>=4.45.0 accelerate>=0.34.0 \
    bitsandbytes>=0.43.0 pillow>=10.0.0 numpy>=1.24.0 jiwer>=3.0.0 \
    tqdm>=4.66.0 huggingface_hub>=0.23.0 nltk>=3.8.0 requests
```

---

## Step 1 — Configure environment

```bash
cp .env.example .env
# Open .env and fill in HF_TOKEN plus any paths you want to override
source .env          # bash/zsh
# Windows PowerShell: Get-Content .env | ForEach-Object { if ($_ -match '=') { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k,$v) } }
```

---

## Step 2 — Test dataset access (no download, just peek)

```bash
python prepare_dataset.py --mode test
```

Streams 3 rows from each dataset and exits. Verifies credentials and column schemas before committing to a multi-hour download.

---

## Step 3 — Prepare datasets (download + augment + shard)

```bash
python prepare_dataset.py --mode all
```

- Downloads all three datasets (nastaliq via streaming)
- Applies augmentation baked into each Parquet shard
- Saves `$LOCAL_DATASET_DIR/train/*.parquet` and `$LOCAL_DATASET_DIR/validation/*.parquet`
- Prints a full statistics report on completion

Expected runtime: 2–4 hours on a fast connection  
Expected disk usage: ~15–25 GB

To cap rows per dataset during development:
```bash
MAX_SAMPLES_PER_DS=5000 python prepare_dataset.py --mode all
```

---

## Step 4 — Train

```bash
python train.py --mode train
```

Expected runtime: 6–12 hours on a single A100 80 GB for `MAX_STEPS=5000`

To run a quick smoke test (100 steps):
```bash
MAX_STEPS=100 TRAIN_SAMPLES=2000 VAL_SAMPLES=200 python train.py --mode train
```

---

## Step 5 — Benchmark

```bash
python benchmark.py \
  --model_path $MODEL_SAVE_DIR \
  --dataset_dir $LOCAL_DATASET_DIR \
  --split validation \
  --n_samples 500 \
  --output_file benchmark_results.json
```

Prints a formatted table with CER, WER, character accuracy, exact match, BLEU-4, and normalised edit distance — overall and per source. Also saves a JSON with per-sample predictions.

### Target metrics

| Metric | Minimum acceptable | Production target |
|---|---|---|
| CER (Urdu news) | < 15% | < 5% |
| CER (Nastaliq synthetic) | < 8% | < 3% |
| WER (Urdu news) | < 30% | < 12% |
| Exact Match | > 20% | > 55% |

If CER on synthetic is < 5% but CER on real news is > 20%, the model has overfit. Fix: oversample the qaari news dataset 5–10x or add extra fine-tuning steps on real data only.

---

## Step 6 — Inference test

```bash
IMAGE_PATH=/path/to/urdu_image.jpg python train.py --mode infer
```

---

## Step 7 — Export to GGUF (for Ollama / llama.cpp)

```bash
python train.py --mode export
```

Saves:
- `$OUTPUT_DIR/merged_16bit/` — full merged weights
- `$OUTPUT_DIR/gguf_q8/` — Q8_0 quantisation
- `$OUTPUT_DIR/gguf_q4_k_m/` — Q4_K_M quantisation

---

## Dataset statistics (after prepare)

```bash
python prepare_dataset.py --mode stats
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face access token |
| `LOCAL_DATASET_DIR` | `/workspace/hf_dataset` | Where sharded Parquet is stored |
| `OUTPUT_DIR` | `/workspace/outputs` | Training outputs root |
| `MODEL_SAVE_DIR` | `/workspace/outputs/qwen_lora` | LoRA checkpoint path |
| `BASE_MODEL` | `unsloth/Qwen3.5-0.8B` | Base model |
| `LOAD_IN_4BIT` | `false` | Load model in 4-bit (saves VRAM) |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha |
| `LORA_DROPOUT` | `0.0` | LoRA dropout |
| `PER_DEVICE_BATCH_SIZE` | `4` | Train batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Effective batch = 4 × 8 = 32 |
| `WARMUP_STEPS` | `100` | Linear warmup steps |
| `MAX_STEPS` | `5000` | Total training steps (-1 → use epochs) |
| `NUM_TRAIN_EPOCHS` | `1` | Epochs (used when MAX_STEPS = -1) |
| `LEARNING_RATE` | `2e-4` | Peak learning rate |
| `LR_SCHEDULER` | `cosine` | Scheduler type |
| `MAX_LENGTH` | `1024` | Max token length |
| `SHARD_SIZE` | `1000` | Rows per Parquet shard |
| `MAX_WORKERS` | `16` | Parallel image download threads |
| `MAX_SAMPLES_PER_DS` | `-1` | Cap rows per dataset (-1 = all) |
| `TRAIN_SAMPLES` | `-1` | Cap train rows fed to trainer |
| `VAL_SAMPLES` | `-1` | Cap val rows fed to trainer |
| `SAVE_MERGED_16BIT` | `true` | Export merged full-precision model |
| `SAVE_GGUF_Q8` | `true` | Export GGUF Q8_0 |
| `SAVE_GGUF_Q4_K_M` | `true` | Export GGUF Q4_K_M |
| `OCR_PROMPT` | Urdu instruction | Instruction sent to model per image |
| `FINETUNE_VISION_LAYERS` | `true` | Apply LoRA to vision encoder |
| `FINETUNE_LANGUAGE_LAYERS` | `true` | Apply LoRA to language decoder |
| `FINETUNE_ATTENTION_MODULES` | `true` | LoRA on attention layers |
| `FINETUNE_MLP_MODULES` | `true` | LoRA on MLP layers |

---

## File structure

```
├── train.py               — fine-tuning script (Unsloth + TRL SFTTrainer)
├── prepare_dataset.py     — download, normalise, augment, shard all datasets
├── augment.py             — image augmentation module (Pillow + NumPy only)
├── benchmark.py           — evaluation: CER, WER, BLEU-4, exact match, NED
├── .env.example           — all environment variables documented
└── README.md              — this file
```
