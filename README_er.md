# Business Entity Resolution — GPU pipeline (v2)

Matches every Source-1 (reference) business to its Source-2/3 records.
Target-centric: every Source-2/3 record belongs to **at most one** Source-1 entity
(verified on all 7.64M training pairs), so each record is assigned to its most
likely Source-1 entity when the model's probability clears a macro-F0.5-tuned
threshold. A Source-1 entity collects all records assigned to it; one with none
is a singleton (empty list).

```
raw TSV ─► normalise (polars) ─► GPU rare-key blocking + char-gram re-rank ─►
GPU pair features ─► XGBoost (CUDA) ─► per-record argmax + threshold ─► TSVs
```

## Requirements

* Python 3.11, an NVIDIA GPU with ≥ 6 GB VRAM (tested on an RTX 3060 Laptop, CUDA 12.4)
* ~16 GB system RAM (the pipeline itself stays at ~2–5 GB), ~10 GB free disk

```bash
pip install -r requirements.txt
```

## Data layout

Point `ER_DATA_DIR` at the folder holding `train/` and `test/` (default
`student_resource/dataset`):

```
<ER_DATA_DIR>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<ER_DATA_DIR>/test/test_source{1,2,3}.tsv
```

## Reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`

Run from this folder (the one containing `src/`). Intermediate files go to `artifacts/er/`.

```bash
python -m src.er.steps.prepare --split train   # learn transliteration maps, normalise (~1 min)
python -m src.er.steps.prepare --split test
python -m src.er.steps.block --split train     # GPU candidate generation (~15 min)
python -m src.er.steps.block --split test
python -m src.er.steps.train --n-train 1200000 --train-files 8 --max-bin 128   # GPU features + XGBoost, validation, threshold
python -m src.er.steps.predict --validator <path>/validate_submission.py      # score test, write output/, validate
```

Resource knobs (environment variables): `POLARS_MAX_THREADS` (CPU threads, default all),
`ER_WORKERS` (rapidfuzz threads, default 2), `ER_MIN_FREE_GB` (the job pauses while
free RAM is below this, default 2.5). All jobs run at below-normal priority.

## Source layout

| Path | Role |
|---|---|
| `src/er/io.py` | TSV loading, compact integer ids (`S2-123` ↔ `2·10¹⁰+123`) |
| `src/er/text.py` | vectorised name/address normalisation (aliases, domains, honorifics, leet typos, abbreviations, states) |
| `src/er/translit.py` | Indic-script → Latin token dictionary learned from training pairs |
| `src/er/blocking.py` | blocking-key generation (name tokens, squashed name, address tokens/bigrams) |
| `src/er/gpu_blocking.py` | CUDA inverted-index lookup, IDF scoring, top-k, char-gram re-rank |
| `src/er/features.py` | per-record / per-entity candidate context statistics |
| `src/er/gpu_features.py` | CUDA char n-gram and hashed-token pair features (+3 rapidfuzz scores) |
| `src/er/model.py` | XGBoost params, assignment, official macro-F0.5, threshold tuning |
| `src/er/resources.py` | low priority + free-RAM guard |
| `src/er/steps/` | the four entry points above |

No external data, APIs or services are used; only the provided competition files are read.
