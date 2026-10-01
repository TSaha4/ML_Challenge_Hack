# ML Challenge 2026 — Business Entity Resolution

Reference **Source 1** business entities are matched against records from **Source 2** and
**Source 3** for the ML Challenge 2026 entity-resolution task. Scoring uses **Macro F_0.5**
(precision weighted 2×, singletons included). This repository contains the complete pipeline:
multi-representation normalization, multi-channel adaptive blocking, rare-token IDF weighting,
pairwise feature engineering, LightGBM matching with F_0.5 threshold tuning, and memory-safe
full-test inference with submission export.

## ⭐ Current submission pipeline: GPU v2 (`src/er/`)

**Correction branch:** see the [fresh retraining guide](README_er.md#corrected-pipeline-on-codexer-v2-quality-fixes). Character similarity and validation have changed; rebuild all artifacts in a new directory. Scores below describe historical runs, not the corrected code.

The submission is produced by the **GPU v2 pipeline** in `src/er/` (branch `er-v2`).
It replaces the prototype DuckDB/LightGBM path described further below, which is kept
for reference only. Full details: [`README_er.md`](README_er.md) (how to run) and
[`Documentation_template.md`](Documentation_template.md) (methodology and results).

| | Status |
|---|---|
| Validation macro F0.5 (22,224 held-out Source-1 entities, singletons included) | **0.9844** (pair precision 99.7 %, recall 96.3 %) |
| Candidate recall on all 7.64M training pairs | **98.1 %** (~13 candidates per record) |
| Full test run (1.73M Source-1, 9.97M records) | done, `output/*.tsv` pass the official validator incl. `--check-ids` |
| Final zip | `python scripts/er_package.py --team <TEAM>` → `dist/<TEAM>_submission.zip` |

Key idea: every Source-2/3 record belongs to **at most one** Source-1 entity, so each record
is assigned to its most likely Source-1 entity when an XGBoost (CUDA) model is confident
enough. Blocking, re-ranking, features and training all run on the GPU (RTX 3060, 6 GB).

```bash
pip install -r requirements-er.txt
python -m src.er.steps.prepare --split train && python -m src.er.steps.prepare --split test
python -m src.er.steps.block --split train   && python -m src.er.steps.block --split test
python -m src.er.steps.nn                      # char-level neural matcher (GPU)
python -m src.er.steps.train --n-train 800000 --max-bin 128 --nn artifacts/er/nn/matcher.pt --model-dir artifacts/er/model_v4
python -m src.er.steps.predict --model-dir artifacts/er/model_v4 --nn artifacts/er/nn/matcher.pt
```

Outputs (`output/`, `artifacts/`, `dist/`) are git-ignored; regenerate them with the
commands above (~1 hour on a laptop GPU).

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/entity_resolution.ipynb` | The 23-phase solution notebook (Phase 0–22), executed end to end. |
| `src/normalization.py` | Unicode/legal-suffix normalization, transliteration, address + postal-code parsing. |
| `src/blocking.py` | Multi-channel blocking keys and the adaptive inverted-index blocker. |
| `src/features.py` | 27 pairwise similarity / token-overlap / IDF comparison features. |
| `src/metrics.py` | Macro F_0.5 and per-entity F_β, matching the official competition formula. |
| `src/pipeline.py` | Chunked TSV streaming, corpus IDF computation, submission export. |
| `src/inference.py` | Memory-safe test-inference engine, model-bundle persistence, CLI entry point. |
| `scripts/` | Notebook builder/executor and the end-to-end audit harness. |
| `student_resource/` | Challenge statement, dataset and the official submission validator. |

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Developed on Python 3.12 with polars 1.44, LightGBM 4.7, DuckDB 1.5 and rapidfuzz.

## Reproducing the results

### 1. Build and run the notebook (analysis + training + validation)

```bash
python scripts/build_notebook.py      # regenerate notebooks/entity_resolution.ipynb from scripts/sec*.py
python scripts/execute_notebook.py    # execute every cell and store the outputs
```

`RUN_EXPENSIVE = True` in notebook Phase 0 enables exact full-file row counting; the default
`False` keeps every cell fast by sampling.

### 2. Persist the trained model bundle

Notebook Phase 14 writes `artifacts/models/lgb_f05/` containing `model.joblib` plus
`metadata.json` (ordered feature list, corpus IDF weights, F_0.5-optimal threshold).

### 3. Generate the submission files for the full test set

```bash
python -m src.inference \
    --model-dir artifacts/models/lgb_f05 \
    --output-dir output \
    --validator student_resource/utils/validate_submission.py
```

The engine indexes Source 2/3 into an on-disk DuckDB store, streams Source 1 in bounded
batches, scores candidates with the LightGBM model, and writes `output/matching_results.tsv`
and `output/candidate_pairs.tsv` incrementally so peak RAM stays bounded on ~10M targets.
`--help` lists all tunables (`--chunk-size`, `--s1-chunk-size`, `--max-cands`, `--threshold`).

### 4. Validate before uploading

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

### 5. Fast end-to-end audit (recommended before the full run)

```bash
python scripts/validate_inference_mini.py
```

Slices the real test files, trains a throwaway model on real training ground truth, runs the
production inference engine on the slice, then asserts every submission rule (coverage,
ordering, S2-/S3- prefixes, no duplicates, matches ⊆ candidates) and runs the official
validator (`PASS`, exit 0).

## Submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv          # produced by step 3 (scored on the leaderboard)
│   └── candidate_pairs.tsv           # last candidate set fed to the model (blocking audit)
├── code/business_entity_resolution/
│   ├── src/                          # this repository's src/ modules
│   ├── README.md                     # this file
│   └── requirements.txt
└── Documentation_template.md         # filled-in methodology write-up
```

## Status and known gaps of the original pipeline (superseded)

> These gaps apply to the original DuckDB/LightGBM pipeline only. They are resolved in the GPU
> v2 pipeline above: the 5M+ scale crash is avoided with GPU blocking and streamed data, the
> model is trained on real candidates from the full training set, the threshold is tuned on
> held-out entities, and the full test submission is generated and validated.

* **Ready:** normalization, blocking, features, metrics, the memory-safe inference engine, the
  `output/` format and the official validator round-trip are implemented and verified
  (mini audit + CLI run both end with validator `PASS`, exit 0).
* **Ready:** every notebook cell executes (35/35 code cells, Phases 0–22) and Phase 14/20/21
  persist the model bundle and produce/validate the submission files.
* **Gap — model scale:** notebook Phases 12–14 currently train the shipped LightGBM model on a
  *prototype* feature matrix (a few thousand synthesised pairs). Scale it up before submitting:
  build candidates for the full train split with the real blocker, add mined hard negatives
  (Phase 15), keep the entity-aware split (Phase 13), then re-tune the threshold and re-run
  Phase 14 so `artifacts/models/lgb_f05` holds the full model.
* **Gap — ablation table:** the Phase 22 figures are hand-entered planning targets, *not*
  measured scores; replace them with measured validation numbers before citing them.
* **Note:** `candidate_pairs.tsv` contains exactly the candidate set scored by the model, as the
  challenge requires (not an intermediate blocking pass).
* **Note:** no external data/services are used anywhere — normalization is purely local
  (Unidecode + RapidFuzz + regex); only the provided competition files are read.
