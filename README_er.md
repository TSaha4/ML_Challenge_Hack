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
pip install -r requirements-er.txt
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
python -m src.er.steps.nn                      # char-level neural matcher on a reserved record fold (~12 min)
python -m src.er.steps.train --n-train 800000 --max-bin 128 --nn artifacts/er/nn/matcher.pt --model-dir artifacts/er/model_v4
python -m src.er.steps.predict --model-dir artifacts/er/model_v4 --nn artifacts/er/nn/matcher.pt --validator <path>/validate_submission.py
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
| `src/er/siblings.py` | collective evidence from records competing for the same Source-1 entity |
| `src/er/nn.py` | character-level decomposable-attention neural matcher (stacked as `nn_p`) |
| `src/er/model.py` | XGBoost params, assignment, official macro-F0.5, threshold tuning |
| `src/er/resources.py` | low priority + free-RAM guard |
| `src/er/steps/` | the four entry points above |

No external data, APIs or services are used; only the provided competition files are read.

## Corrected pipeline on codex/er-v2-quality-fixes

The previously reported 0.9844 local validation and 0.968 user-reported leaderboard
scores belong to earlier runs. The changes below have no measured challenge score
until retraining and evaluation. A score above 0.999 is not guaranteed.

The character n-gram intersection now counts repeated grams correctly; the old
presence-only calculation could report a perfect match for different strings.
This changes both candidate reranking and model inputs. **Rebuild preparation,
blocking, neural weights and XGBoost together. Old models are rejected.** Keep the
original successful model and submission as your baseline.

Other corrections:

- Prediction caches are isolated by the contents of the model, neural weights,
  candidates, records, statistics, code and dependency versions. Changing only the
  threshold can reuse the same scores. Partial score files are written atomically.
- Reusing training features requires a matching completed cache manifest.
- Missing neural weights and mismatched checkpoints fail before inference.
- Calibration (`s1 % 100 == 0`) and independent holdout (`s1 % 100 == 4`) are separate.
  Their candidate neighborhoods and true targets are excluded from both supervised
  matchers. Both folds are also excluded from learned transliteration dictionaries.
- Threshold selection searches all distinct winning-score boundaries, including
  predicting no matches, and resolves assignment ties deterministically by S1 ID.
- The holdout report includes country slices, the candidate-set oracle ceiling and
  per-entity blocking misses, matcher misses and false positives. Repeatedly selecting
  experiments on this holdout would turn it into another tuning set; keep decisions
  on calibration and reserve holdout reporting for the final candidate.
- Exports preserve original ID strings, including leading zeros. The official
  validator checks ID membership and failures now return a failing process status.
- Threshold reports are created inside each model directory; no pre-existing logs
  directory is needed.

### Fresh Linux CUDA run

Use Python 3.11 with an NVIDIA GPU and CUDA-compatible driver. The pinned environment
is in `requirements-er.txt` (the root `requirements.txt` is for the legacy pipeline).
This branch must be transferred to the remote machine; uncommitted local edits are
not included when cloning the GitHub er-v2 branch.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-er.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export ER_DATA_DIR=/workspace/student_resource/dataset
export ER_ARTIFACT_DIR=/workspace/er-fixed-artifacts

python -m src.er.steps.prepare --split train
python -m src.er.steps.prepare --split test
python -m src.er.steps.block --split train
python -m src.er.steps.block --split test
python -m src.er.steps.nn --out "$ER_ARTIFACT_DIR/nn/matcher.pt"
python -m src.er.steps.train --n-train 800000 --max-bin 128 \
  --nn "$ER_ARTIFACT_DIR/nn/matcher.pt" \
  --model-dir "$ER_ARTIFACT_DIR/model_fixed"
python -m src.er.steps.predict \
  --model-dir "$ER_ARTIFACT_DIR/model_fixed" \
  --nn "$ER_ARTIFACT_DIR/nn/matcher.pt" \
  --out-dir output \
  --validator /workspace/student_resource/utils/validate_submission.py
```

Use the saved calibrated threshold by default; do not automatically override it
with the previous run's 0.75. Inspect `model_fixed/meta.json`,
`model_fixed/threshold_curve.json` and `model_fixed/holdout_errors.parquet` before
uploading `output/matching_results.tsv`. An explicit `--skip-validation` exists
for experiments; run the official validator before any portal submission.

To examine the next accuracy bottleneck, compare the **macro** candidate-set ceiling
with the holdout score. Pair recall alone is not the macro score ceiling. If retrieval
is limiting, test a new blocking run with a larger `--k`, `--k-wide` or `--k-char` in
a separate artifact directory, then retrain and compare on calibration. Increasing
candidate counts also increases runtime and false-positive opportunities.

### Where to run

A Linux NVIDIA cloud GPU or a teammate's CUDA desktop can run the pipeline while
you connect from the Mac. For headroom, aim for 16–24 GB VRAM, 32 GB RAM and a
persistent disk sized for the dataset plus several generations of candidates,
features and scores. These are planning recommendations, not measured minimums
for the corrected version. GPU type, runtime limits and cost depend on the provider.
Runpod supports GPU pods and persistent storage; Colab provides GPU notebook
runtimes with variable availability and session limits. Back up artifacts and
outputs before stopping a temporary runtime.

References: [Runpod setup](https://www.runpod.io/blog/configuring-runpod),
[Colab limitations](https://research.google.com/colaboratory/faq.html).

### Regression checks

```bash
CUDA_VISIBLE_DEVICES="" ER_MIN_FREE_GB=0 python -m unittest discover -s tests/er -v
```

Tests use tiny synthetic fixtures and a small CPU XGBoost fit. They verify pipeline
correctness, cache invalidation, threshold optimization and submission behavior;
they do not establish competition accuracy or full-scale CUDA performance.

### Check fitting and track the exact submission

Training now saves `learning_curves.json` and `fit_diagnostics` in `meta.json`.
Compare train and early-stop loss trajectories alongside entity holdout macro F0.5;
a low pairwise loss alone does not prove robustness to unseen countries.
For a controlled regularization comparison, keep the same data and candidate
artifacts and vary `--max-depth`, `--min-child-weight`, and `--reg-lambda`. Defaults
remain unchanged pending full-scale evidence. `--rounds`, `--early-stopping-rounds`
and `--early-stop-metric` are explicit experiment controls.

Prediction now writes `submission_manifest.json` beside the TSV files, recording
model and output content identities, the selected threshold, any threshold override,
blocking settings and validator status. Keep this manifest with each portal upload
and its returned score. Train and test blocking settings must match; prediction
rejects mismatches such as train cap=1000 versus test cap=500.

Read `docs/dataset_generalization_audit.md` for the actual dataset and v4/v5 log audit.
