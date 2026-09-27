#!/usr/bin/env bash
# Hard-mode training: 18.6% of train S1 entities hidden (their records become look-alike noise),
# matching the test set's density (5.74 vs 5.75 records per S1). Uses only training data.
set -o pipefail
cd "$(dirname "$0")/.."
export ER_DATA_DIR=student_resource_sim/dataset ER_ARTIFACT_DIR=artifacts/er_sim
export ER_MIN_FREE_GB=2 ER_WORKERS=4 POLARS_MAX_THREADS=8
export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$ER_ARTIFACT_DIR/logs; mkdir -p $L
step() { local name=$1; shift; echo "[$(date '+%H:%M:%S')] START $name"
         if python -u -X faulthandler "$@" > $L/$name.log 2>&1; then echo "[$(date '+%H:%M:%S')] OK    $name"
         else echo "[$(date '+%H:%M:%S')] FAIL  $name"; exit 1; fi; }
step 01_prepare_train -m src.er.steps.prepare --split train
step 03_block_train   -m src.er.steps.block --split train
step 06_train         -m src.er.steps.train --n-train 800000 --max-bin 128 --nn $ER_ARTIFACT_DIR/nn/matcher.pt --model-dir $ER_ARTIFACT_DIR/model_hard
echo "[$(date '+%H:%M:%S')] TRAINING COMPLETE"
