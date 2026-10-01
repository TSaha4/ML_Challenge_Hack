#!/usr/bin/env bash
# Full from-scratch run of the corrected ER v2 pipeline (README_er.md recipe) on the local GPU.
# Every step logs to $ER_ARTIFACT_DIR/logs/NN_<step>.log; the chain stops at the first failure.
set -o pipefail
cd "$(dirname "$0")/.."
export ER_DATA_DIR=student_resource/dataset
export ER_ARTIFACT_DIR=artifacts/er_fixed
export ER_MIN_FREE_GB=2 ER_WORKERS=4 POLARS_MAX_THREADS=8
export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$ER_ARTIFACT_DIR/logs; mkdir -p $L
step() { local name=$1; shift; echo "[$(date '+%H:%M:%S')] START $name"
         if python -u -X faulthandler "$@" > $L/$name.log 2>&1; then echo "[$(date '+%H:%M:%S')] OK    $name"
         else echo "[$(date '+%H:%M:%S')] FAIL  $name (see $L/$name.log)"; exit 1; fi; }
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
step 01_prepare_train -m src.er.steps.prepare --split train
step 02_prepare_test  -m src.er.steps.prepare --split test
step 03_block_train   -m src.er.steps.block --split train
step 04_block_test    -m src.er.steps.block --split test
step 05_nn            -m src.er.steps.nn --out $ER_ARTIFACT_DIR/nn/matcher.pt
step 06_train         -m src.er.steps.train --n-train 800000 --max-bin 128 --nn $ER_ARTIFACT_DIR/nn/matcher.pt --model-dir $ER_ARTIFACT_DIR/model_fixed
step 07_predict       -m src.er.steps.predict --model-dir $ER_ARTIFACT_DIR/model_fixed --nn $ER_ARTIFACT_DIR/nn/matcher.pt --out-dir output_fixed --validator student_resource/utils/validate_submission.py
echo "[$(date '+%H:%M:%S')] PIPELINE COMPLETE"
