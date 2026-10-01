#!/usr/bin/env bash
# Round 6: wider blocking (cap 1000) -> siblings -> XGBoost with v5 NN -> validation.
# Waits for the running v5 test prediction (it reads the current candidate files).
cd "$(dirname "$0")/.."
L=artifacts/er/logs
until grep -q "validator exit code\|Traceback" $L/v5_predict.log; do sleep 30; done
export ER_MIN_FREE_GB=2 ER_WORKERS=6 POLARS_MAX_THREADS=12 PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -X faulthandler -m src.er.steps.block --split train --cap 1000 > $L/v6_block_train.log 2>&1 &&
python -u -X faulthandler -m src.er.steps.block --split test --cap 1000 > $L/v6_block_test.log 2>&1 &&
python -u -X faulthandler -m src.er.steps.train --n-train 800000 --max-bin 128 --nn artifacts/er/nn/matcher_v5.pt --model-dir artifacts/er/model_v6 > $L/v6_train.log 2>&1
echo "round6 exit $?"
