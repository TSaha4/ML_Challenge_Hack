#!/usr/bin/env bash
# French normalisation fix: re-normalise test, rebuild test candidates, predict with v2 (A/B) and v5.
cd "$(dirname "$0")/.."
L=artifacts/er/logs
export ER_MIN_FREE_GB=2 ER_WORKERS=6 POLARS_MAX_THREADS=10 PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u -m src.er.steps.prepare --split test > $L/fr_prepare.log 2>&1 &&
python -u -X faulthandler -m src.er.steps.block --split test --cap 500 > $L/fr_block_test.log 2>&1 &&
rm -rf artifacts/er/test_scored &&
python -u -X faulthandler -m src.er.steps.predict --model-dir artifacts/er_v2/model --threshold 0.70 --out-dir output_v2fr > $L/fr_v2_predict.log 2>&1 &&
rm -rf artifacts/er/test_scored &&
python -u -X faulthandler -m src.er.steps.predict --model-dir artifacts/er/model_v5 --nn artifacts/er/nn/matcher_v5.pt --threshold 0.70 --out-dir output_v5fr > $L/fr_v5_predict.log 2>&1
echo "frfix exit $?"
