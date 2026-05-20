#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${1:-0}"
TRAIN_DATA="data/processed/splits/router_targeted_retraining_pack/train_clean_strict_plus_routerfix19.jsonl"
EVAL_DATA="data/processed/splits/router_calibration/calibration_clean_strict.jsonl"
OUT_DIR="outputs/routerfix_qwen_targeted_phase1"

echo "======================================================"
echo " run_router_targeted_phase1.sh"
echo " GPU=${GPU}"
echo " TRAIN=${TRAIN_DATA}"
echo " EVAL=${EVAL_DATA}"
echo " OUT=${OUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/train_mgnm.py \
  --models configs/model_config.json \
  --model-name qwen2_5_7b \
  --train-input "${TRAIN_DATA}" \
  --eval-input "${EVAL_DATA}" \
  --output-dir "${OUT_DIR}" \
  --epochs 3 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lambda-preserve 1.5 \
  --lambda-rank 1.5 \
  --lambda-rank-select 3.0 \
  --lambda-ret 0.05 \
  --behavior-token \
  --scope-oversample 1 \
  --dn-oversample 2
