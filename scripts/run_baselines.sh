#!/usr/bin/env bash
# Run both baselines on the balanced training split.
#
# Baseline 1 — Prompt Engineering: zero-shot warning / persona prefixes
# Baseline 2 — Vanilla Negation SFT: uniform ranking loss, same data as MGNM
#
# Usage:
#   bash scripts/run_baselines.sh
#
# Outputs:
#   outputs/eval_prompt_baseline_baltest.json
#   outputs/e4_qwen_vanilla_3ep/   (adapter + training_summary)
#   outputs/e4_train_vanilla_3ep.log

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_DATA="data/processed/splits/balanced/train.jsonl"
EVAL_DATA="data/processed/splits/balanced/dev.jsonl"
TEST_DATA="data/processed/splits/balanced/test.jsonl"
MODELS="configs/model_config.json"

echo "=== Baseline 1: Prompt Engineering ==="
python scripts/evaluate_prompt_baseline.py \
    --models       "$MODELS" \
    --model-name   qwen2_5_7b \
    --input        "$TEST_DATA" \
    --output       outputs/eval_prompt_baseline_baltest.json \
    --cache-dir    outputs/score_cache_prompt_baseline \
    --styles       none warning persona

echo ""
echo "=== Baseline 2: Vanilla Negation SFT (3 epochs, balanced data) ==="
python scripts/train_vanilla_sft.py \
    --models           "$MODELS" \
    --model-name       qwen2_5_7b \
    --train-input      "$TRAIN_DATA" \
    --eval-input       "$EVAL_DATA" \
    --output-dir       outputs/e4_qwen_vanilla_3ep \
    --epochs           3 \
    --learning-rate    2e-4 \
    --grad-accum-steps 8 \
    --margin           0.5 \
    --lambda-pos       1.0 \
    --lambda-rank      1.0 \
    --lambda-ret       0.05 \
    2>&1 | tee outputs/e4_train_vanilla_3ep.log

echo ""
echo "=== Evaluate Vanilla SFT on balanced test ==="
# Build a temp model config pointing at the vanilla adapter
python3 -c "
import json, pathlib
cfg = json.load(open('outputs/e4_qwen_vanilla_3ep/model_config_with_adapter.json'))
json.dump(cfg, open('/tmp/model_config_vanilla.json','w'), indent=2)
print('wrote /tmp/model_config_vanilla.json')
"
python scripts/evaluate_models.py \
    --models     /tmp/model_config_vanilla.json \
    --input      "$TEST_DATA" \
    --output     outputs/eval_vanilla_3ep_baltest.json \
    --cache-dir  outputs/score_cache_vanilla_baltest \
    --model-names qwen2_5_7b_vanilla

echo ""
echo "=== Done ==="
echo "Prompt baseline: outputs/eval_prompt_baseline_baltest.json"
echo "Vanilla SFT:     outputs/eval_vanilla_3ep_baltest.json"
