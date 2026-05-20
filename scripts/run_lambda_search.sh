#!/usr/bin/env bash
# Train two MGNM variants with higher lambda_sup, then evaluate on large_test.
#
# Variant 1 (lsup2): lambda_sup=2.0, others unchanged  → GPU 0
# Variant 2 (lsup3): lambda_sup=3.0, others unchanged  → GPU 1
#
# After training, evaluate both on large_test and print comparison.
#
# Usage:
#   bash scripts/run_lambda_search.sh 2>&1 | tee outputs/lambda_search.log

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_DATA="data/processed/splits/balanced/train.jsonl"
EVAL_DATA="data/processed/splits/balanced/dev.jsonl"
TEST_DATA="data/processed/splits/large_test/test.jsonl"
MODELS="configs/model_config.json"

echo "=== Training MGNM lsup2 on GPU 0 / lsup3 on GPU 1 in parallel ==="

CUDA_VISIBLE_DEVICES=0 python scripts/train_mgnm.py \
    --models           "$MODELS" \
    --model-name       qwen2_5_7b \
    --train-input      "$TRAIN_DATA" \
    --eval-input       "$EVAL_DATA" \
    --output-dir       outputs/e4_qwen_lsup2 \
    --epochs           3 \
    --learning-rate    2e-4 \
    --grad-accum-steps 8 \
    --lambda-pos       1.0 \
    --lambda-sup       2.0 \
    --lambda-rank      1.0 \
    --lambda-preserve  1.0 \
    --lambda-ret       0.05 \
    --margin-pos       0.5 \
    --margin-sup       0.5 \
    --margin-rank      0.5 \
    --margin-preserve  0.5 \
    2>&1 | tee outputs/train_lsup2.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python scripts/train_mgnm.py \
    --models           "$MODELS" \
    --model-name       qwen2_5_7b \
    --train-input      "$TRAIN_DATA" \
    --eval-input       "$EVAL_DATA" \
    --output-dir       outputs/e4_qwen_lsup3 \
    --epochs           3 \
    --learning-rate    2e-4 \
    --grad-accum-steps 8 \
    --lambda-pos       1.0 \
    --lambda-sup       3.0 \
    --lambda-rank      1.0 \
    --lambda-preserve  1.0 \
    --lambda-ret       0.05 \
    --margin-pos       0.5 \
    --margin-sup       0.5 \
    --margin-rank      0.5 \
    --margin-preserve  0.5 \
    2>&1 | tee outputs/train_lsup3.log &
PID2=$!

echo "Training PID1=$PID1 (lsup2, GPU 0), PID2=$PID2 (lsup3, GPU 1)"
wait $PID1 && echo "lsup2 training complete" || echo "lsup2 training FAILED (exit $?)"
wait $PID2 && echo "lsup3 training complete" || echo "lsup3 training FAILED (exit $?)"

echo ""
echo "=== Evaluate lsup2 on large_test (GPU 0) ==="
python3 -c "
import json
cfg = json.load(open('outputs/e4_qwen_lsup2/model_config_with_adapter.json'))
json.dump(cfg, open('/tmp/model_cfg_lsup2.json','w'), indent=2)
print('wrote /tmp/model_cfg_lsup2.json, models:', [m['name'] for m in cfg['models']])
"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_models.py \
    --models     /tmp/model_cfg_lsup2.json \
    --input      "$TEST_DATA" \
    --output     outputs/eval_lsup2_largetest.json \
    --cache-dir  outputs/score_cache_lsup2_largetest \
    --model-names qwen2_5_7b_e4 \
    2>&1 | tee outputs/eval_lsup2_largetest.log

echo ""
echo "=== Evaluate lsup3 on large_test (GPU 0) ==="
python3 -c "
import json
cfg = json.load(open('outputs/e4_qwen_lsup3/model_config_with_adapter.json'))
json.dump(cfg, open('/tmp/model_cfg_lsup3.json','w'), indent=2)
print('wrote /tmp/model_cfg_lsup3.json, models:', [m['name'] for m in cfg['models']])
"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_models.py \
    --models     /tmp/model_cfg_lsup3.json \
    --input      "$TEST_DATA" \
    --output     outputs/eval_lsup3_largetest.json \
    --cache-dir  outputs/score_cache_lsup3_largetest \
    --model-names qwen2_5_7b_e4 \
    2>&1 | tee outputs/eval_lsup3_largetest.log

echo ""
echo "=== Results Comparison ==="
python3 - <<'PYEOF'
import json, os

def fmt(path, label):
    if not os.path.exists(path):
        print(f"{label}: file not found")
        return
    d = json.load(open(path))
    key = list(d.keys())[-1]
    agg = d[key].get("aggregate", d[key])
    s = agg.get("summary", agg)
    def g(k): return s.get(k, {}).get("mean", 0) * 100
    print(f"{label:<20} PosAcc={g('PosAcc'):.1f}%  NegSuppRate={g('NegSuppRate'):.1f}%  "
          f"FlipAcc={g('FlipAcc'):.1f}%  ScopeCtrl={g('ScopeControlAcc'):.1f}%  "
          f"DoubleNeg={g('DoubleNegationAcc'):.1f}%  OverNeg={g('OverNegationRate'):.1f}%")

fmt("outputs/eval_bal_full_3ep_largetest.json", "MGNM bal_full_3ep")
fmt("outputs/eval_vanilla_3ep_largetest.json",  "Vanilla SFT 3ep")
fmt("outputs/eval_lsup2_largetest.json",         "MGNM lsup2 (λs=2)")
fmt("outputs/eval_lsup3_largetest.json",         "MGNM lsup3 (λs=3)")
PYEOF

echo ""
echo "Done. Check outputs/lambda_search.log for full output."
