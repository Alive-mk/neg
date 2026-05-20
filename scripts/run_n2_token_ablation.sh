#!/usr/bin/env bash
# N2: Behavior-token ablation
#   1. Train Qwen MGNM without --behavior-token (no-token variant)
#   2. Eval oracle token model (auto prefix)      → already done: eval_qwen_mgnm_v2.json
#   3. Eval oracle model with no prefix           → tests: does oracle model degrade without tokens?
#   4. Eval oracle model with wrong token         → [SUPPRESS] everywhere (wrong for preserve records)
#   5. Eval no-token model                        → clean no-token baseline
#   6. Print summary table
#
# Usage: bash scripts/run_n2_token_ablation.sh [TRAIN_GPU] [EVAL_GPU]
#   default: TRAIN_GPU=0, EVAL_GPU=0

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_GPU="${1:-0}"
EVAL_GPU="${2:-0}"

TRAIN_DATA="data/processed/splits/merged_excl_boost/train.jsonl"
EVAL_DATA="data/processed/validated_largetest_v2.jsonl"
DEV_DATA="data/processed/splits/balanced/dev.jsonl"
NOTOKEN_DIR="outputs/e4_qwen_notoken"
CFG_TMP="/tmp/model_cfg_notoken.json"

echo "======================================================"
echo " N2: Behavior-token ablation"
echo " TRAIN_GPU=cuda:${TRAIN_GPU}  EVAL_GPU=cuda:${EVAL_GPU}"
echo "======================================================"

# ── Step 1: Train no-token variant ───────────────────────
echo ""
echo "=== Step 1: Train Qwen MGNM (no behavior token) ==="
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python scripts/train_mgnm.py \
    --models          configs/model_config.json \
    --model-name      qwen2_5_7b \
    --train-input     "$TRAIN_DATA" \
    --eval-input      "$DEV_DATA" \
    --output-dir      "$NOTOKEN_DIR" \
    --epochs          3 \
    --learning-rate   2e-4 \
    --weight-decay    0.0 \
    --lambda-preserve 1.5 \
    --lambda-rank     1.5 \
    --lambda-rank-select 3.0 \
    --lambda-rank-dist   3.0 \
    --lambda-ret      0.05 \
    --scope-oversample 3 \
    2>&1 | tee outputs/train_qwen_notoken.log
# NOTE: no --behavior-token flag

echo "Training done."

# ── Step 2: Build eval config ─────────────────────────────
echo ""
echo "=== Step 2: Build eval config ==="
python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
base = next(m for m in cfg['models'] if m['name']=='qwen2_5_7b')

oracle = dict(base)
oracle['name'] = 'qwen_mgnm_oracle'
oracle['adapter_path'] = 'outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3/adapter'

notoken = dict(base)
notoken['name'] = 'qwen_mgnm_notoken'
notoken['adapter_path'] = '${NOTOKEN_DIR}/adapter'

json.dump({'models': cfg['models'] + [oracle, notoken]}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
"

# ── Step 3: Eval oracle model with no prefix ─────────────
echo ""
echo "=== Step 3: Oracle model + no prefix at inference ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_models.py \
    --models      "${CFG_TMP}" \
    --input       "$EVAL_DATA" \
    --output      outputs/eval_qwen_mgnm_no_prefix.json \
    --cache-dir   outputs/score_cache_no_prefix \
    --model-names qwen_mgnm_oracle \
    --neg-prefix  "" \
    2>&1 | tee outputs/eval_qwen_mgnm_no_prefix.log

# ── Step 4: Eval oracle model with wrong token ───────────
# Use [SUPPRESS] for ALL records:
#   - suppress_target records: CORRECT token (no change)
#   - preserve_positive records: WRONG token ([SUPPRESS] instead of [PRESERVE])
#   - select_gold_neg records: wrong token
# This tests what happens when caller provides wrong token for preserve/select records
echo ""
echo "=== Step 4: Oracle model + wrong token ([SUPPRESS] everywhere) ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_models.py \
    --models      "${CFG_TMP}" \
    --input       "$EVAL_DATA" \
    --output      outputs/eval_qwen_mgnm_wrong_token.json \
    --cache-dir   outputs/score_cache_wrong_token \
    --model-names qwen_mgnm_oracle \
    --neg-prefix  "[SUPPRESS]" \
    2>&1 | tee outputs/eval_qwen_mgnm_wrong_token.log

# ── Step 5: Eval no-token model ──────────────────────────
echo ""
echo "=== Step 5: No-token model (no prefix at inference) ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_models.py \
    --models      "${CFG_TMP}" \
    --input       "$EVAL_DATA" \
    --output      outputs/eval_qwen_mgnm_notoken_model.json \
    --cache-dir   outputs/score_cache_notoken_model \
    --model-names qwen_mgnm_notoken \
    --neg-prefix  "" \
    2>&1 | tee outputs/eval_qwen_mgnm_notoken_model.log

# ── Step 6: Summary table ─────────────────────────────────
echo ""
echo "======================================================"
echo " SUMMARY: Behavior-Token Ablation"
echo "======================================================"
python3 << 'PYEOF'
import json
from pathlib import Path

BASE = Path('outputs')

FILES = {
    'Oracle token (current)':      BASE / 'eval_qwen_mgnm_v2.json',
    'Oracle model, no prefix':     BASE / 'eval_qwen_mgnm_no_prefix.json',
    'Oracle model, wrong token':   BASE / 'eval_qwen_mgnm_wrong_token.json',
    'No-token model':              BASE / 'eval_qwen_mgnm_notoken_model.json',
}

print(f"{'Setting':<30} {'FlipAcc':>8} {'ScopeCtrl':>10} {'NegRank':>8}")
print("-" * 60)
for label, path in FILES.items():
    if not path.exists():
        print(f"{label:<30} {'(missing)':>8}")
        continue
    d = json.loads(path.read_text())
    key = list(d.keys())[0]
    s = d[key]['summary']
    flip  = s.get('FlipAcc', {}).get('mean', 0) * 100
    scope = s.get('ScopeControlAcc', {}).get('mean', 0) * 100
    rank  = s.get('NegRankAcc', {}).get('mean', 0) * 100
    print(f"{label:<30} {flip:>7.1f}% {scope:>9.1f}% {rank:>7.1f}%")
PYEOF
