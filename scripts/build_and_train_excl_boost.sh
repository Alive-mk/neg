#!/usr/bin/env bash
# After data generation completes, this script:
#   1. Validates generated data (with verifier)
#   2. Merges with existing merged train split
#   3. Trains Qwen rs3 on augmented data
#   4. Evaluates: largetest + WikiFact + MMLU
#
# Usage: bash scripts/build_and_train_excl_boost.sh [TRAIN_GPU] [EVAL_GPU]
#   default: TRAIN_GPU=1, EVAL_GPU=6

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_GPU="${1:-1}"
EVAL_GPU="${2:-6}"
RAW="data/raw/generated_excl_boost.jsonl"
VALIDATED="data/processed/validated_excl_boost.jsonl"
MERGED_ORIG="data/processed/splits/merged/train.jsonl"
MERGED_NEW="data/processed/splits/merged_excl_boost/train.jsonl"
LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
OUT_DIR="outputs/e4_qwen_r3v5_lp15_rs3_boost"
CFG_TMP="/tmp/model_cfg_boost.json"

echo "======================================================"
echo " build_and_train_excl_boost.sh"
echo " TRAIN_GPU=cuda:${TRAIN_GPU}  EVAL_GPU=cuda:${EVAL_GPU}"
echo "======================================================"

# ── Step 1: Validate generated data ────────────────────────
echo ""
echo "=== Step 1: Validate generated_excl_boost.jsonl ==="
python scripts/validate_dataset.py \
    --input   "$RAW" \
    --output  "$VALIDATED" \
    --report  outputs/validation_report_excl_boost.json \
    --models  configs/model_config.json \
    --verifier-name verifier \
    --verifier-threshold 0.7

python3 -c "
import json
from collections import Counter
report = json.load(open('outputs/validation_report_excl_boost.json'))
print('Validated items:', report.get('valid_count', report.get('n_valid','?')))
print('Rejected:', report.get('invalid_count', report.get('n_invalid','?')))
# count by mode
lines = open('data/processed/validated_excl_boost.jsonl').readlines()
c = Counter(json.loads(l).get('semantic_mode','?')+'|'+json.loads(l).get('scope_type','?') for l in lines)
for k,v in sorted(c.items()): print(f'  {k}: {v}')
"

# ── Step 2: Merge with existing train split ─────────────────
echo ""
echo "=== Step 2: Merge with existing merged train ==="
mkdir -p data/processed/splits/merged_excl_boost
python3 -c "
import json, random

orig = open('$MERGED_ORIG').readlines()
boost = open('$VALIDATED').readlines()

# Only keep exclusive_choice|in_scope from boost
boost_excl = [l for l in boost
              if json.loads(l).get('semantic_mode')=='exclusive_choice'
              and json.loads(l).get('scope_type')=='in_scope']

print(f'Original train: {len(orig)} records')
print(f'Boost (excl|in_scope only): {len(boost_excl)} records')

combined = orig + boost_excl
random.seed(42)
random.shuffle(combined)

with open('$MERGED_NEW', 'w') as f:
    for line in combined:
        f.write(line.rstrip() + '\n')

print(f'New merged train: {len(combined)} records')

# Distribution
from collections import Counter
c = Counter(json.loads(l).get('semantic_mode','?')+'|'+json.loads(l).get('scope_type','?') for l in combined)
for k,v in sorted(c.items()):
    print(f'  {k}: {v} ({round(v/len(combined)*100,1)}%)')
"

# ── Step 3: Train Qwen rs3 on augmented data ────────────────
echo ""
echo "=== Step 3: Train Qwen R3v5_lp15_rs3_boost ==="
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python scripts/train_mgnm.py \
    --models         configs/model_config.json \
    --model-name     qwen2_5_7b \
    --train-input    "$MERGED_NEW" \
    --eval-input     data/processed/splits/balanced/dev.jsonl \
    --output-dir     "$OUT_DIR" \
    --epochs         3 \
    --learning-rate  2e-4 \
    --weight-decay   0.01 \
    --lambda-preserve 1.5 \
    --lambda-rank    1.5 \
    --lambda-rank-select 3.0 \
    --lambda-ret     0.05 \
    --behavior-token \
    --scope-oversample 1 \
    2>&1 | tee outputs/train_qwen_rs3_boost.log

echo "Training done."

# ── Step 4: Build eval config ────────────────────────────────
echo ""
echo "=== Step 4: Build eval model config ==="
python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
models = cfg['models']

base = next(m for m in models if m['name']=='qwen2_5_7b')
rs3_entry = dict(base)
rs3_entry['name'] = 'qwen_rs3'
rs3_entry['adapter_path'] = 'outputs/e4_qwen_r3v5_lp15_rs3/adapter'
boost_entry = dict(base)
boost_entry['name'] = 'qwen_rs3_boost'
boost_entry['adapter_path'] = '${OUT_DIR}/adapter'

models = models + [rs3_entry, boost_entry]
json.dump({'models': models}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
"

# ── Step 5: largetest eval ───────────────────────────────────
echo ""
echo "=== Step 5: largetest eval ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_models.py \
    --models    "${CFG_TMP}" \
    --input     "$LARGETEST" \
    --output    outputs/eval_qwen_rs3_boost_largetest.json \
    --cache-dir outputs/score_cache_qwen_rs3_boost \
    --model-names qwen_rs3_boost \
    --neg-prefix auto \
    2>&1 | tee outputs/eval_qwen_rs3_boost_largetest.log

# ── Step 6: WikiFact eval ────────────────────────────────────
echo ""
echo "=== Step 6: WikiFact eval ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_models.py \
    --models    "${CFG_TMP}" \
    --input     "$WIKIFACT" \
    --output    outputs/eval_qwen_rs3_boost_wikifact.json \
    --cache-dir outputs/score_cache_qwen_rs3_boost_wf \
    --model-names qwen_rs3_boost \
    --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_qwen_rs3_boost_wikifact.log

# ── Step 7: MMLU eval ────────────────────────────────────────
echo ""
echo "=== Step 7: MMLU eval ==="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python scripts/evaluate_mmlu.py \
    --base-model-path model/Qwen2.5-7B \
    --adapter-path    "${OUT_DIR}/adapter" \
    --output          outputs/mmlu_qwen_rs3_boost.json \
    --n-per-subject   10 --seed 42 \
    2>&1 | tee outputs/mmlu_qwen_rs3_boost.log

# ── Step 8: Comparison table ─────────────────────────────────
echo ""
echo "======================================================"
echo " FINAL COMPARISON: rs3  vs  rs3_boost"
echo "======================================================"
python3 - <<'PYEOF'
import json, os
from collections import defaultdict

files = {
    "qwen2_5_7b_e4":   "outputs/eval_r3v5_lp15_largetest.json",
    "qwen_rs3":        "outputs/eval_qwen_rs3_largetest.json",
    "qwen_rs3_boost":  "outputs/eval_qwen_rs3_boost_largetest.json",
}

merged = {}
for key, path in files.items():
    if not os.path.exists(path): continue
    d = json.load(open(path))
    if key in d:
        merged[key] = d[key]
    else:
        merged[key] = next(iter(d.values()))

MODELS = [
    ("qwen2_5_7b_e4",  "Qwen R3v5_lp15 (baseline)"),
    ("qwen_rs3",       "Qwen rs3 (λ_sel=3.0, 412条)"),
    ("qwen_rs3_boost", "Qwen rs3_boost (λ_sel=3.0, 增强数据)"),
]

def g(d, k): return round(d.get(k,{}).get("mean",0)*100,1)

print(f"\n{'Model':<38} {'NegSupp':>7} {'NegRank':>7} {'FlipAcc':>7} {'Scope':>6} {'OverNeg':>7}")
print("-"*80)
for key, label in MODELS:
    if key not in merged: continue
    agg = merged[key].get("aggregate", merged[key].get("summary", {}))
    print(f"{label:<38} {g(agg,'NegSuppRate'):>7} {g(agg,'NegRankAcc'):>7} "
          f"{g(agg,'FlipAcc'):>7} {g(agg,'ScopeControlAcc'):>6} {g(agg,'OverNegationRate'):>7}")

# per-mode breakdown for NegRankAcc
print("\n── exclusive_choice|in_scope NegRankAcc breakdown ──")
for key, label in MODELS:
    if key not in merged: continue
    records = merged[key].get("per_record", [])
    excl = [r for r in records if r.get("semantic_mode")=="exclusive_choice" and r.get("scope_type")=="in_scope"]
    if excl:
        rank = sum(1 for r in excl if r.get("neg_rank_correct"))
        print(f"  {label}: {rank}/{len(excl)} = {round(rank/len(excl)*100,1)}%")

print()
# MMLU
for label, path in [("rs3","outputs/mmlu_qwen_rs3.json"),("rs3_boost","outputs/mmlu_qwen_rs3_boost.json")]:
    if os.path.exists(path):
        d = json.load(open(path))
        delta = round(d.get("delta",{}).get("overall_accuracy",0)*100,2)
        print(f"  MMLU {label}: Δ={delta:+.2f}%")

print()
for label, path in [("rs3","outputs/eval_qwen_rs3_wikifact.json"),
                    ("rs3_boost","outputs/eval_qwen_rs3_boost_wikifact.json")]:
    if os.path.exists(path):
        d = json.load(open(path))
        v = next(iter(d.values()))
        s = v.get("summary", {})
        nsr = round(s.get("NegSuppRate",{}).get("mean",0)*100,1)
        fa  = round(s.get("FlipAcc",{}).get("mean",0)*100,1)
        print(f"  WikiFact {label}: NegSuppRate={nsr}%  FlipAcc={fa}%")
PYEOF

echo ""
echo "Done."
