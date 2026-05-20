#!/usr/bin/env bash
# P18: Training data scale ablation — 25% / 50% / 75% / 100%
# Trains three Qwen MGNM variants in parallel (GPU 0/1/2).
# 100% uses the existing best model (no retraining).
# Evaluates all four on v2 test set, then prints learning curve table.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_FULL="data/processed/splits/merged_excl_boost/train.jsonl"
TEST_V2="data/processed/validated_largetest_v2.jsonl"
MODELS_CFG="configs/model_config.json"
MODEL_NAME="qwen2_5_7b"

# ── Step 1: Generate sampled training files ──────────────────────────────
echo "=== Step 1: Sampling training data ==="
python3 - << 'PYEOF'
import json, random, pathlib

random.seed(42)
recs = [json.loads(l) for l in open("data/processed/splits/merged_excl_boost/train.jsonl")]
random.shuffle(recs)
n = len(recs)  # 471

for frac, label in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
    k = round(n * frac)
    sample = recs[:k]
    out = pathlib.Path(f"data/processed/splits/scale_{label}pct_train.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in sample) + "\n")
    print(f"  {label}% → {k} records → {out}")
PYEOF

# ── Step 2: Train 25% / 50% / 75% in parallel ───────────────────────────
echo ""
echo "=== Step 2: Training (GPU 0/1/2 in parallel) ==="

_train() {
  local frac="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_mgnm.py \
      --models          "$MODELS_CFG" \
      --model-name      "$MODEL_NAME" \
      --train-input     "data/processed/splits/scale_${frac}pct_train.jsonl" \
      --output-dir      "outputs/qwen_scale_${frac}pct" \
      --epochs          3 \
      --learning-rate   2e-4 \
      --lambda-pos      1.0 \
      --lambda-sup      1.0 \
      --lambda-rank     1.5 \
      --lambda-preserve 1.5 \
      --lambda-ret      0.05 \
      --lambda-rank-select 3.0 \
      --lambda-rank-dist   3.0 \
      --scope-oversample   3 \
      --behavior-token \
      > "outputs/train_scale_${frac}pct.log" 2>&1
  echo "  ${frac}% training done ✓"
}

_train 25 0 &  PID_25=$!
_train 50 1 &  PID_50=$!
_train 75 2 &  PID_75=$!

echo "  PIDs: 25%=$PID_25  50%=$PID_50  75%=$PID_75"
echo "  Waiting for training to complete..."
wait $PID_25 || { echo "  25% training FAILED"; exit 1; }
wait $PID_50 || { echo "  50% training FAILED"; exit 1; }
wait $PID_75 || { echo "  75% training FAILED"; exit 1; }

# ── Step 3: Build eval configs ────────────────────────────────────────────
echo ""
echo "=== Step 3: Building eval configs ==="
python3 - << 'PYEOF'
import json
base_cfg = json.load(open("configs/model_config.json"))
qwen_entry = next(m for m in base_cfg["models"] if m["name"] == "qwen2_5_7b")
cfg = {"models": []}
for frac in ["25", "50", "75"]:
    entry = dict(qwen_entry)
    entry["name"] = f"qwen_scale_{frac}pct"
    entry["adapter_path"] = f"outputs/qwen_scale_{frac}pct/adapter"
    cfg["models"].append(entry)
json.dump(cfg, open("configs/model_config_scale.json", "w"), indent=2)
print("  Wrote configs/model_config_scale.json")
PYEOF

# ── Step 4: Evaluate all three on v2 test set ────────────────────────────
echo ""
echo "=== Step 4: Evaluating on v2 test set (GPU 0/1/2) ==="

_eval() {
  local frac="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_models.py \
      --models configs/model_config_scale.json \
      --input  "$TEST_V2" \
      --output "outputs/eval_qwen_scale_${frac}pct_v2.json" \
      --cache-dir "outputs/score_cache_scale_${frac}pct" \
      --model-names "qwen_scale_${frac}pct" \
      --neg-prefix auto \
      >> "outputs/eval_scale_${frac}pct.log" 2>&1
  echo "  ${frac}% eval done ✓"
}

_eval 25 0 &  PID_E25=$!
_eval 50 1 &  PID_E50=$!
_eval 75 2 &  PID_E75=$!

wait $PID_E25; wait $PID_E50; wait $PID_E75

# ── Step 5: Print learning curve table ───────────────────────────────────
echo ""
echo "=== Step 5: Learning Curve Table ==="
python3 - << 'PYEOF'
import json, os

def load(path, key=None):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    v = d[key] if key and key in d else next(iter(d.values()))
    return v.get("summary") or {}

def row(label, path, key=None):
    s = load(path, key)
    if not s:
        print(f"  {label:<25} {'(not found)':>8}")
        return
    nr = s.get("NegRankAcc", {}).get("mean", 0) * 100
    fa = s.get("FlipAcc", {}).get("mean", 0) * 100
    sc = s.get("ScopeControlAcc", {}).get("mean", 0) * 100
    on = s.get("OverNegationRate", {}).get("mean", 0) * 100
    print(f"  {label:<25} {nr:>8.1f} {fa:>8.1f} {sc:>10.1f} {on:>8.1f}")

print(f"\n  {'Scale':25s} {'NegRank':>8} {'FlipAcc':>8} {'ScopeCtrl':>10} {'OverNeg':>8}")
print("  " + "-"*67)
row("25%  (118 raw / ~216 eff)", "outputs/eval_qwen_scale_25pct_v2.json",  "qwen_scale_25pct")
row("50%  (236 raw / ~432 eff)", "outputs/eval_qwen_scale_50pct_v2.json",  "qwen_scale_50pct")
row("75%  (353 raw / ~649 eff)", "outputs/eval_qwen_scale_75pct_v2.json",  "qwen_scale_75pct")
row("100% (471 raw / 865 eff) ", "outputs/eval_qwen_mgnm_v2.json",         "qwen_mgnm")
print()
print("  Note: 'eff' = effective records after scope_oversample=3 on oos/dn types")
PYEOF

echo ""
echo "Done. Results saved to outputs/eval_qwen_scale_*_v2.json"
