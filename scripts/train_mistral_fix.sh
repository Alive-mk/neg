#!/usr/bin/env bash
# Mistral fix: dramatically lower λ_ret to stop KL explosion.
# Root cause: L_ret for Mistral is 5-13x higher than Qwen even at λ_ret=0.05.
# Solution: try λ_ret=0.01 and 0.02 with os=2.
#
# Usage: bash scripts/train_mistral_fix.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="data/processed/splits/merged/train.jsonl"
MODELS="configs/model_config.json"
MODEL_NAME="mistral_7b_instruct_v0_3"
MISTRAL_PATH="/data/share/neg/model/Mistral-7B-Instruct-v0.3"
LARGETEST="data/processed/validated_largetest.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"

# ── λ_ret=0.01, os=2 (GPU 2) ──────────────────────────────────────────────
echo "[mistral_ret001_os2] starting training on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/mistral_ret001_os2 \
    --epochs          3 \
    --lambda-ret      0.01 \
    --lambda-pos      1.0 \
    --lambda-sup      1.5 \
    --lambda-rank     1.0 \
    --lambda-preserve 1.5 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_mistral_ret001_os2.log &
PID1=$!

# ── λ_ret=0.02, os=2 (GPU 3) ──────────────────────────────────────────────
echo "[mistral_ret002_os2] starting training on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/mistral_ret002_os2 \
    --epochs          3 \
    --lambda-ret      0.02 \
    --lambda-pos      1.0 \
    --lambda-sup      1.5 \
    --lambda-rank     1.0 \
    --lambda-preserve 1.5 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_mistral_ret002_os2.log &
PID2=$!

echo "Both training jobs launched (PIDs: $PID1, $PID2). Waiting..."
wait $PID1
echo "[mistral_ret001_os2] training done."
wait $PID2
echo "[mistral_ret002_os2] training done."

# ── Eval ret001_os2 (GPU 4) ────────────────────────────────────────────────
echo "[eval] mistral_ret001_os2 on GPU 4..."
python3 - << PYEOF
import json, subprocess, sys

def make_cfg(key, adapter_dir, base_path, cfg_path):
    cfg = json.load(open("configs/model_config.json"))
    entry = next(m for m in cfg["models"] if m["name"] == "mistral_7b_instruct_v0_3").copy()
    entry["name"] = key
    entry["adapter_path"] = adapter_dir
    cfg["models"].append(entry)
    json.dump(cfg, open(cfg_path, "w"), indent=2)

for key, adapter in [
    ("mistral_ret001_os2", "outputs/mistral_ret001_os2/adapter"),
    ("mistral_ret002_os2", "outputs/mistral_ret002_os2/adapter"),
]:
    cfg_path = f"/tmp/model_cfg_{key}.json"
    make_cfg(key, adapter, "mistral", cfg_path)
PYEOF

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_mistral_ret001_os2.json \
    --input "$LARGETEST" \
    --output "outputs/eval_mistral_ret001_os2_largetest.json" \
    --cache-dir "outputs/score_cache_mistral_ret001_os2" \
    --model-names "mistral_ret001_os2" --neg-prefix auto \
    2>&1 | tee outputs/eval_mistral_ret001_os2_largetest.log

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_mistral_ret001_os2.json \
    --input "$WIKIFACT" \
    --output "outputs/eval_mistral_ret001_os2_wikifact.json" \
    --cache-dir "outputs/score_cache_mistral_ret001_os2_wf" \
    --model-names "mistral_ret001_os2" --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_mistral_ret001_os2_wikifact.log

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_mmlu.py \
    --base-model-path "$MISTRAL_PATH" \
    --adapter-path "outputs/mistral_ret001_os2/adapter" \
    --output "outputs/mmlu_mistral_ret001_os2.json" \
    --n-per-subject 10 --seed 42 \
    2>&1 | tee outputs/mmlu_mistral_ret001_os2.log

# ── Eval ret002_os2 (GPU 5) ────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_mistral_ret002_os2.json \
    --input "$LARGETEST" \
    --output "outputs/eval_mistral_ret002_os2_largetest.json" \
    --cache-dir "outputs/score_cache_mistral_ret002_os2" \
    --model-names "mistral_ret002_os2" --neg-prefix auto \
    2>&1 | tee outputs/eval_mistral_ret002_os2_largetest.log

CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_mistral_ret002_os2.json \
    --input "$WIKIFACT" \
    --output "outputs/eval_mistral_ret002_os2_wikifact.json" \
    --cache-dir "outputs/score_cache_mistral_ret002_os2_wf" \
    --model-names "mistral_ret002_os2" --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_mistral_ret002_os2_wikifact.log

CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_mmlu.py \
    --base-model-path "$MISTRAL_PATH" \
    --adapter-path "outputs/mistral_ret002_os2/adapter" \
    --output "outputs/mmlu_mistral_ret002_os2.json" \
    --n-per-subject 10 --seed 42 \
    2>&1 | tee outputs/mmlu_mistral_ret002_os2.log

# ── Final summary ─────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " MISTRAL FIX RESULTS"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

def show(key):
    lt = f"outputs/eval_{key}_largetest.json"
    wf = f"outputs/eval_{key}_wikifact.json"
    mm = f"outputs/mmlu_{key}.json"
    print(f"\n=== {key} ===")
    if os.path.exists(lt):
        d = json.load(open(lt))
        s = next(iter(d.values()))["summary"]
        for k in ["NegRankAcc","FlipAcc","ScopeControlAcc","OverNegationRate","NegSuppRate"]:
            print(f"  {k}: {s[k]['mean']*100:.1f}%")
    if os.path.exists(wf):
        d = json.load(open(wf))
        s = next(iter(d.values()))["summary"]
        print(f"  WikiFact FlipAcc: {s['FlipAcc']['mean']*100:.1f}%")
    if os.path.exists(mm):
        d = json.load(open(mm))
        delta = d["delta"]["overall_accuracy"] * 100
        print(f"  MMLU delta: {delta:+.2f}%")

show("mistral_ret001_os2")
show("mistral_ret002_os2")
PYEOF
echo "Done."
