#!/usr/bin/env bash
# Llama λ_rank_dist fine-grained search.
# Goal: reduce Qwen/Llama NegRank gap from 23pp to ≤15pp.
# Previous results:
#   Llama rd15 (λ_dist=1.5): NegRank=56.9%, MMLU=-3.68%, WikiFact=86%  ← too costly
#   Llama ret01 (λ_dist=0):  NegRank=45.8%, MMLU=-2.11%, WikiFact=92%  ← current best
# Try: λ_dist=0.5 and λ_dist=1.0 with os=2.

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="data/processed/splits/merged/train.jsonl"
MODELS="configs/model_config.json"
MODEL_NAME="llama_3_1_8b"
LLAMA_PATH="/data/share/neg/model/Meta-Llama-3.1-8B"
LARGETEST="data/processed/validated_largetest.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"

# ── λ_dist=0.5, os=2 (GPU 0) ─────────────────────────────────────────────
echo "[llama_rd05_os2] starting training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/llama_rd05_os2 \
    --epochs          3 \
    --lambda-ret      0.1 \
    --lambda-pos      1.0 \
    --lambda-sup      1.5 \
    --lambda-rank     1.0 \
    --lambda-preserve 1.5 \
    --lambda-rank-dist 0.5 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_llama_rd05_os2.log &
PID1=$!

# ── λ_dist=1.0, os=2 (GPU 1) ─────────────────────────────────────────────
echo "[llama_rd10_os2] starting training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/llama_rd10_os2 \
    --epochs          3 \
    --lambda-ret      0.1 \
    --lambda-pos      1.0 \
    --lambda-sup      1.5 \
    --lambda-rank     1.0 \
    --lambda-preserve 1.5 \
    --lambda-rank-dist 1.0 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_llama_rd10_os2.log &
PID2=$!

echo "Both Llama training jobs launched (PIDs: $PID1, $PID2). Waiting..."
wait $PID1
echo "[llama_rd05_os2] training done."
wait $PID2
echo "[llama_rd10_os2] training done."

# ── Build model configs ───────────────────────────────────────────────────
python3 - << 'PYEOF'
import json
for key, adapter in [
    ("llama_rd05_os2", "outputs/llama_rd05_os2/adapter"),
    ("llama_rd10_os2", "outputs/llama_rd10_os2/adapter"),
]:
    cfg = json.load(open("configs/model_config.json"))
    entry = next(m for m in cfg["models"] if m["name"] == "llama_3_1_8b").copy()
    entry["name"] = key
    entry["adapter_path"] = adapter
    cfg["models"].append(entry)
    json.dump(cfg, open(f"/tmp/model_cfg_{key}.json", "w"), indent=2)
PYEOF

# ── Eval rd05_os2 (GPU 6) ────────────────────────────────────────────────
echo "[eval] llama_rd05_os2 on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_llama_rd05_os2.json \
    --input "$LARGETEST" \
    --output "outputs/eval_llama_rd05_os2_largetest.json" \
    --cache-dir "outputs/score_cache_llama_rd05_os2" \
    --model-names "llama_rd05_os2" --neg-prefix auto \
    2>&1 | tee outputs/eval_llama_rd05_os2_largetest.log

CUDA_VISIBLE_DEVICES=6 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_llama_rd05_os2.json \
    --input "$WIKIFACT" \
    --output "outputs/eval_llama_rd05_os2_wikifact.json" \
    --cache-dir "outputs/score_cache_llama_rd05_os2_wf" \
    --model-names "llama_rd05_os2" --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_llama_rd05_os2_wikifact.log

CUDA_VISIBLE_DEVICES=6 python scripts/evaluate_mmlu.py \
    --base-model-path "$LLAMA_PATH" \
    --adapter-path "outputs/llama_rd05_os2/adapter" \
    --output "outputs/mmlu_llama_rd05_os2.json" \
    --n-per-subject 10 --seed 42 \
    2>&1 | tee outputs/mmlu_llama_rd05_os2.log

# ── Eval rd10_os2 (GPU 7) ────────────────────────────────────────────────
echo "[eval] llama_rd10_os2 on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_llama_rd10_os2.json \
    --input "$LARGETEST" \
    --output "outputs/eval_llama_rd10_os2_largetest.json" \
    --cache-dir "outputs/score_cache_llama_rd10_os2" \
    --model-names "llama_rd10_os2" --neg-prefix auto \
    2>&1 | tee outputs/eval_llama_rd10_os2_largetest.log

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_models.py \
    --models /tmp/model_cfg_llama_rd10_os2.json \
    --input "$WIKIFACT" \
    --output "outputs/eval_llama_rd10_os2_wikifact.json" \
    --cache-dir "outputs/score_cache_llama_rd10_os2_wf" \
    --model-names "llama_rd10_os2" --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_llama_rd10_os2_wikifact.log

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_mmlu.py \
    --base-model-path "$LLAMA_PATH" \
    --adapter-path "outputs/llama_rd10_os2/adapter" \
    --output "outputs/mmlu_llama_rd10_os2.json" \
    --n-per-subject 10 --seed 42 \
    2>&1 | tee outputs/mmlu_llama_rd10_os2.log

# ── Final summary ────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " LLAMA λ_rank_dist SEARCH RESULTS"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

ref = {"NegRank": 45.8, "FlipAcc": 57.3, "ScopeCtrl": 94.0, "OverNeg": 6.0, "WikiFact": 92.0, "MMLU": -2.11}
print(f"{'Model':25s} {'NegRank':>8s} {'FlipAcc':>8s} {'ScopeCtrl':>10s} {'OverNeg':>8s} {'WikiFact':>9s} {'MMLU':>7s}")
print(f"{'Llama ret01 (baseline)':25s} {ref['NegRank']:>8.1f} {ref['FlipAcc']:>8.1f} {ref['ScopeCtrl']:>10.1f} {ref['OverNeg']:>8.1f} {ref['WikiFact']:>9.1f} {ref['MMLU']:>+7.2f}")

for key in ["llama_rd05_os2", "llama_rd10_os2"]:
    lt = f"outputs/eval_{key}_largetest.json"
    wf = f"outputs/eval_{key}_wikifact.json"
    mm = f"outputs/mmlu_{key}.json"
    if not os.path.exists(lt):
        print(f"{key}: not yet evaluated")
        continue
    s  = json.load(open(lt))
    s  = next(iter(s.values()))["summary"]
    nr = s["NegRankAcc"]["mean"]*100
    fa = s["FlipAcc"]["mean"]*100
    sc = s["ScopeControlAcc"]["mean"]*100
    on = s["OverNegationRate"]["mean"]*100
    wf_v = json.load(open(wf)) if os.path.exists(wf) else {}
    wf_fa = next(iter(wf_v.values()))["summary"]["FlipAcc"]["mean"]*100 if wf_v else 0
    mm_v = json.load(open(mm)) if os.path.exists(mm) else {}
    mmlu = mm_v.get("delta", {}).get("overall_accuracy", 0)*100 if mm_v else 0
    print(f"{key:25s} {nr:>8.1f} {fa:>8.1f} {sc:>10.1f} {on:>8.1f} {wf_fa:>9.1f} {mmlu:>+7.2f}")
PYEOF
echo "Done."
