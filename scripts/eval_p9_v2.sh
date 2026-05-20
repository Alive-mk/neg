#!/usr/bin/env bash
# P9: v2 re-eval for Mistral lr5e5_os1, Qwen/Llama ablations, and prompt baselines
# Uses GPUs 1-4 in parallel

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="/tmp/model_cfg_abl_v2.json"
V2="data/processed/validated_largetest_v2.jsonl"

# GPU 1: Mistral lr5e5_os1 on v2
echo "[P9] GPU1: mistral_lr5e5_os1 v2..."
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_models.py \
    --models "$MODELS" \
    --input "$V2" \
    --output "outputs/eval_mistral_lr5e5_os1_v2.json" \
    --cache-dir "outputs/score_cache_mistral_lr5e5_v2" \
    --model-names "mistral_lr5e5_os1" --neg-prefix auto \
    2>&1 | tee outputs/eval_mistral_lr5e5_os1_v2.log &
PID1=$!

# GPU 2: Qwen ablation on v2 (sequential nosup then nopre)
(
echo "[P9] GPU2: qwen_nosup v2..."
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_models.py \
    --models "$MODELS" \
    --input "$V2" \
    --output "outputs/eval_qwen_nosup_v2.json" \
    --cache-dir "outputs/score_cache_qwen_nosup_v2" \
    --model-names "qwen_nosup" --neg-prefix auto \
    2>&1 | tee outputs/eval_qwen_nosup_v2.log

echo "[P9] GPU2: qwen_nopre v2..."
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_models.py \
    --models "$MODELS" \
    --input "$V2" \
    --output "outputs/eval_qwen_nopre_v2.json" \
    --cache-dir "outputs/score_cache_qwen_nopre_v2" \
    --model-names "qwen_nopre" --neg-prefix auto \
    2>&1 | tee outputs/eval_qwen_nopre_v2.log
) &
PID2=$!

# GPU 3: Llama ablation on v2 (sequential nosup then nopre)
(
echo "[P9] GPU3: llama_nosup v2..."
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_models.py \
    --models "$MODELS" \
    --input "$V2" \
    --output "outputs/eval_llama_nosup_v2.json" \
    --cache-dir "outputs/score_cache_llama_nosup_v2" \
    --model-names "llama_nosup" --neg-prefix auto \
    2>&1 | tee outputs/eval_llama_nosup_v2.log

echo "[P9] GPU3: llama_nopre v2..."
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_models.py \
    --models "$MODELS" \
    --input "$V2" \
    --output "outputs/eval_llama_nopre_v2.json" \
    --cache-dir "outputs/score_cache_llama_nopre_v2" \
    --model-names "llama_nopre" --neg-prefix auto \
    2>&1 | tee outputs/eval_llama_nopre_v2.log
) &
PID3=$!

# GPU 4: Prompt baselines (none/warning/persona/cot) on v2
echo "[P9] GPU4: prompt baselines v2..."
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_prompt_baseline.py \
    --models configs/model_config.json \
    --model-name qwen2_5_7b \
    --input "$V2" \
    --output "outputs/eval_prompt_baseline_v2.json" \
    --cache-dir "outputs/score_cache_prompt_v2" \
    --styles none warning persona cot \
    2>&1 | tee outputs/eval_prompt_baseline_v2.log &
PID4=$!

echo "All 4 eval jobs launched (PIDs: $PID1, $PID2, $PID3, $PID4)"
wait $PID1 && echo "[done] mistral_lr5e5_os1 v2"
wait $PID2 && echo "[done] qwen ablation v2"
wait $PID3 && echo "[done] llama ablation v2"
wait $PID4 && echo "[done] prompt baselines v2"

echo ""
echo "======================================================"
echo " P9 RESULTS SUMMARY"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

def show(label, path, keys=None):
    if not os.path.exists(path):
        print(f"  {label}: MISSING {path}")
        return
    d = json.load(open(path))
    for model_name, result in d.items():
        s = result.get("summary", {})
        nr = s.get("NegRankAcc", {}).get("mean", 0) * 100
        fa = s.get("FlipAcc", {}).get("mean", 0) * 100
        sc = s.get("ScopeControlAcc", {}).get("mean", 0) * 100
        print(f"  {model_name:30s} NegRank={nr:.1f}% FlipAcc={fa:.1f}% ScopeCtrl={sc:.1f}%")

print("\n--- Mistral ---")
show("mistral_lr5e5_os1", "outputs/eval_mistral_lr5e5_os1_v2.json")

print("\n--- Qwen Ablation ---")
show("qwen_nosup", "outputs/eval_qwen_nosup_v2.json")
show("qwen_nopre", "outputs/eval_qwen_nopre_v2.json")

print("\n--- Llama Ablation ---")
show("llama_nosup", "outputs/eval_llama_nosup_v2.json")
show("llama_nopre", "outputs/eval_llama_nopre_v2.json")

print("\n--- Prompt Baselines ---")
if os.path.exists("outputs/eval_prompt_baseline_v2.json"):
    d = json.load(open("outputs/eval_prompt_baseline_v2.json"))
    for style, result in d.items():
        s = result.get("summary", {})
        nr = s.get("NegRankAcc", {}).get("mean", 0) * 100
        fa = s.get("FlipAcc", {}).get("mean", 0) * 100
        sc = s.get("ScopeControlAcc", {}).get("mean", 0) * 100
        print(f"  prompt_{style:20s} NegRank={nr:.1f}% FlipAcc={fa:.1f}% ScopeCtrl={sc:.1f}%")
PYEOF

echo ""
echo "Done."
