#!/usr/bin/env bash
# Llama ablation study: nosup and nopre variants.
# These mirror the existing Qwen ablation (eval_ablation_mgnm_nosup/nopre)
# to demonstrate that each loss component is necessary across models.
#
# nosup: remove L_sup (set lambda_sup=0), keep everything else
# nopre: remove L_preserve (set lambda_preserve=0), keep everything else
#
# Base Llama config: ret01 (lambda_ret=0.1, os=2, lambda_rank_dist=0)

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="data/processed/splits/merged/train.jsonl"
MODELS="configs/model_config.json"
MODEL_NAME="llama_3_1_8b"
LLAMA_PATH="/data/share/neg/model/Meta-Llama-3.1-8B"
LARGETEST="data/processed/validated_largetest.jsonl"

# ── Llama nosup: λ_sup=0 (GPU 6, sequential after llama_rd) ─────────────
# Note: runs sequentially after the main llama_rd pipeline since GPU 6 is used there.
# We use a fresh GPU assignment here.

echo "[llama_nosup] starting training on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/llama_abl_nosup \
    --epochs          3 \
    --lambda-ret      0.1 \
    --lambda-pos      1.0 \
    --lambda-sup      0.0 \
    --lambda-rank     1.0 \
    --lambda-preserve 1.5 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_llama_abl_nosup.log

echo "[llama_nopre] starting training on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python scripts/train_mgnm.py \
    --models          "$MODELS" \
    --model-name      "$MODEL_NAME" \
    --train-input     "$TRAIN" \
    --output-dir      outputs/llama_abl_nopre \
    --epochs          3 \
    --lambda-ret      0.1 \
    --lambda-pos      1.0 \
    --lambda-sup      1.5 \
    --lambda-rank     1.0 \
    --lambda-preserve 0.0 \
    --scope-oversample 2 \
    --behavior-token \
    2>&1 | tee outputs/train_llama_abl_nopre.log

# ── Build eval configs ────────────────────────────────────────────────────
python3 - << 'PYEOF'
import json
for key, adapter in [
    ("llama_abl_nosup", "outputs/llama_abl_nosup/adapter"),
    ("llama_abl_nopre", "outputs/llama_abl_nopre/adapter"),
]:
    cfg = json.load(open("configs/model_config.json"))
    entry = next(m for m in cfg["models"] if m["name"] == "llama_3_1_8b").copy()
    entry["name"] = key
    entry["adapter_path"] = adapter
    cfg["models"].append(entry)
    json.dump(cfg, open(f"/tmp/model_cfg_{key}.json", "w"), indent=2)
PYEOF

# ── Eval nosup and nopre (GPU 6 and 7) ───────────────────────────────────
for KEY in llama_abl_nosup llama_abl_nopre; do
    GPU=$( [ "$KEY" = "llama_abl_nosup" ] && echo 6 || echo 7 )
    echo "[eval] $KEY on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate_models.py \
        --models /tmp/model_cfg_${KEY}.json \
        --input "$LARGETEST" \
        --output "outputs/eval_${KEY}_largetest.json" \
        --cache-dir "outputs/score_cache_${KEY}" \
        --model-names "$KEY" --neg-prefix auto \
        2>&1 | tee outputs/eval_${KEY}_largetest.log &
done
wait

# ── Final comparison table ────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " LLAMA ABLATION RESULTS"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

def show(key, label):
    lt = f"outputs/eval_{key}_largetest.json"
    if not os.path.exists(lt):
        print(f"{label:30s}: not found")
        return
    d = json.load(open(lt))
    s = next(iter(d.values()))["summary"]
    nr = s["NegRankAcc"]["mean"]*100
    fa = s["FlipAcc"]["mean"]*100
    sc = s["ScopeControlAcc"]["mean"]*100
    on = s["OverNegationRate"]["mean"]*100
    print(f"{label:30s} NegRank={nr:.1f}% FlipAcc={fa:.1f}% ScopeCtrl={sc:.1f}% OverNeg={on:.1f}%")

# Reference from existing Qwen ablation
qwen_ref = {
    "Base":          (41.7, 17.3, 86.2, 13.8),
    "nosup":         (48.6, 31.6, 84.4, 15.6),
    "nopre":         (38.2, 52.0, 13.8, 86.2),
    "MGNM-full":     (68.8, 68.4, 91.0,  9.0),
}
print(f"{'Model':30s} {'NegRank':>8s} {'FlipAcc':>8s} {'ScopeCtrl':>10s} {'OverNeg':>8s}")
for name, (nr, fa, sc, on) in qwen_ref.items():
    print(f"{'Qwen ' + name:30s} {nr:>8.1f} {fa:>8.1f} {sc:>10.1f} {on:>8.1f}")
print()
# Llama ablations
show("llama_3_1_8b",    "Llama Base")          # base model result from existing eval
show("llama_abl_nosup", "Llama nosup")
show("llama_abl_nopre", "Llama nopre")
print(f"{'Llama MGNM-full (ret01)':30s} {'45.8':>8s} {'57.3':>8s} {'94.0':>10s} {'6.0':>8s}")
PYEOF
echo "Done."
