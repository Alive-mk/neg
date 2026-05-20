#!/usr/bin/env bash
# One-shot evaluation for the three new training runs:
#   - llama_r3v5_lp15_ret01   (Llama λ_ret=0.10)
#   - e4_qwen_r3v5_lp15_rs3   (Qwen λ_rank_select=3.0)
#   - e4_qwen_r3v5_lp15_rs5   (Qwen λ_rank_select=5.0)
#
# Usage:
#   bash scripts/eval_new_models.sh [GPU_ID]
#   GPU_ID defaults to 6

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-6}"
LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_new_models.json"

echo "======================================================"
echo " eval_new_models.sh  GPU=cuda:${GPU}"
echo "======================================================"

# ── helpers ────────────────────────────────────────────────
wait_for_training() {
    local dir="$1"
    local label="$2"
    echo ""
    echo "[wait] Checking ${label} (${dir}) ..."
    while [ ! -f "${dir}/training_summary.json" ]; do
        echo "[wait] ${label} still training — sleeping 120s ..."
        sleep 120
    done
    echo "[wait] ${label} done."
}

build_cfg() {
    python3 - <<PYEOF
import json

base_cfg = json.load(open("configs/model_config.json"))
models = base_cfg["models"]

def add_adapter(base_name, adapter_dir, new_name, device="auto"):
    base = next(m for m in models if m["name"] == base_name)
    entry = dict(base)
    entry["name"] = new_name
    entry["adapter_path"] = f"{adapter_dir}/adapter"
    entry["device_map"] = device
    return entry

models = models + [
    add_adapter("qwen2_5_7b",  "outputs/e4_qwen_r3v5_lp15_rs3",  "qwen_rs3"),
    add_adapter("qwen2_5_7b",  "outputs/e4_qwen_r3v5_lp15_rs5",  "qwen_rs5"),
    add_adapter("llama_3_1_8b","outputs/llama_r3v5_lp15_ret01",   "llama_ret01"),
]
json.dump({"models": models}, open("${CFG_TMP}", "w"), indent=2)
print("Config written to ${CFG_TMP}")
PYEOF
}

print_table() {
    local json_file="$1"
    local label="$2"
    echo ""
    echo "── ${label} ──"
    python3 - "$json_file" <<'PYEOF'
import json, sys

data = json.load(open(sys.argv[1]))

MODELS = [
    ("qwen2_5_7b_e4",  "Qwen R3v5_lp15 (baseline)"),
    ("qwen_rs3",       "Qwen lp15_rs3 (λ_sel=3.0)"),
    ("qwen_rs5",       "Qwen lp15_rs5 (λ_sel=5.0)"),
    ("llama_3_1_8b_e4","Llama lp15_ret02 (λ_ret=0.2)"),
    ("llama_ret01",    "Llama lp15_ret01 (λ_ret=0.1)"),
]

def g(d, k): return round(d.get(k, {}).get("mean", 0) * 100, 1)

header = f"{'Model':<34} {'NegSupp':>7} {'NegRank':>7} {'FlipAcc':>7} {'ScopeCtrl':>9} {'OverNeg':>7} {'PosAcc':>7}"
print(header)
print("-" * len(header))
for key, label in MODELS:
    if key not in data:
        continue
    agg = data[key].get("aggregate", data[key].get("summary", {}))
    print(f"{label:<34} "
          f"{g(agg,'NegSuppRate'):>7} "
          f"{g(agg,'NegRankAcc'):>7} "
          f"{g(agg,'FlipAcc'):>7} "
          f"{g(agg,'ScopeControlAcc'):>9} "
          f"{g(agg,'OverNegationRate'):>7} "
          f"{g(agg,'PosAcc'):>7}")
PYEOF
}

# ── wait for all three to finish ───────────────────────────
wait_for_training "outputs/llama_r3v5_lp15_ret01"  "Llama ret01"
wait_for_training "outputs/e4_qwen_r3v5_lp15_rs3"  "Qwen rs3"
wait_for_training "outputs/e4_qwen_r3v5_lp15_rs5"  "Qwen rs5"

echo ""
echo "All training done. Building model config ..."
build_cfg

# ── largetest ──────────────────────────────────────────────
echo ""
echo "====== largetest (n=392) ======"

echo "[eval] Qwen rs3 — largetest"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${LARGETEST}" \
    --output   outputs/eval_qwen_rs3_largetest.json \
    --cache-dir outputs/score_cache_qwen_rs3 \
    --model-names qwen_rs3 \
    --neg-prefix auto \
    2>&1 | tee outputs/eval_qwen_rs3_largetest.log

echo "[eval] Qwen rs5 — largetest"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${LARGETEST}" \
    --output   outputs/eval_qwen_rs5_largetest.json \
    --cache-dir outputs/score_cache_qwen_rs5 \
    --model-names qwen_rs5 \
    --neg-prefix auto \
    2>&1 | tee outputs/eval_qwen_rs5_largetest.log

echo "[eval] Llama ret01 — largetest"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${LARGETEST}" \
    --output   outputs/eval_llama_ret01_largetest.json \
    --cache-dir outputs/score_cache_llama_ret01 \
    --model-names llama_ret01 \
    --neg-prefix auto \
    2>&1 | tee outputs/eval_llama_ret01_largetest.log

# ── wikifact ───────────────────────────────────────────────
echo ""
echo "====== WikiFact OOD (n=100) ======"

echo "[eval] Qwen rs3 — WikiFact"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${WIKIFACT}" \
    --output   outputs/eval_qwen_rs3_wikifact.json \
    --cache-dir outputs/score_cache_qwen_rs3_wf \
    --model-names qwen_rs3 \
    --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_qwen_rs3_wikifact.log

echo "[eval] Qwen rs5 — WikiFact"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${WIKIFACT}" \
    --output   outputs/eval_qwen_rs5_wikifact.json \
    --cache-dir outputs/score_cache_qwen_rs5_wf \
    --model-names qwen_rs5 \
    --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_qwen_rs5_wikifact.log

echo "[eval] Llama ret01 — WikiFact"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models   "${CFG_TMP}" \
    --input    "${WIKIFACT}" \
    --output   outputs/eval_llama_ret01_wikifact.json \
    --cache-dir outputs/score_cache_llama_ret01_wf \
    --model-names llama_ret01 \
    --neg-prefix "[SUPPRESS]" \
    2>&1 | tee outputs/eval_llama_ret01_wikifact.log

# ── MMLU ───────────────────────────────────────────────────
echo ""
echo "====== MMLU capability check ======"

echo "[mmlu] Qwen rs3"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
    --base-model-path model/Qwen2.5-7B \
    --adapter-path    outputs/e4_qwen_r3v5_lp15_rs3/adapter \
    --output          outputs/mmlu_qwen_rs3.json \
    --n-per-subject   10 --seed 42 \
    2>&1 | tee outputs/mmlu_qwen_rs3.log

echo "[mmlu] Qwen rs5"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
    --base-model-path model/Qwen2.5-7B \
    --adapter-path    outputs/e4_qwen_r3v5_lp15_rs5/adapter \
    --output          outputs/mmlu_qwen_rs5.json \
    --n-per-subject   10 --seed 42 \
    2>&1 | tee outputs/mmlu_qwen_rs5.log

echo "[mmlu] Llama ret01"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
    --base-model-path /data/share/neg/model/Meta-Llama-3.1-8B \
    --adapter-path    outputs/llama_r3v5_lp15_ret01/adapter \
    --output          outputs/mmlu_llama_ret01.json \
    --n-per-subject   10 --seed 42 \
    2>&1 | tee outputs/mmlu_llama_ret01.log

# ── summary table ──────────────────────────────────────────
echo ""
echo "======================================================"
echo " FINAL RESULTS"
echo "======================================================"

# merge per-model results into one dict for the table printer
python3 - <<'PYEOF'
import json, os

files = {
    "qwen_rs3":    "outputs/eval_qwen_rs3_largetest.json",
    "qwen_rs5":    "outputs/eval_qwen_rs5_largetest.json",
    "llama_ret01": "outputs/eval_llama_ret01_largetest.json",
}
# also load baseline for comparison
baselines = {
    "qwen2_5_7b_e4":   "outputs/eval_r3v5_lp15_largetest.json",
    "llama_3_1_8b_e4": "outputs/eval_llama_r3v5_lp15_ret02_largetest.json",
}

merged = {}
for key, path in {**baselines, **files}.items():
    if os.path.exists(path):
        d = json.load(open(path))
        if key in d:
            merged[key] = d[key]
        else:
            # single-model output
            for v in d.values():
                merged[key] = v
                break

MODELS = [
    ("qwen2_5_7b_e4",  "Qwen R3v5_lp15 (baseline)"),
    ("qwen_rs3",       "Qwen lp15_rs3 λ_sel=3.0"),
    ("qwen_rs5",       "Qwen lp15_rs5 λ_sel=5.0"),
    ("llama_3_1_8b_e4","Llama ret02 λ_ret=0.2"),
    ("llama_ret01",    "Llama ret01 λ_ret=0.1"),
]

def g(d, k): return round(d.get(k, {}).get("mean", 0) * 100, 1)

print(f"\n{'Model':<36} {'NegSupp':>7} {'NegRank':>7} {'FlipAcc':>7} {'Scope':>6} {'OverNeg':>7} {'PosAcc':>7}")
print("-" * 80)
for key, label in MODELS:
    if key not in merged:
        print(f"{label:<36} (not found)")
        continue
    agg = merged[key].get("aggregate", merged[key].get("summary", {}))
    print(f"{label:<36} "
          f"{g(agg,'NegSuppRate'):>7} "
          f"{g(agg,'NegRankAcc'):>7} "
          f"{g(agg,'FlipAcc'):>7} "
          f"{g(agg,'ScopeControlAcc'):>6} "
          f"{g(agg,'OverNegationRate'):>7} "
          f"{g(agg,'PosAcc'):>7}")

print("\n── MMLU Δ ──")
mmlu_files = {
    "Qwen rs3":    "outputs/mmlu_qwen_rs3.json",
    "Qwen rs5":    "outputs/mmlu_qwen_rs5.json",
    "Llama ret01": "outputs/mmlu_llama_ret01.json",
}
for label, path in mmlu_files.items():
    if os.path.exists(path):
        d = json.load(open(path))
        delta = round(d.get("delta", {}).get("overall_accuracy", 0) * 100, 2)
        base  = round(d.get("base", {}).get("overall_accuracy", 0) * 100, 1)
        ft    = round(d.get("finetuned", {}).get("overall_accuracy", 0) * 100, 1)
        print(f"  {label}: base={base}%  ft={ft}%  Δ={delta:+.2f}%")

print("\nWikiFact NegSuppRate / FlipAcc:")
wf_files = {
    "Qwen rs3":    "outputs/eval_qwen_rs3_wikifact.json",
    "Qwen rs5":    "outputs/eval_qwen_rs5_wikifact.json",
    "Llama ret01": "outputs/eval_llama_ret01_wikifact.json",
}
for label, path in wf_files.items():
    if os.path.exists(path):
        d = json.load(open(path))
        for v in d.values():
            agg = v.get("summary", {})
            nsr = round(agg.get("NegSuppRate", {}).get("mean", 0) * 100, 1)
            fa  = round(agg.get("FlipAcc", {}).get("mean", 0) * 100, 1)
            print(f"  {label}: NegSuppRate={nsr}%  FlipAcc={fa}%")
            break
PYEOF

echo ""
echo "Done. All results saved to outputs/eval_*_largetest.json / wikifact / mmlu_*.json"
