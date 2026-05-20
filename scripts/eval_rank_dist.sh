#!/usr/bin/env bash
# Evaluate rank-dist ablation models:
#   - e4_qwen_r3v5_lp15_rs3_boost_os2_rd15  (λ_rank_dist=1.5)
#   - e4_qwen_r3v5_lp15_rs3_boost_os2_rd30  (λ_rank_dist=3.0)
#   - llama_r3v5_lp15_ret01_rd15            (λ_rank_dist=1.5)
#
# Usage: bash scripts/eval_rank_dist.sh [GPU_ID]

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-6}"
LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_rank_dist.json"

echo "======================================================"
echo " eval_rank_dist.sh  GPU=cuda:${GPU}"
echo "======================================================"

wait_for() {
    local dir="$1" label="$2"
    while [ ! -f "${dir}/training_summary.json" ]; do
        echo "[wait] ${label} still training — sleeping 120s ..."
        sleep 120
    done
    echo "[wait] ${label} done."
}

wait_for "outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd15" "Qwen rd15"
wait_for "outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd30" "Qwen rd30"
wait_for "outputs/llama_r3v5_lp15_ret01_rd15"           "Llama rd15"

echo "Building model config ..."
python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
models = cfg['models']

def add_adapter(base_name, adapter_dir, new_name):
    base = next(m for m in models if m['name'] == base_name)
    e = dict(base)
    e['name'] = new_name
    e['adapter_path'] = f'{adapter_dir}/adapter'
    return e

models = models + [
    add_adapter('qwen2_5_7b',   'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2',      'qwen_boost_os2'),
    add_adapter('qwen2_5_7b',   'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd15', 'qwen_rd15'),
    add_adapter('qwen2_5_7b',   'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd30', 'qwen_rd30'),
    add_adapter('llama_3_1_8b', 'outputs/llama_r3v5_lp15_ret01',                'llama_ret01'),
    add_adapter('llama_3_1_8b', 'outputs/llama_r3v5_lp15_ret01_rd15',           'llama_rd15'),
]
json.dump({'models': models}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
"

for NAME in qwen_rd15 qwen_rd30 llama_rd15; do
    echo ""
    echo "=== largetest: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
        --models "${CFG_TMP}" \
        --input "${LARGETEST}" \
        --output "outputs/eval_${NAME}_largetest.json" \
        --cache-dir "outputs/score_cache_${NAME}" \
        --model-names "${NAME}" \
        --neg-prefix auto \
        2>&1 | tee "outputs/eval_${NAME}_largetest.log"

    echo "=== WikiFact: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
        --models "${CFG_TMP}" \
        --input "${WIKIFACT}" \
        --output "outputs/eval_${NAME}_wikifact.json" \
        --cache-dir "outputs/score_cache_${NAME}_wf" \
        --model-names "${NAME}" \
        --neg-prefix "[SUPPRESS]" \
        2>&1 | tee "outputs/eval_${NAME}_wikifact.log"
done

# MMLU
for NAME_BASE in "qwen_rd15 e4_qwen_r3v5_lp15_rs3_boost_os2_rd15 model/Qwen2.5-7B" \
                 "qwen_rd30 e4_qwen_r3v5_lp15_rs3_boost_os2_rd30 model/Qwen2.5-7B" \
                 "llama_rd15 llama_r3v5_lp15_ret01_rd15 /data/share/neg/model/Meta-Llama-3.1-8B"; do
    NAME=$(echo $NAME_BASE | awk '{print $1}')
    OUTDIR=$(echo $NAME_BASE | awk '{print $2}')
    BMODEL=$(echo $NAME_BASE | awk '{print $3}')
    echo ""
    echo "=== MMLU: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
        --base-model-path "${BMODEL}" \
        --adapter-path "outputs/${OUTDIR}/adapter" \
        --output "outputs/mmlu_${NAME}.json" \
        --n-per-subject 10 --seed 42 \
        2>&1 | tee "outputs/mmlu_${NAME}.log"
done

echo ""
echo "======================================================"
echo " FINAL: rank-dist ablation results"
echo "======================================================"
python3 - <<'PYEOF'
import json, os

baselines = {
    "qwen_boost_os2": "outputs/eval_qwen_rs3_boost_os2_largetest.json",
    "llama_ret01":    "outputs/eval_llama_ret01_largetest.json",
}
new_runs = {
    "qwen_rd15": "outputs/eval_qwen_rd15_largetest.json",
    "qwen_rd30": "outputs/eval_qwen_rd30_largetest.json",
    "llama_rd15": "outputs/eval_llama_rd15_largetest.json",
}

merged = {}
for key, path in {**baselines, **new_runs}.items():
    if not os.path.exists(path): continue
    d = json.load(open(path))
    merged[key] = d[key] if key in d else next(iter(d.values()))

MODELS = [
    ("qwen_boost_os2", "Qwen rs3_boost_os2 (baseline)"),
    ("qwen_rd15",      "Qwen rd15 (λ_dist=1.5)"),
    ("qwen_rd30",      "Qwen rd30 (λ_dist=3.0)"),
    ("llama_ret01",    "Llama ret01 (baseline)"),
    ("llama_rd15",     "Llama rd15 (λ_dist=1.5)"),
]

def g(d, k): return round(d.get(k, {}).get("mean", 0) * 100, 1)

print(f"\n{'Model':<38} {'NegRank':>7} {'FlipAcc':>7} {'ScopeCtrl':>9} {'OverNeg':>7} {'NegSupp':>7}")
print("-" * 80)
for key, label in MODELS:
    if key not in merged: print(f"{label:<38} (not found)"); continue
    agg = merged[key].get("aggregate", merged[key].get("summary", {}))
    print(f"{label:<38} {g(agg,'NegRankAcc'):>7} {g(agg,'FlipAcc'):>7} "
          f"{g(agg,'ScopeControlAcc'):>9} {g(agg,'OverNegationRate'):>7} {g(agg,'NegSuppRate'):>7}")

# per-mode breakdown: contrastive|in_scope rank vs flip
print("\n── contrastive|in_scope breakdown ──")
for key, label in MODELS:
    if key not in merged: continue
    records = merged[key].get("per_record", [])
    cr = [r for r in records if r.get("semantic_mode") == "contrastive_resolution"
          and r.get("scope_type") == "in_scope"]
    if cr:
        rank = sum(1 for r in cr if r.get("neg_rank_correct"))
        flip = sum(1 for r in cr if r.get("flip_correct"))
        print(f"  {label}: rank={rank}/{len(cr)}={round(rank/len(cr)*100,1)}%  "
              f"flip={flip}/{len(cr)}={round(flip/len(cr)*100,1)}%  gap={round((rank-flip)/len(cr)*100,1)}pp")

print("\n── exclusive_choice|in_scope breakdown ──")
for key, label in MODELS:
    if key not in merged: continue
    records = merged[key].get("per_record", [])
    ec = [r for r in records if r.get("semantic_mode") == "exclusive_choice"
          and r.get("scope_type") == "in_scope"]
    if ec:
        rank = sum(1 for r in ec if r.get("neg_rank_correct"))
        flip = sum(1 for r in ec if r.get("flip_correct"))
        print(f"  {label}: rank={rank}/{len(ec)}={round(rank/len(ec)*100,1)}%  "
              f"flip={flip}/{len(ec)}={round(flip/len(ec)*100,1)}%")

print("\n── MMLU Δ ──")
for key, label in MODELS:
    path = f"outputs/mmlu_{key}.json"
    if os.path.exists(path):
        d = json.load(open(path))
        delta = round(d.get("delta", {}).get("overall_accuracy", 0) * 100, 2)
        print(f"  {label}: Δ={delta:+.2f}%")

print("\n── WikiFact NegSuppRate / FlipAcc ──")
for key, label in MODELS:
    path = f"outputs/eval_{key}_wikifact.json"
    if os.path.exists(path):
        d = json.load(open(path))
        for v in d.values():
            s = v.get("summary", {})
            nsr = round(s.get("NegSuppRate", {}).get("mean", 0) * 100, 1)
            fa  = round(s.get("FlipAcc", {}).get("mean", 0) * 100, 1)
            print(f"  {label}: NegSuppRate={nsr}%  FlipAcc={fa}%")
            break
PYEOF

echo "Done."
