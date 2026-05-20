#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-6}"
LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_rd_refined.json"

wait_for() {
    local dir="$1" label="$2"
    while [ ! -f "${dir}/training_summary.json" ]; do
        echo "[wait] ${label} still training — sleeping 120s ..."
        sleep 120
    done
    echo "[wait] ${label} done."
}

wait_for "outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd20"  "Qwen rd20"
wait_for "outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3"  "Qwen rd30_os3"

python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
models = cfg['models']
def add_adapter(base_name, adapter_dir, new_name):
    base = next(m for m in models if m['name'] == base_name)
    e = dict(base); e['name'] = new_name
    e['adapter_path'] = f'{adapter_dir}/adapter'
    return e
models = models + [
    add_adapter('qwen2_5_7b', 'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2',      'qwen_bos2'),
    add_adapter('qwen2_5_7b', 'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd30', 'qwen_rd30'),
    add_adapter('qwen2_5_7b', 'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2_rd20', 'qwen_rd20'),
    add_adapter('qwen2_5_7b', 'outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3', 'qwen_rd30_os3'),
]
json.dump({'models': models}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
"

for NAME in qwen_rd20 qwen_rd30_os3; do
    echo "=== largetest: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
        --models "${CFG_TMP}" --input "${LARGETEST}" \
        --output "outputs/eval_${NAME}_largetest.json" \
        --cache-dir "outputs/score_cache_${NAME}" \
        --model-names "${NAME}" --neg-prefix auto \
        2>&1 | tee "outputs/eval_${NAME}_largetest.log"

    echo "=== WikiFact: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
        --models "${CFG_TMP}" --input "${WIKIFACT}" \
        --output "outputs/eval_${NAME}_wikifact.json" \
        --cache-dir "outputs/score_cache_${NAME}_wf" \
        --model-names "${NAME}" --neg-prefix "[SUPPRESS]" \
        2>&1 | tee "outputs/eval_${NAME}_wikifact.log"

    echo "=== MMLU: ${NAME} ==="
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
        --base-model-path model/Qwen2.5-7B \
        --adapter-path "outputs/${NAME/qwen_/e4_qwen_r3v5_lp15_rs3_boost_}/adapter" \
        --output "outputs/mmlu_${NAME}.json" \
        --n-per-subject 10 --seed 42 \
        2>&1 | tee "outputs/mmlu_${NAME}.log" || true
done

echo "======================================================"
echo " FINAL: rd_refined results"
echo "======================================================"
python3 - <<'PYEOF'
import json, os

files = {
    "qwen_bos2":     "outputs/eval_qwen_rs3_bos2_largetest.json",
    "qwen_rd30":     "outputs/eval_qwen_rd30_largetest.json",
    "qwen_rd20":     "outputs/eval_qwen_rd20_largetest.json",
    "qwen_rd30_os3": "outputs/eval_qwen_rd30_os3_largetest.json",
}
merged = {}
for key, path in files.items():
    if not os.path.exists(path): continue
    d = json.load(open(path))
    merged[key] = d[key] if key in d else next(iter(d.values()))

MODELS = [
    ("qwen_bos2",     "Qwen rs3_boost_os2（基线）"),
    ("qwen_rd30",     "Qwen rd30 λ_dist=3.0 os2"),
    ("qwen_rd20",     "Qwen rd20 λ_dist=2.0 os2"),
    ("qwen_rd30_os3", "Qwen rd30 λ_dist=3.0 os3"),
]

def g(d, k): return round(d.get(k,{}).get("mean",0)*100,1)
def sub(d, mode, scope, metric):
    recs = d.get("per_record",[])
    s = [r for r in recs if r.get("semantic_mode")==mode and r.get("scope_type")==scope]
    return (sum(1 for r in s if r.get(metric)), len(s)) if s else (0,0)

print(f"\n{'模型':<38} {'NegRank':>7} {'FlipAcc':>7} {'Scope':>6} {'OverNeg':>7} {'NegSupp':>7}")
print("-"*80)
for key, label in MODELS:
    if key not in merged: print(f"{label:<38} (not found)"); continue
    agg = merged[key].get("aggregate", merged[key].get("summary",{}))
    print(f"{label:<38} {g(agg,'NegRankAcc'):>7} {g(agg,'FlipAcc'):>7} "
          f"{g(agg,'ScopeControlAcc'):>6} {g(agg,'OverNegationRate'):>7} {g(agg,'NegSuppRate'):>7}")

print("\n── contrastive|in_scope rank / flip ──")
for key, label in MODELS:
    if key not in merged: continue
    n_rank, n = sub(merged[key],"contrastive_resolution","in_scope","neg_rank_correct")
    n_flip, _ = sub(merged[key],"contrastive_resolution","in_scope","flip_correct")
    if n: print(f"  {label}: rank={n_rank/n*100:.1f}% flip={n_flip/n*100:.1f}% gap={((n_rank-n_flip)/n*100):.1f}pp")

print("\n── MMLU Δ ──")
for key, label in MODELS:
    path = f"outputs/mmlu_{key}.json"
    if os.path.exists(path):
        d = json.load(open(path))
        delta = round(d.get("delta",{}).get("overall_accuracy",0)*100,2)
        print(f"  {label}: {delta:+.2f}%")
PYEOF
echo "Done."
