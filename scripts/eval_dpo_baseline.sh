#!/usr/bin/env bash
# DPO baseline evaluation pipeline.
# Usage: bash scripts/eval_dpo_baseline.sh <GPU> <MODEL_KEY> <ADAPTER_DIR> <BASE_MODEL_PATH>
# Example: bash scripts/eval_dpo_baseline.sh 2 qwen_dpo outputs/qwen_dpo_baseline model/Qwen2.5-7B
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
MODEL_KEY="${2:-qwen_dpo}"
ADAPTER_DIR="${3:-outputs/qwen_dpo_baseline}"
BASE_MODEL_PATH="${4:-model/Qwen2.5-7B}"

LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_dpo_${MODEL_KEY}.json"

wait_for() {
    local dir="$1" label="$2"
    while [ ! -f "${dir}/training_summary.json" ]; do
        echo "[wait] ${label} still training — sleeping 120s ..."
        sleep 120
    done
    echo "[wait] ${label} done."
}

wait_for "${ADAPTER_DIR}" "${MODEL_KEY}"

python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
models = cfg['models']
base_name = '$(echo ${BASE_MODEL_PATH} | grep -oP "[^/]+" | tail -1)'
base_entry = None
for m in models:
    if m.get('model_path','').endswith(base_name) or '${BASE_MODEL_PATH}' in m.get('model_path',''):
        base_entry = dict(m)
        break
if base_entry is None:
    # fallback: use qwen2_5_7b or llama_3_1_8b based on key
    key = '${MODEL_KEY}'
    if 'llama' in key:
        base_entry = next(m for m in models if m['name'] == 'llama_3_1_8b')
    elif 'mistral' in key:
        base_entry = next(m for m in models if m['name'] == 'mistral_7b_instruct_v0_3')
    else:
        base_entry = next(m for m in models if m['name'] == 'qwen2_5_7b')
    base_entry = dict(base_entry)
entry = dict(base_entry)
entry['name'] = '${MODEL_KEY}'
entry['adapter_path'] = '${ADAPTER_DIR}/adapter'
models_out = [m for m in models if m['name'] not in {'generator','verifier'}] + [entry]
json.dump({'models': models_out}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written: ${CFG_TMP}')
"

echo "=== largetest: ${MODEL_KEY} ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models "${CFG_TMP}" --input "${LARGETEST}" \
    --output "outputs/eval_${MODEL_KEY}_largetest.json" \
    --cache-dir "outputs/score_cache_${MODEL_KEY}" \
    --model-names "${MODEL_KEY}" --neg-prefix auto \
    2>&1 | tee "outputs/eval_${MODEL_KEY}_largetest.log"

echo "=== WikiFact: ${MODEL_KEY} ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_models.py \
    --models "${CFG_TMP}" --input "${WIKIFACT}" \
    --output "outputs/eval_${MODEL_KEY}_wikifact.json" \
    --cache-dir "outputs/score_cache_${MODEL_KEY}_wf" \
    --model-names "${MODEL_KEY}" --neg-prefix "[SUPPRESS]" \
    2>&1 | tee "outputs/eval_${MODEL_KEY}_wikifact.log"

echo "=== MMLU: ${MODEL_KEY} ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_mmlu.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --adapter-path "${ADAPTER_DIR}/adapter" \
    --output "outputs/mmlu_${MODEL_KEY}.json" \
    --n-per-subject 10 --seed 42 \
    2>&1 | tee "outputs/mmlu_${MODEL_KEY}.log" || true

echo "======================================================"
echo " FINAL: ${MODEL_KEY} DPO results"
echo "======================================================"
python3 - <<PYEOF
import json, os

def g(d, k): return round(d.get(k,{}).get("mean",0)*100,1)

key = "${MODEL_KEY}"
lt_path = f"outputs/eval_{key}_largetest.json"
wf_path = f"outputs/eval_{key}_wikifact.json"
mm_path = f"outputs/mmlu_{key}.json"

if os.path.exists(lt_path):
    d = json.load(open(lt_path))
    agg = next(iter(d.values())).get("summary", {})
    print(f"\n=== {key} largetest ===")
    for metric in ["NegSuppRate","NegRankAcc","FlipAcc","ScopeControlAcc","OverNegationRate"]:
        print(f"  {metric}: {g(agg, metric)}%")
else:
    print(f"largetest output not found: {lt_path}")

if os.path.exists(wf_path):
    d = json.load(open(wf_path))
    agg = next(iter(d.values())).get("summary", {})
    print(f"\n=== {key} WikiFact FlipAcc ===")
    print(f"  FlipAcc: {g(agg,'FlipAcc')}%")

if os.path.exists(mm_path):
    d = json.load(open(mm_path))
    delta = round(d.get("delta",{}).get("overall_accuracy",0)*100,2)
    print(f"\n=== {key} MMLU ===")
    print(f"  delta: {delta:+.2f}%")
PYEOF
echo "Done."
