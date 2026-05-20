#!/usr/bin/env bash
# NC-SFT eval pipeline — waits for training then runs largetest / WikiFact / MMLU
# Usage: bash scripts/eval_nc_sft.sh <GPU> <MODEL_KEY> <ADAPTER_DIR> <BASE_MODEL_PATH>
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-2}"
MODEL_KEY="${2:-qwen_nc_sft}"
ADAPTER_DIR="${3:-outputs/qwen_nc_sft}"
BASE_MODEL_PATH="${4:-model/Qwen2.5-7B}"

LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_nc_sft_${MODEL_KEY}.json"

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
base_key = 'qwen2_5_7b' if 'qwen' in '${MODEL_KEY}' else ('llama_3_1_8b' if 'llama' in '${MODEL_KEY}' else 'mistral_7b_instruct_v0_3')
base = dict(next(m for m in models if m['name'] == base_key))
entry = dict(base)
entry['name'] = '${MODEL_KEY}'
entry['adapter_path'] = '${ADAPTER_DIR}/adapter'
models_out = [m for m in models if m['name'] not in {'generator','verifier'}] + [entry]
json.dump({'models': models_out}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
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
python3 - <<PYEOF
import json, os

def g(d, m): return round(d.get(m,{}).get('mean',0)*100,1)

key = "${MODEL_KEY}"
for label, path in [("largetest", f"outputs/eval_{key}_largetest.json"),
                    ("WikiFact",  f"outputs/eval_{key}_wikifact.json")]:
    if not os.path.exists(path): continue
    d = json.load(open(path))
    s = next(iter(d.values())).get('summary', {})
    print(f"{label}: NegRank={g(s,'NegRankAcc')} FlipAcc={g(s,'FlipAcc')} "
          f"ScopeCtrl={g(s,'ScopeControlAcc')} OverNeg={g(s,'OverNegationRate')}")

mm = f"outputs/mmlu_{key}.json"
if os.path.exists(mm):
    d = json.load(open(mm))
    delta = round(d.get('delta',{}).get('overall_accuracy',0)*100,2)
    print(f"MMLU delta: {delta:+.2f}%")
PYEOF
echo "Done."
