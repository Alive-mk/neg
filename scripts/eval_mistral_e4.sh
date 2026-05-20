#!/usr/bin/env bash
# Evaluate Mistral MGNM training + E4.5 mechanism comparison
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-7}"
LARGETEST="data/processed/splits/large_test/test.jsonl"
WIKIFACT="data/external/wikifact_neg_patched.jsonl"
CFG_TMP="/tmp/model_cfg_mistral.json"

wait_for() {
    local dir="$1" label="$2"
    while [ ! -f "${dir}/training_summary.json" ]; do
        echo "[wait] ${label} still training — sleeping 120s ..."; sleep 120
    done
    echo "[wait] ${label} done."
}

wait_for "outputs/mistral_r3v5_lp15_ret01" "Mistral ret01"

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
    add_adapter('qwen2_5_7b',            'outputs/e4_qwen_r3v5_lp15_rs3_boost_os2', 'qwen_best'),
    add_adapter('llama_3_1_8b',          'outputs/llama_r3v5_lp15_ret01',           'llama_best'),
    add_adapter('mistral_7b_instruct_v0_3', 'outputs/mistral_r3v5_lp15_ret01',      'mistral_ret01'),
]
json.dump({'models': models}, open('${CFG_TMP}', 'w'), indent=2)
print('Config written.')
"

for NAME in mistral_ret01; do
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
        --base-model-path /data/share/neg/model/Mistral-7B-Instruct-v0.3 \
        --adapter-path "outputs/mistral_r3v5_lp15_ret01/adapter" \
        --output "outputs/mmlu_${NAME}.json" \
        --n-per-subject 10 --seed 42 \
        2>&1 | tee "outputs/mmlu_${NAME}.log"
done

# E4.5 for Mistral (post-training mechanism check)
echo "=== E4.5: Mistral post-training mechanism ==="
python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
models = cfg['models']
base = next(m for m in models if m['name'] == 'mistral_7b_instruct_v0_3')
e = dict(base); e['name'] = 'mistral_ret01'
e['adapter_path'] = 'outputs/mistral_r3v5_lp15_ret01/adapter'
json.dump({'models': [e]}, open('/tmp/cfg_e45_mistral.json', 'w'), indent=2)
"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/analyze_mechanism.py \
    --models /tmp/cfg_e45_mistral.json \
    --model-name mistral_ret01 \
    --input data/manual/mech_probes.jsonl \
    --output outputs/mechanism_e45_mistral_ret01.json \
    2>&1 | tee outputs/mechanism_e45_mistral_ret01.log

echo "=== FINAL Mistral Results ==="
python3 - <<'PYEOF'
import json, os

def g(d, k): return round(d.get(k,{}).get('mean',0)*100,1)

files = {
    'Mistral base (no training)': ('outputs/mechanism_mistral_200_full.json', None, None),
    'Qwen best':   (None, 'outputs/eval_qwen_rs3_bos2_largetest.json', 'qwen_rs3_bos2'),
    'Llama best':  (None, 'outputs/eval_llama_ret01_largetest.json', None),
    'Mistral ret01': (None, 'outputs/eval_mistral_ret01_largetest.json', None),
}
for label, (mech_path, eval_path, key) in files.items():
    if eval_path and os.path.exists(eval_path):
        d = json.load(open(eval_path))
        v = d[key] if (key and key in d) else next(iter(d.values()))
        agg = v.get('aggregate', v.get('summary', {}))
        print(f'{label}: NegRank={g(agg,"NegRankAcc")} FlipAcc={g(agg,"FlipAcc")} Scope={g(agg,"ScopeControlAcc")}')
PYEOF

echo "Done."
