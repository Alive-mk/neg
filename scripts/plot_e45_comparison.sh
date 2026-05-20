#!/usr/bin/env bash
# After all E4.5 runs complete, plot the before/after mechanism comparison
set -euo pipefail
cd "$(dirname "$0")/.."

wait_mech() {
    local f="$1" label="$2"
    while [ ! -f "$f" ]; do echo "[wait] $label not done — sleeping 60s ..."; sleep 60; done
    echo "[wait] $label done."
}

wait_mech "outputs/mechanism_e45_qwen_boost_os2.json"  "Qwen E4.5"
wait_mech "outputs/mechanism_e45_llama_ret01.json"     "Llama E4.5"
wait_mech "outputs/mechanism_mistral_200_full.json"    "Mistral base mech"

echo "=== Plotting 3-way mechanism comparison (base → best_e4) ==="
python scripts/plot_mechanism_compare.py \
    --inputs \
        outputs/mechanism_qwen_200.json \
        outputs/mechanism_e45_qwen_boost_os2.json \
    --labels "Qwen (base)" "Qwen (MGNM best)" \
    --output outputs/mechanism_e45_qwen_comparison \
    2>&1 || echo "[warn] plot failed"

python scripts/plot_mechanism_compare.py \
    --inputs \
        outputs/mechanism_llama_200.json \
        outputs/mechanism_e45_llama_ret01.json \
    --labels "Llama (base)" "Llama (MGNM best)" \
    --output outputs/mechanism_e45_llama_comparison \
    2>&1 || echo "[warn] plot failed"

# Print E1 signal comparison: mid-layer neg_margin (key metric)
python3 - <<'PYEOF'
import json, os

def get_e1_peak(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    layers = d['aggregate']['e1']['per_layer']
    # peak neg_margin_mean (most negative = strongest suppression signal)
    peak = min(layers, key=lambda x: x.get('neg_margin_mean', 0))
    return peak['layer'], round(peak['neg_margin_mean'], 4)

files = {
    'Qwen base':         'outputs/mechanism_qwen_200.json',
    'Qwen MGNM best':    'outputs/mechanism_e45_qwen_boost_os2.json',
    'Llama base':        'outputs/mechanism_llama_200.json',
    'Llama MGNM best':   'outputs/mechanism_e45_llama_ret01.json',
    'Mistral base':      'outputs/mechanism_mistral_200_full.json',
    'Mistral MGNM':      'outputs/mechanism_e45_mistral_ret01.json',
}
print(f"\n{'Model':<24} {'Peak layer':>10} {'neg_margin':>12} {'direction'}")
print('-'*55)
for label, path in files.items():
    result = get_e1_peak(path)
    if result:
        layer, margin = result
        direction = 'IMPROVED↑' if 'MGNM' in label and margin < -1.0 else ''
        print(f'{label:<24} {layer:>10} {margin:>12.4f}  {direction}')
    else:
        print(f'{label:<24} {"(missing)":>10}')
PYEOF

echo "Done."
