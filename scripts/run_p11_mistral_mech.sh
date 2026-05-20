#!/usr/bin/env bash
# P11: Mechanism analysis for mistral_lr5e5_os1, then update cross-model figure
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=/tmp/model_cfg_mistral_lr5e5_mech.json

python3 -c "
import json
cfg = json.load(open('configs/model_config.json'))
base = next(m for m in cfg['models'] if m['name'] == 'mistral_7b_instruct_v0_3')
entry = dict(base)
entry['name'] = 'mistral_lr5e5_os1'
entry['adapter_path'] = 'outputs/mistral_lr5e5_os1/adapter'
json.dump({'models': cfg['models'] + [entry]}, open('$CFG', 'w'), indent=2)
print('[P11] Config written:', '$CFG')
"

echo "[P11] Running mechanism analysis for mistral_lr5e5_os1 on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python scripts/analyze_mechanism.py \
    --models "$CFG" \
    --model-name mistral_lr5e5_os1 \
    --input data/manual/mech_probes.jsonl \
    --output outputs/mechanism_e45_mistral_lr5e5_os1.json \
    2>&1 | tee outputs/mechanism_e45_mistral_lr5e5_os1.log

echo "[P11] Mechanism analysis done."

echo "[P11] Generating updated cross-model figure..."
python scripts/plot_mechanism_compare.py \
    --inputs \
        outputs/mechanism_qwen_200.json \
        outputs/mechanism_e45_qwen_rd30_os3.json \
        outputs/mechanism_llama_200.json \
        outputs/mechanism_e45_llama_rd10_os2.json \
        outputs/mechanism_mistral_200_full.json \
        outputs/mechanism_e45_mistral_lr5e5_os1.json \
    --labels \
        "Qwen (base)" "Qwen (MGNM)" \
        "Llama (base)" "Llama (MGNM)" \
        "Mistral (base)" "Mistral (MGNM)" \
    --output outputs/mechanism_cross_model_final \
    2>&1 || echo "[warn] plot failed — check mechanism_e45_mistral_lr5e5_os1.json manually"

echo ""
echo "======================================================"
echo " P11 MECHANISM SUMMARY"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

def show(label, path):
    if not os.path.exists(path):
        print(f"  {label}: MISSING")
        return
    d = json.load(open(path))
    agg = d['aggregate']
    e1 = agg['e1']['per_layer']
    peak = min(e1, key=lambda x: x.get('neg_margin_mean', 0))
    e3_last = agg['e3']['per_layer'][-1]
    print(f"  {label}:")
    print(f"    E1 peak neg_margin: layer={peak['layer']} val={peak['neg_margin_mean']:.4f}")
    print(f"    E3 pos_recovery (last): {e3_last['positive_recovery_rate']:.3f}")

show("Mistral base",           "outputs/mechanism_mistral_200_full.json")
show("Mistral MGNM lr5e5_os1","outputs/mechanism_e45_mistral_lr5e5_os1.json")
PYEOF

echo "Done."
