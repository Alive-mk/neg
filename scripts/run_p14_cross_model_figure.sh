#!/usr/bin/env bash
# P14: Generate updated cross-model mechanism figure after P11 (Mistral+Llama mech done)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[P14] Generating updated cross-model mechanism figure..."
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
    2>&1

echo ""
echo "======================================================"
echo " P14 MECHANISM SUMMARY (all 3 models)"
echo "======================================================"
python3 - << 'PYEOF'
import json, os

def show(label, path):
    if not os.path.exists(path):
        print(f"  {label:30s}: MISSING {path}")
        return
    d = json.load(open(path))
    agg = d['aggregate']
    e1 = agg['e1']['per_layer']
    peak = min(e1, key=lambda x: x.get('neg_margin_mean', 0))
    e3_last = agg['e3']['per_layer'][-1]
    print(f"  {label:30s} E1_peak=layer{peak['layer']:2d},{peak['neg_margin_mean']:7.4f}  E3_last={e3_last['positive_recovery_rate']:.3f}")

print("\n  Model                          E1_peak              E3_last")
print("  " + "-"*65)
show("Qwen base",               "outputs/mechanism_qwen_200.json")
show("Qwen MGNM",               "outputs/mechanism_e45_qwen_rd30_os3.json")
show("Llama base",              "outputs/mechanism_llama_200.json")
show("Llama MGNM (rd10_os2)",   "outputs/mechanism_e45_llama_rd10_os2.json")
show("Mistral base",            "outputs/mechanism_mistral_200_full.json")
show("Mistral MGNM (lr5e5)",    "outputs/mechanism_e45_mistral_lr5e5_os1.json")
PYEOF

echo ""
echo "Output: outputs/mechanism_cross_model_final.{png,pdf}"
echo "Done."
