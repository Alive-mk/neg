#!/usr/bin/env bash
# P12: Ablation significance tests (MGNM vs nosup, MGNM vs nopre) on v2 test set
# Run AFTER P9 evals are done (outputs/eval_*_v2.json must exist)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "======================================================"
echo " P12: Ablation Significance Tests on v2"
echo "======================================================"

for pair in \
    "qwen_mgnm qwen_nosup" \
    "qwen_mgnm qwen_nopre" \
    "llama_mgnm llama_nosup" \
    "llama_mgnm llama_nopre"; do

    mgnm_key=$(echo $pair | awk '{print $1}')
    abl_key=$(echo $pair | awk '{print $2}')

    echo ""
    echo "--- $mgnm_key vs $abl_key ---"
    python scripts/stats_significance.py \
        --mgnm    "outputs/eval_${mgnm_key}_v2.json" \
        --vanilla "outputs/eval_${abl_key}_v2.json" \
        --mgnm-key    "$mgnm_key" \
        --vanilla-key "$abl_key" \
        --output  "outputs/stats_sig_${mgnm_key}_vs_${abl_key}_v2.json"
done

echo ""
echo "======================================================"
echo " P12 DONE - significance test files written:"
echo "  outputs/stats_sig_qwen_mgnm_vs_qwen_nosup_v2.json"
echo "  outputs/stats_sig_qwen_mgnm_vs_qwen_nopre_v2.json"
echo "  outputs/stats_sig_llama_mgnm_vs_llama_nosup_v2.json"
echo "  outputs/stats_sig_llama_mgnm_vs_llama_nopre_v2.json"
echo "======================================================"
