#!/usr/bin/env bash
# MMLU capability-preservation check.
# Evaluates base model and bal_full_3ep fine-tuned model side-by-side.
# Run after Vanilla SFT training finishes so you can also add that adapter.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_mmlu_check.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BASE=/data/mingkai/neg/model/Qwen2.5-7B

echo "=== MMLU: base vs bal_full_3ep ==="
python scripts/evaluate_mmlu.py \
    --base-model-path  "$BASE" \
    --adapter-path     outputs/e4_qwen_bal_full_3ep/adapter \
    --output           outputs/mmlu_bal_full_3ep.json \
    --n-per-subject    10 \
    --device           cuda:0

echo ""
if [ -d "outputs/e4_qwen_vanilla_3ep/adapter" ]; then
    echo "=== MMLU: base vs vanilla_sft ==="
    python scripts/evaluate_mmlu.py \
        --base-model-path  "$BASE" \
        --adapter-path     outputs/e4_qwen_vanilla_3ep/adapter \
        --output           outputs/mmlu_vanilla_3ep.json \
        --n-per-subject    10 \
        --device           cuda:0
fi

echo ""
echo "=== Results summary ==="
python3 -c "
import json, os
for fname, label in [
    ('outputs/mmlu_bal_full_3ep.json', 'bal_full_3ep'),
    ('outputs/mmlu_vanilla_3ep.json', 'vanilla_3ep'),
]:
    if not os.path.exists(fname): continue
    d = json.load(open(fname))
    base_acc = d['base']['overall_accuracy']
    ft_acc   = d['finetuned']['overall_accuracy'] if d['finetuned'] else None
    delta    = d['delta']['overall_accuracy']
    print(f'{label}: base={base_acc:.4f}  ft={ft_acc:.4f}  delta={delta:+.4f}')
    if d['delta']['per_group']:
        for g, v in d['delta']['per_group'].items():
            print(f'  {g}: {v:+.4f}')
"
