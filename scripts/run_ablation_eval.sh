#!/usr/bin/env bash
# Evaluate ablation models on large_test and print comparison table.
# Run after train_ablation_nosup and train_ablation_nopre complete.
#
# Usage:
#   bash scripts/run_ablation_eval.sh 2>&1 | tee outputs/ablation_eval.log

set -euo pipefail
cd "$(dirname "$0")/.."

TEST_DATA="data/processed/splits/large_test/test.jsonl"

echo "=== Building ablation model configs ==="
python3 -c "
import json

base = [m for m in json.load(open('configs/model_config.json'))['models']
        if m['name'] == 'qwen2_5_7b']

def make_adapted(adapter_dir, name):
    m = json.load(open(f'{adapter_dir}/model_config_with_adapter.json'))['models'][-1]
    m['name'] = name
    return m

models = base + [
    make_adapted('outputs/e4_qwen_lsup2',          'mgnm_lsup2'),
    make_adapted('outputs/e4_qwen_ablation_nosup',  'mgnm_nosup'),
    make_adapted('outputs/e4_qwen_ablation_nopre',  'mgnm_nopre'),
    make_adapted('outputs/e4_qwen_vanilla_3ep',     'vanilla_sft'),
]
json.dump({'models': models}, open('/tmp/model_cfg_ablation.json','w'), indent=2)
print('Models:', [m['name'] for m in models])
"

echo ""
echo "=== Evaluating on large_test (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_models.py \
    --models     /tmp/model_cfg_ablation.json \
    --input      "$TEST_DATA" \
    --output     outputs/eval_ablation_largetest.json \
    --cache-dir  outputs/score_cache_ablation \
    --model-names qwen2_5_7b mgnm_lsup2 mgnm_nosup mgnm_nopre vanilla_sft \
    2>&1 | tee outputs/eval_ablation_largetest.log

echo ""
echo "=== Ablation Comparison Table ==="
python3 - <<'PYEOF'
import json, os

def fmt(d, key, label):
    if key not in d:
        return f'{label:<24} NOT FOUND'
    s = d[key]['summary']
    def g(k): return s.get(k, {}).get('mean', 0) * 100
    return (f'{label:<24} PosAcc={g("PosAcc"):.1f}%  '
            f'NegSupp={g("NegSuppRate"):.1f}%  '
            f'Flip={g("FlipAcc"):.1f}%  '
            f'ScopeCtrl={g("ScopeControlAcc"):.1f}%  '
            f'DoubleNeg={g("DoubleNegationAcc"):.1f}%  '
            f'OverNeg={g("OverNegationRate"):.1f}%')

if not os.path.exists('outputs/eval_ablation_largetest.json'):
    print('File not found')
else:
    d = json.load(open('outputs/eval_ablation_largetest.json'))
    rows = [
        ('qwen2_5_7b',   'Base (Qwen2.5-7B)'),
        ('vanilla_sft',  'Vanilla SFT'),
        ('mgnm_nosup',   'MGNM w/o L_sup'),
        ('mgnm_nopre',   'MGNM w/o L_preserve'),
        ('mgnm_lsup2',   'MGNM lsup2 (full)'),
    ]
    for key, label in rows:
        print(fmt(d, key, label))
PYEOF
