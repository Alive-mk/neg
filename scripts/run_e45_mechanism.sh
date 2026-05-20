#!/usr/bin/env bash
# E4.5: Post-mitigation mechanism check for the best model (bal_full_3ep).
#
# Runs E1/E2/E3 probes on the fine-tuned model using the same 200 manual
# probes used for the baseline analysis, so results are directly comparable.
#
# Usage:
#   bash scripts/run_e45_mechanism.sh
#
# Output:
#   outputs/mechanism_e4_bal_full_3ep.json
#   outputs/mechanism_e4_bal_full_3ep.log
#   outputs/mechanism_e4_comparison_full.{png,pdf}  (updated 3-way plot)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_CFG=/tmp/model_config_bal_full.json

# Build config if it doesn't exist in this shell session
python3 -c "
import json
cfg = {
  'models': [{
    'name': 'qwen2_5_7b_e4_bal_full',
    'mode': 'hf_local',
    'model_path': '/data/mingkai/neg/model/Qwen2.5-7B',
    'adapter_path': 'outputs/e4_qwen_bal_full_3ep/adapter',
    'device_map': 'cuda:0',
    'torch_dtype': 'float16',
    'trust_remote_code': False,
  }]
}
json.dump(cfg, open('$MODEL_CFG', 'w'), indent=2)
"
# CUDA_VISIBLE_DEVICES=1 makes physical GPU 1 appear as cuda:0 inside this process

echo "=== E4.5: Mechanism analysis for bal_full_3ep (200 probes) ==="
CUDA_VISIBLE_DEVICES=1 python scripts/analyze_mechanism.py \
    --models      "$MODEL_CFG" \
    --model-name  qwen2_5_7b_e4_bal_full \
    --input       data/manual/mech_probes.jsonl \
    --output      outputs/mechanism_e4_bal_full_3ep.json \
    --max-records 200 \
    2>&1 | tee outputs/mechanism_e4_bal_full_3ep.log

echo ""
echo "=== Generating 3-way comparison plot ==="
CUDA_VISIBLE_DEVICES="" python scripts/plot_mechanism_compare.py \
    --inputs \
        outputs/mechanism_qwen_200.json \
        outputs/mechanism_e4_core_5ep.json \
        outputs/mechanism_e4_bal_full_3ep.json \
    --labels "Baseline" "core_5ep" "bal_full_3ep" \
    --output outputs/mechanism_e4_comparison_full \
    2>&1 || echo "[warn] plot_mechanism_compare failed — check outputs manually"

echo ""
echo "=== Summary ==="
python3 -c "
import json

def show(path, label):
    d = json.load(open(path))
    agg = d['aggregate']
    e1_last = agg['e1']['per_layer'][-1]
    e2 = agg['e2']
    e3_last = agg['e3']['per_layer'][-1]
    top_heads = [h['head'] for h in agg['e2']['top_heads'][:4]]
    print(f'{label}:')
    print(f'  E1 neg_margin_last: {e1_last[\"neg_margin_mean\"]:.4f}')
    print(f'  E2 top heads: {top_heads}')
    print(f'  E3 pos_recovery_last: {e3_last[\"positive_recovery_rate\"]:.3f}')
    print()

show('outputs/mechanism_qwen_200.json',        'Baseline')
show('outputs/mechanism_e4_core_5ep.json',     'core_5ep')
show('outputs/mechanism_e4_bal_full_3ep.json', 'bal_full_3ep')
"
