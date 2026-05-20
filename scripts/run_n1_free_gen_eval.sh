#!/usr/bin/env bash
# N1: Free-generation evaluation
# Compares Base / DPO / NC-SFT / MGNM on free-text generation (greedy decode)
# for suppress_target records in v2 test set.
#
# Usage: bash scripts/run_n1_free_gen_eval.sh [GPU]
#   default GPU=3

set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-3}"
MODEL_BASE="model/Qwen2.5-7B"
TEST_DATA="data/processed/validated_largetest_v2.jsonl"

echo "======================================================"
echo " N1: Free-generation evaluation (GPU cuda:${GPU})"
echo "======================================================"

# ── Base model ────────────────────────────────────────────
echo ""
echo "=== Base model ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_free_generation.py \
    --model-path  "$MODEL_BASE" \
    --input       "$TEST_DATA" \
    --output      outputs/eval_freegen_base.json \
    2>&1 | tee outputs/eval_freegen_base.log

# ── DPO model ─────────────────────────────────────────────
echo ""
echo "=== DPO model ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_free_generation.py \
    --model-path  "$MODEL_BASE" \
    --adapter-path outputs/qwen_dpo_baseline/adapter \
    --input       "$TEST_DATA" \
    --output      outputs/eval_freegen_dpo.json \
    2>&1 | tee outputs/eval_freegen_dpo.log

# ── NC-SFT model ──────────────────────────────────────────
echo ""
echo "=== NC-SFT model ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_free_generation.py \
    --model-path  "$MODEL_BASE" \
    --adapter-path outputs/qwen_nc_sft/adapter \
    --input       "$TEST_DATA" \
    --output      outputs/eval_freegen_ncsft.json \
    2>&1 | tee outputs/eval_freegen_ncsft.log

# ── MGNM model (oracle token) ─────────────────────────────
echo ""
echo "=== MGNM (oracle token) ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_free_generation.py \
    --model-path   "$MODEL_BASE" \
    --adapter-path outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3/adapter \
    --input        "$TEST_DATA" \
    --output       outputs/eval_freegen_mgnm_oracle.json \
    --use-behavior-token \
    2>&1 | tee outputs/eval_freegen_mgnm_oracle.log

# ── MGNM model (no token at test time) ───────────────────
echo ""
echo "=== MGNM (no token at test time) ==="
CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate_free_generation.py \
    --model-path   "$MODEL_BASE" \
    --adapter-path outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3/adapter \
    --input        "$TEST_DATA" \
    --output       outputs/eval_freegen_mgnm_notoken.json \
    2>&1 | tee outputs/eval_freegen_mgnm_notoken.log

# ── Summary ───────────────────────────────────────────────
echo ""
echo "======================================================"
echo " N1 SUMMARY: Free-generation FlipAcc (suppress_target)"
echo "======================================================"
python3 << 'PYEOF'
import json
from pathlib import Path

FILES = {
    'Base':                   'outputs/eval_freegen_base.json',
    'DPO':                    'outputs/eval_freegen_dpo.json',
    'NC-SFT':                 'outputs/eval_freegen_ncsft.json',
    'MGNM (oracle token)':    'outputs/eval_freegen_mgnm_oracle.json',
    'MGNM (no token)':        'outputs/eval_freegen_mgnm_notoken.json',
}

print(f"{'Model':<28} {'FlipAcc':>9} {'n':>5}")
print("-" * 45)
for label, path in FILES.items():
    p = Path(path)
    if not p.exists():
        print(f"{label:<28} {'(missing)':>9}")
        continue
    d = json.loads(p.read_text())
    acc = d.get('free_gen_flip_acc', 0) * 100
    n   = d.get('n_records', 0)
    print(f"{label:<28} {acc:>8.1f}% {n:>5}")
PYEOF
