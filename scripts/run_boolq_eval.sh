#!/usr/bin/env bash
# BoolQ negation eval: Qwen (GPU 3) and Llama (GPU 4) run in parallel.
# Each group evaluates on: negation subset (209) and full validation (3270).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Step 1: Prepare BoolQ data ==="
python3 scripts/prepare_boolq_negation.py

QWEN_MODELS="qwen_base qwen_dpo qwen_nc_sft qwen_mgnm"
LLAMA_MODELS="llama_base llama_dpo llama_nc_sft llama_mgnm"
CFG="configs/model_config_boolq.json"
CACHE="outputs/score_cache_boolq"

run_group() {
  local group="$1"; local models="$2"; local gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval_boolq_negation.py \
    --models-config "$CFG" \
    --input data/boolq_negation.jsonl \
    --output "outputs/eval_boolq_neg_${group}.json" \
    --cache-dir "$CACHE" \
    --model-names $models \
    >> "outputs/eval_boolq_${group}.log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval_boolq_negation.py \
    --models-config "$CFG" \
    --input data/boolq_full_val.jsonl \
    --output "outputs/eval_boolq_full_${group}.json" \
    --cache-dir "$CACHE" \
    --model-names $models \
    >> "outputs/eval_boolq_${group}.log" 2>&1
}

echo ""
echo "=== Step 2: Launch evals (Qwen→GPU3, Llama→GPU4, parallel) ==="

run_group qwen "$QWEN_MODELS" 3 &
PID_QWEN=$!
echo "  Qwen group started (PID $PID_QWEN)"

run_group llama "$LLAMA_MODELS" 4 &
PID_LLAMA=$!
echo "  Llama group started (PID $PID_LLAMA)"

echo "  Waiting..."
wait $PID_QWEN && echo "  Qwen done ✓" || echo "  Qwen FAILED (see outputs/eval_boolq_qwen.log)"
wait $PID_LLAMA && echo "  Llama done ✓" || echo "  Llama FAILED (see outputs/eval_boolq_llama.log)"

echo ""
echo "=== Step 3: PMI calibration (corrects prior shift in MGNM/DPO) ==="

CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_boolq_pmi.py \
  --models-config "$CFG" \
  --input data/boolq_negation.jsonl \
  --output outputs/eval_boolq_pmi_qwen.json \
  --cache-dir "$CACHE" \
  --model-names $QWEN_MODELS \
  >> outputs/eval_boolq_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_boolq_pmi.py \
  --models-config "$CFG" \
  --input data/boolq_full_val.jsonl \
  --output outputs/eval_boolq_full_pmi_qwen.json \
  --cache-dir "$CACHE" \
  --model-names $QWEN_MODELS \
  >> outputs/eval_boolq_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python3 scripts/eval_boolq_pmi.py \
  --models-config "$CFG" \
  --input data/boolq_negation.jsonl \
  --output outputs/eval_boolq_pmi_llama.json \
  --cache-dir "$CACHE" \
  --model-names $LLAMA_MODELS \
  >> outputs/eval_boolq_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python3 scripts/eval_boolq_pmi.py \
  --models-config "$CFG" \
  --input data/boolq_full_val.jsonl \
  --output outputs/eval_boolq_full_pmi_llama.json \
  --cache-dir "$CACHE" \
  --model-names $LLAMA_MODELS \
  >> outputs/eval_boolq_llama.log 2>&1 &

wait
echo "  PMI calibration done ✓"

echo ""
echo "=== Step 4: Print summary tables ==="
python3 scripts/print_boolq_results.py
