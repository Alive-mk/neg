#!/usr/bin/env bash
# Contrastive Decoding baseline on E4 v2 test set.
# Uses Qwen2.5-7B as expert and Qwen2.5-0.5B as amateur (GPU 6, sequential).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXPERT="/data/mingkai/neg/model/Qwen2.5-7B"
AMATEUR="/data/mingkai/neg/model/Qwen2.5-0.5B"
INPUT="data/processed/validated_largetest_v2.jsonl"
OUTPUT="outputs/eval_cd_qwen.json"
LOG="outputs/eval_cd_qwen.log"

echo "=== Contrastive Decoding (Qwen-7B expert / Qwen-0.5B amateur) ==="
echo "  Expert : $EXPERT"
echo "  Amateur: $AMATEUR"
echo "  Input  : $INPUT"
echo "  Output : $OUTPUT"
echo ""

CUDA_VISIBLE_DEVICES=6 python3 scripts/eval_contrastive_decoding.py \
  --expert-path  "$EXPERT" \
  --amateur-path "$AMATEUR" \
  --input        "$INPUT" \
  --output       "$OUTPUT" \
  --cache-dir    outputs/score_cache_cd \
  --alpha        0.5 \
  2>&1 | tee "$LOG"

echo ""
echo "Done. Results in $OUTPUT"
