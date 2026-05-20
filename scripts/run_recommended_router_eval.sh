#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STRICT_INPUT="${1:-data/processed/validated_largetest_v2_clean_strict.jsonl}"
CAL_INPUT="${2:-data/processed/splits/router_calibration/calibration_clean_strict.jsonl}"

STRICT_LLM="${3:-outputs/llm_router_candidate_predictions.json}"
CAL_LLM="${4:-outputs/llm_router_candidate_calibration_clean_strict_predictions.json}"

STRICT_PRED_OUT="${5:-outputs/router_recommended_strict_predictions.json}"
CAL_PRED_OUT="${6:-outputs/router_recommended_calibration_predictions.json}"

STRICT_EVAL_OUT="${7:-outputs/eval_hybrid_router_narrow_preserve_empty_select_strict.json}"
CAL_EVAL_OUT="${8:-outputs/eval_router_recommended_calibration.json}"

SUMMARY_OUT="${9:-outputs/router_recommended_summary.json}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

python scripts/hybrid_router_candidate.py \
  --input "$STRICT_INPUT" \
  --llm-predictions "$STRICT_LLM" \
  --policy recommended \
  --output "$STRICT_PRED_OUT"

if [[ "$REUSE_EXISTING" != "1" || ! -f "$STRICT_EVAL_OUT" ]]; then
  python scripts/eval_with_router.py \
    --models configs/model_config.json \
    --model-names qwen2_5_7b \
    --input "$STRICT_INPUT" \
    --predictions "$STRICT_PRED_OUT" \
    --output "$STRICT_EVAL_OUT" \
    --cache-dir outputs/score_cache_router_policy_narrow_preserve
fi

python scripts/hybrid_router_candidate.py \
  --input "$CAL_INPUT" \
  --llm-predictions "$CAL_LLM" \
  --policy recommended \
  --output "$CAL_PRED_OUT"

if [[ "$REUSE_EXISTING" != "1" || ! -f "$CAL_EVAL_OUT" ]]; then
  python scripts/eval_with_router.py \
    --models configs/model_config.json \
    --model-names qwen2_5_7b \
    --input "$CAL_INPUT" \
    --predictions "$CAL_PRED_OUT" \
    --output "$CAL_EVAL_OUT" \
    --cache-dir outputs/score_cache_router_policy_calibration
fi

python scripts/summarize_recommended_router.py \
  --strict-records "$STRICT_INPUT" \
  --strict-predictions "$STRICT_PRED_OUT" \
  --strict-eval "$STRICT_EVAL_OUT" \
  --calibration-records "$CAL_INPUT" \
  --calibration-predictions "$CAL_PRED_OUT" \
  --calibration-eval "$CAL_EVAL_OUT" \
  --output "$SUMMARY_OUT"
