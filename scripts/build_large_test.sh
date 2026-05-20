#!/usr/bin/env bash
# Build a large held-out test set for AAAI submission.
#
# Steps:
#   1. Generate raw candidates (~1500) with new seed topics and random_seed=456
#   2. Rule-validate + LLM-verify the raw candidates
#   3. Create test-only split with entity/family holdout report
#
# Usage:
#   export NEG_OPENAI_API_KEY='your_key'
#   bash scripts/build_large_test.sh
#
# Outputs:
#   data/raw/largetest_raw.jsonl
#   data/processed/validated_largetest.jsonl
#   data/processed/splits/large_test/test.jsonl
#   outputs/generation_report_largetest.json
#   outputs/validation_report_largetest.json
#   outputs/split_report_largetest.json

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: Generate raw candidates ==="
python scripts/generate_dataset.py \
    --plan    configs/dataset_plan_largetest.json \
    --models  configs/model_config.json \
    --output  data/raw/largetest_raw.jsonl \
    --report  outputs/generation_report_largetest.json

echo ""
echo "=== Step 2: Validate (rule + LLM verifier) ==="
python scripts/validate_dataset.py \
    --input             data/raw/largetest_raw.jsonl \
    --output            data/processed/validated_largetest.jsonl \
    --report            outputs/validation_report_largetest.json \
    --models            configs/model_config.json \
    --verifier-name     verifier \
    --verifier-threshold 0.7

echo ""
echo "=== Step 3: Create test-only split with holdout check ==="
python scripts/split_test_only.py \
    --input       data/processed/validated_largetest.jsonl \
    --output-dir  data/processed/splits/large_test \
    --report      outputs/split_report_largetest.json \
    --ref-train \
        data/processed/splits/balanced/train.jsonl \
        data/processed/splits/balanced/dev.jsonl \
        data/processed/splits/merged/train.jsonl \
    --existing-tests \
        data/processed/splits/balanced/test.jsonl \
        data/processed/splits/proposal_600/test.jsonl

echo ""
echo "=== Done ==="
echo "Test set: data/processed/splits/large_test/test.jsonl"
echo "Report:   outputs/split_report_largetest.json"
