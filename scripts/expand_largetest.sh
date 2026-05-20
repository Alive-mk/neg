#!/usr/bin/env bash
# Expand the large test set from 392 to 600+ items for AAAI submission.
#
# Steps:
#   1. Generate ~500 raw candidates (new seed topics, random_seed=789)
#   2. Rule-validate + LLM-verify
#   3. Deduplicate against existing train and largetest
#   4. Merge with existing validated_largetest.jsonl
#   5. Write expanded validated_largetest_v2.jsonl
#
# Usage: bash scripts/expand_largetest.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1: Generate new raw candidates (v2, seed=789) ==="
python scripts/generate_dataset.py \
    --plan    configs/dataset_plan_largetest_v2.json \
    --models  configs/model_config.json \
    --output  data/raw/largetest_v2_raw.jsonl \
    --report  outputs/generation_report_largetest_v2.json

echo ""
echo "=== Step 2: Validate (rule + LLM verifier) ==="
python scripts/validate_dataset.py \
    --input             data/raw/largetest_v2_raw.jsonl \
    --output            data/processed/validated_largetest_v2_pool.jsonl \
    --report            outputs/validation_report_largetest_v2.json \
    --models            configs/model_config.json \
    --verifier-name     verifier \
    --verifier-threshold 0.7

echo ""
echo "=== Step 3: Create test-only split (exclude train + existing largetest) ==="
python scripts/split_test_only.py \
    --input       data/processed/validated_largetest_v2_pool.jsonl \
    --output-dir  data/processed/splits/large_test_v2 \
    --report      outputs/split_report_largetest_v2.json \
    --ref-train \
        data/processed/splits/balanced/train.jsonl \
        data/processed/splits/balanced/dev.jsonl \
        data/processed/splits/merged/train.jsonl \
    --existing-tests \
        data/processed/validated_largetest.jsonl

echo ""
echo "=== Step 4: Merge v1 + v2 into expanded largetest ==="
python3 - << 'PYEOF'
import json
from pathlib import Path

v1 = [json.loads(l) for l in open("data/processed/validated_largetest.jsonl")]
v2_path = Path("data/processed/splits/large_test_v2/test.jsonl")

if v2_path.exists():
    v2 = [json.loads(l) for l in open(v2_path)]
else:
    v2 = []
    print("[merge] WARNING: v2 split not found, check generation output")

# Deduplicate by ID
all_ids = {it['id'] for it in v1}
new_items = [it for it in v2 if it['id'] not in all_ids]

merged = v1 + new_items
print(f"[merge] v1={len(v1)}, v2_new={len(new_items)}, merged={len(merged)}")

# Scope/mode breakdown
scopes = {}
modes = {}
for it in merged:
    scopes[it.get('scope_type','?')] = scopes.get(it.get('scope_type','?'),0)+1
    modes[it.get('semantic_mode','?')] = modes.get(it.get('semantic_mode','?'),0)+1
print(f"  scope: {scopes}")
print(f"  mode:  {modes}")

with open("data/processed/validated_largetest_v2.jsonl", "w") as f:
    for it in merged:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"[merge] wrote -> data/processed/validated_largetest_v2.jsonl")
PYEOF

echo ""
echo "=== Done ==="
echo "Expanded test set: data/processed/validated_largetest_v2.jsonl"
echo "Reports: outputs/generation_report_largetest_v2.json"
echo "         outputs/validation_report_largetest_v2.json"
echo "         outputs/split_report_largetest_v2.json"
