#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${1:-data/processed/validated_largetest_v2_clean.jsonl}"
OUTPUT="${2:-outputs/llm_router_candidate_predictions.json}"
POSTPROCESS_POLICY="${3:-}"
N_WORKERS="${N_WORKERS:-5}"
CHUNK_DIR="outputs/router_candidate_chunks"
RAW_OUTPUT="${CHUNK_DIR}/merged_raw_predictions.json"

TOTAL=$(wc -l < "$INPUT")
CHUNK=$((TOTAL / N_WORKERS + 1))
mkdir -p "$CHUNK_DIR"

for i in $(seq 0 $((N_WORKERS - 1))); do
    START=$((i * CHUNK))
    END=$(((i + 1) * CHUNK))
    CHUNK_INPUT="${CHUNK_DIR}/records_${i}.jsonl"
    CHUNK_OUTPUT="${CHUNK_DIR}/preds_${i}.json"

    python - "$INPUT" "$CHUNK_INPUT" "$START" "$END" <<'PY'
import sys
src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = open(src, encoding="utf-8").read().splitlines()
open(dst, "w", encoding="utf-8").write("\n".join(lines[start:end]) + ("\n" if lines[start:end] else ""))
PY

    python scripts/llm_router_candidate.py \
        --input "$CHUNK_INPUT" \
        --output "$CHUNK_OUTPUT" \
        --resume \
        --sleep 0.02 > "${CHUNK_DIR}/worker_${i}.log" 2>&1 &
    echo "Started worker $i records [$START,$END) pid=$!"
done

wait

python - "$RAW_OUTPUT" "$CHUNK_DIR" <<'PY'
import glob, json, sys
out_path, chunk_dir = sys.argv[1], sys.argv[2]
merged = {}
for path in sorted(glob.glob(f"{chunk_dir}/preds_*.json")):
    merged.update(json.load(open(path, encoding="utf-8")))
json.dump(merged, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Saved {len(merged)} predictions to {out_path}")
PY

if [[ -n "$POSTPROCESS_POLICY" ]]; then
    python scripts/hybrid_router_candidate.py \
        --input "$INPUT" \
        --llm-predictions "$RAW_OUTPUT" \
        --policy "$POSTPROCESS_POLICY" \
        --output "$OUTPUT"
else
    cp "$RAW_OUTPUT" "$OUTPUT"
fi
