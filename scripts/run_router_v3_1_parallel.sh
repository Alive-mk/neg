#!/bin/bash
# Run llm_router_v3_1 in 5 parallel workers
# Usage: ./run_router_v3_1_parallel.sh

INPUT="data/processed/validated_largetest_v2.jsonl"
TOTAL=$(wc -l < "$INPUT")
N_WORKERS=5
CHUNK=$((TOTAL / N_WORKERS + 1))

mkdir -p outputs/router_v3_1_chunks

SYSTEM=$(python3 -c "
import sys
sys.path.insert(0, 'scripts')
import llm_router_v3_1
print(llm_router_v3_1.SYSTEM)
")

for i in $(seq 0 $((N_WORKERS-1))); do
    START=$((i * CHUNK))
    END=$(((i+1) * CHUNK))
    OUTPUT="outputs/router_v3_1_chunks/preds_${i}.json"

    python3 -c "
import json, sys, os, time, urllib.request
sys.path.insert(0, 'scripts')
import llm_router_v3_1

API_URL = 'https://api.ai-gaochao.cn/v1/chat/completions'
API_KEY = open('/tmp/.neg_api_key').read().strip()

records = [json.loads(l) for l in open('$INPUT')]
records = records[$START:$END]
out_path = '$OUTPUT'

preds = {}
if os.path.exists(out_path):
    preds = json.loads(open(out_path).read())
    print(f'Worker $i: resuming from {len(preds)} done')

for idx, r in enumerate(records):
    if r['id'] in preds:
        continue
    preds[r['id']] = llm_router_v3_1.classify(r['prompt_pos'], r['prompt_neg'])
    if (idx+1) % 10 == 0:
        open(out_path,'w').write(json.dumps(preds))
        print(f'Worker $i: {idx+1}/{len(records)} done', flush=True)
    time.sleep(0.05)

open(out_path,'w').write(json.dumps(preds))
print(f'Worker $i DONE: {len(preds)} predictions')
" &
    echo "Started worker $i (records $START-$END) PID=$!"
done

wait
echo "All workers done. Merging..."

python3 -c "
import json, glob
merged = {}
for f in sorted(glob.glob('outputs/router_v3_1_chunks/preds_*.json')):
    merged.update(json.loads(open(f).read()))
print(f'Total merged: {len(merged)} predictions')
json.dump(merged, open('outputs/llm_router_v3_1_predictions.json','w'), indent=2)
print('Saved to outputs/llm_router_v3_1_predictions.json')
"
