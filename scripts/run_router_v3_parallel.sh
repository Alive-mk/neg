#!/bin/bash
# Run llm_router_v3 in 5 parallel workers, each handling ~105 records
# Usage: ./run_router_v3_parallel.sh

INPUT="data/processed/validated_largetest_v2.jsonl"
TOTAL=$(wc -l < "$INPUT")
N_WORKERS=5
CHUNK=$((TOTAL / N_WORKERS + 1))

mkdir -p outputs/router_v3_chunks

# Split into chunks and process in parallel
for i in $(seq 0 $((N_WORKERS-1))); do
    START=$((i * CHUNK))
    END=$(((i+1) * CHUNK))
    OUTPUT="outputs/router_v3_chunks/preds_${i}.json"

    python3 -c "
import json, sys, os, time, urllib.request
sys.path.insert(0, 'scripts')

API_URL = 'https://api.ai-gaochao.cn/v1/chat/completions'
API_KEY = open('/tmp/.neg_api_key').read().strip()

SYSTEM = open('scripts/llm_router_v3.py').read().split('SYSTEM = \"\"\"')[1].split('\"\"\"')[0]

def classify(prompt_pos, prompt_neg, retries=3):
    user_msg = f'POSITIVE: {prompt_pos}\nNEGATIVE: {prompt_neg}'
    payload = json.dumps({'model':'gpt-4.1-mini','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user_msg}],'temperature':0.0,'max_tokens':10}).encode()
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=12) as resp:
                d = json.loads(resp.read())
            text = d['choices'][0]['message']['content'].strip().upper()
            if 'SUPPRESS' in text: return '[SUPPRESS]'
            if 'PRESERVE' in text: return '[PRESERVE]'
            if 'SELECT' in text: return '[SELECT]'
            return '[SUPPRESS]'
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return '[SUPPRESS]'

records = [json.loads(l) for l in open('$INPUT')]
records = records[$START:$END]
out_path = '$OUTPUT'

preds = {}
if os.path.exists(out_path):
    preds = json.loads(open(out_path).read())
    print(f'Worker $i: resuming from {len(preds)} done')

for i, r in enumerate(records):
    if r['id'] in preds:
        continue
    preds[r['id']] = classify(r['prompt_pos'], r['prompt_neg'])
    if (i+1) % 10 == 0:
        open(out_path,'w').write(json.dumps(preds))
        print(f'Worker $i: {len(preds)}/{len(records)+len([x for x in preds if x not in {rr[\"id\"] for rr in records}])} done', flush=True)
    time.sleep(0.05)

open(out_path,'w').write(json.dumps(preds))
print(f'Worker $i DONE: {len(preds)} predictions')
" &
    echo "Started worker $i (records $START-$END) PID=$!"
done

wait
echo "All workers done. Merging..."

python3 -c "
import json
merged = {}
import glob
for f in sorted(glob.glob('outputs/router_v3_chunks/preds_*.json')):
    merged.update(json.loads(open(f).read()))
print(f'Total merged: {len(merged)} predictions')
json.dump(merged, open('outputs/llm_router_v3_predictions.json','w'), indent=2)
print('Saved to outputs/llm_router_v3_predictions.json')
"
