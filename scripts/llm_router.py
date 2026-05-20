"""
LLM-based behavior token router using gpt-4.1-mini.
Classifies each record as suppress/preserve/select and saves predictions.
"""
import json, os, time, re
from pathlib import Path
import urllib.request, urllib.error

API_URL = "https://api.ai-gaochao.cn/v1/chat/completions"
API_KEY = os.environ.get("NEG_OPENAI_API_KEY", "")
# Allow override via env written to tmp file
_key_file = '/tmp/.neg_api_key'
import os as _os
if not API_KEY and _os.path.exists(_key_file):
    API_KEY = open(_key_file).read().strip()

SYSTEM = """You are classifying negated prompts into one of three behavior categories.

SUPPRESS: The prompt negates a true/affirmative fact and expects the model to avoid that fact.
  Example: "Is metal NOT known for conductivity?" → metal IS conductive, so suppress that fact.

PRESERVE: The negation is out-of-scope; the core factual answer should still be given despite the negation.
  Example: "Explain how to stop bleeding without first aid" → the correct first-aid answer is still valid.
  Includes: double negation ("not without", "not never"), and cases where "without X" is a context modifier not changing the core answer.

SELECT: The prompt explicitly asks for an incorrect, wrong, or negated alternative.
  Example: "Which step should NOT be used?" / "Which planet is NOT the largest?" → select the wrong/negated option.

Reply with exactly one word: SUPPRESS, PRESERVE, or SELECT."""

def classify(prompt_neg: str, retries: int = 3) -> str:
    payload = json.dumps({
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_neg}
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }).encode()
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip().upper()
            if "SUPPRESS" in text: return "[SUPPRESS]"
            if "PRESERVE" in text: return "[PRESERVE]"
            if "SELECT" in text:   return "[SELECT]"
            return "[SUPPRESS]"  # fallback
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return "[SUPPRESS]"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/processed/validated_largetest_v2.jsonl")
    parser.add_argument("--output", default="outputs/router_predictions.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    out_path = Path(args.output)

    # Resume support
    predictions = {}
    if args.resume and out_path.exists():
        predictions = json.loads(out_path.read_text())
        print(f"Resuming from {len(predictions)} existing predictions")

    TOKEN_MAP = {"suppress_target": "[SUPPRESS]",
                 "preserve_positive": "[PRESERVE]",
                 "select_gold_neg": "[SELECT]"}

    correct = total = 0
    for i, r in enumerate(records):
        rid = r["id"]
        if rid in predictions:
            pred = predictions[rid]
        else:
            pred = classify(r["prompt_neg"])
            predictions[rid] = pred
            if (i + 1) % 50 == 0:
                out_path.write_text(json.dumps(predictions, indent=2))
                print(f"  Saved {len(predictions)}/{len(records)} predictions")
            time.sleep(0.1)  # gentle rate limit

        true = TOKEN_MAP[r["expected_neg_behavior"]]
        if pred == true: correct += 1
        total += 1

    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"\nFinal accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    # Per-type breakdown
    for typ, tok in TOKEN_MAP.items():
        recs = [r for r in records if r["expected_neg_behavior"] == typ]
        c = sum(1 for r in recs if predictions.get(r["id"]) == tok)
        print(f"  {typ}: {c}/{len(recs)} = {c/len(recs)*100:.1f}%")
