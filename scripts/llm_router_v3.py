"""
LLM router v3: uses BOTH prompt_pos + prompt_neg, plus improved decision rules.

Key fixes over v2:
1. Pass prompt_pos alongside prompt_neg → allows detecting logical-opposite pairs (→ PRESERVE)
2. Distinguish "without [method]" (PRESERVE) vs "without [ingredient/object]" (SUPPRESS)
3. Distinguish "Is X not Y?" for SELECT (X is truly not Y) vs SUPPRESS (X actually is Y)
"""
import json, os, time, re
from pathlib import Path
import urllib.request

API_URL = "https://api.ai-gaochao.cn/v1/chat/completions"
API_KEY = os.environ.get("NEG_OPENAI_API_KEY", "")
_key_file = '/tmp/.neg_api_key'
if not API_KEY and os.path.exists(_key_file):
    API_KEY = open(_key_file).read().strip()

SYSTEM = """You are classifying negated question pairs into one of three behavior categories.
You receive:
  POSITIVE: the original (non-negated) prompt
  NEGATIVE: the negated prompt

The three categories and their definitions:

──────────────────────────────────────────────────────
PRESERVE: The correct answer for NEGATIVE is the SAME as for POSITIVE.
  The negation is "out of scope" — it does not change which answer is correct.

  Case A — "without [method/approach]" in open instructions:
    POSITIVE: "Explain how to stop bleeding."
    NEGATIVE: "Explain how to stop bleeding without using first aid."
    → PRESERVE (applying direct pressure is still the correct answer)

  Case B — POSITIVE and NEGATIVE are logical negations of each other ("does not" added/removed):
    POSITIVE: "Which cooking method involves using dry heat?"
    NEGATIVE: "Which cooking method does NOT involve using dry heat?"
    → PRESERVE (the dry-heat method like Baking is STILL the expected answer)
    *** CRITICAL RULE: If POS and NEG are exact logical opposites (only "not/does not/is not"
        added or removed), classify as PRESERVE — even if it seems counterintuitive. ***

  Case C — double negation / negation cancellation:
    "must not neglect", "not without", "should not ignore" → PRESERVE
──────────────────────────────────────────────────────
SUPPRESS: The negation requires that some concept be ABSENT from the output.
  The model should resist the negation and still answer accurately, but avoid the negated concept.

  Subcase — factual denial (POS confirms something, NEG denies it):
    POSITIVE: "Is the park located near the river?"  (answer: yes)
    NEGATIVE: "Is the park NOT located near the river?"
    → SUPPRESS (park IS near river; model must not be fooled by "not")

  Subcase — "without [concrete object/ingredient/component]" in instructions:
    POSITIVE: "Bake a cake."
    NEGATIVE: "Bake a cake without frosting." / "Bake a cake without eggs."
    → SUPPRESS (frosting/eggs must be absent from the output)

  KEY DISTINCTION from PRESERVE:
    "without [method/technique/approach]" → PRESERVE
    "without [concrete thing that would normally appear in the output]" → SUPPRESS
──────────────────────────────────────────────────────
SELECT: The negation asks the model to pick the WRONG/NEGATED/REVERSED option.
  The correct answer for NEGATIVE is DIFFERENT from POSITIVE (it is the negated option).

  Case A — asks for incorrect/wrong/invalid option:
    "Which step should NOT be used?" / "Which is NOT correct?" → SELECT

  Case B — negated factual claim where the negation IS CORRECT:
    POSITIVE: "Is the library located north of the cafeteria?" (answer: no, it's east)
    NEGATIVE: "Is the library NOT located north of the cafeteria?"
    → SELECT (the model should agree that it is indeed NOT north)

  KEY DISTINCTION from SUPPRESS:
    "Is X not Y?" → SELECT if X is truly NOT Y in reality
    "Is X not Y?" → SUPPRESS if X IS Y and the "not" is wrong
──────────────────────────────────────────────────────

Reply with exactly one word: SUPPRESS, PRESERVE, or SELECT."""


def classify(prompt_pos: str, prompt_neg: str, retries: int = 3) -> str:
    user_msg = f"POSITIVE: {prompt_pos}\nNEGATIVE: {prompt_neg}"
    payload = json.dumps({
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }).encode()
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip().upper()
            if "SUPPRESS" in text: return "[SUPPRESS]"
            if "PRESERVE" in text: return "[PRESERVE]"
            if "SELECT" in text:   return "[SELECT]"
            return "[SUPPRESS]"
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
    parser.add_argument("--output", default="outputs/llm_router_v3_predictions.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=None, help="Only classify first N records (for testing)")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    if args.sample:
        records = records[:args.sample]
    out_path = Path(args.output)

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
            pred = classify(r["prompt_pos"], r["prompt_neg"])
            predictions[rid] = pred
            if (i + 1) % 50 == 0:
                out_path.write_text(json.dumps(predictions, indent=2))
                print(f"  Saved {len(predictions)}/{len(records)} predictions")
            time.sleep(0.1)

        true = TOKEN_MAP[r["expected_neg_behavior"]]
        if pred == true: correct += 1
        total += 1

    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"\nFinal accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    for typ, tok in TOKEN_MAP.items():
        recs = [r for r in records if r["expected_neg_behavior"] == typ]
        c = sum(1 for r in recs if predictions.get(r["id"]) == tok)
        print(f"  {typ}: {c}/{len(recs)} = {c/len(recs)*100:.1f}%")

    # Per semantic_mode breakdown
    from collections import defaultdict
    mode_stats = defaultdict(lambda: {'c': 0, 'n': 0})
    for r in records:
        key = f"{r['expected_neg_behavior']}|{r.get('semantic_mode','?')}"
        mode_stats[key]['n'] += 1
        if predictions.get(r['id']) == TOKEN_MAP[r['expected_neg_behavior']]:
            mode_stats[key]['c'] += 1
    print("\nPer semantic_mode:")
    for key in sorted(mode_stats):
        v = mode_stats[key]
        print(f"  {key}: {v['c']}/{v['n']} = {v['c']/v['n']*100:.0f}%")
