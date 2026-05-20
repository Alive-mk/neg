"""
LLM router v3.1: targeted fixes over v3.

Key fixes over v3:
1. Remove "exact logical opposites → PRESERVE" CRITICAL RULE (caused select|exclusive_choice -50pp)
   "Which X does NOT Y?" selection questions → always SELECT (answer changes to a different X)
2. Distinguish open-ended factual denial (SUPPRESS) from yes/no question (may be SELECT)
   "Explain why X does NOT Y" where X does Y → SUPPRESS (false-premise instructional prompt)
   "Is X not Y?" where positive answer is "no" → SELECT (contrastive resolution)
3. Keep improvements: double negation → PRESERVE; "without [method]" → PRESERVE;
   contrastive yes/no where negation is correct → SELECT

Expected: recover select|exclusive_choice 38%→~85%; recover suppress_target 20%→~37%;
          lose preserve|exclusive_choice 58%→~41%; keep select|contrastive ~67%.
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
  The negation is "out of scope" — it does not change the correct answer.

  Case A — "without [method/approach/technique]" in an instructional prompt:
    POSITIVE: "Explain how to stop bleeding."
    NEGATIVE: "Explain how to stop bleeding without using first aid."
    → PRESERVE (direct pressure is still the correct method to explain)

  Case B — double negation that cancels out:
    "must not neglect", "not without", "should not ignore", "not never" → PRESERVE
    The double negative resolves to a positive, preserving the original meaning.

  *** IMPORTANT: "Which X does NOT Y?" selection questions are NOT PRESERVE — see SELECT. ***
──────────────────────────────────────────────────────
SUPPRESS: The negation introduces a factually false claim. The model should answer
  accurately and resist the misleading negation.

  Case A — "without [concrete object/ingredient/component]" in output instructions:
    POSITIVE: "Bake a cake."
    NEGATIVE: "Bake a cake without frosting." / "Bake a cake without eggs."
    → SUPPRESS (frosting/eggs must be absent from the output)

  Case B — open-ended instructional/explanatory prompt with false-premise negation:
    POSITIVE: "Explain why deep sleep is important for memory consolidation."
    NEGATIVE: "Explain why deep sleep is NOT important for memory consolidation."
    → SUPPRESS (deep sleep IS important; the negated premise is factually wrong)

    POSITIVE: "Describe how regular cardio exercise improves heart health."
    NEGATIVE: "Describe how regular cardio exercise does NOT improve heart health."
    → SUPPRESS (cardio DOES improve; false premise in open-ended explanatory prompt)

    KEY SIGNAL: if NEGATIVE is an "Explain why..." / "Describe how..." prompt and the positive
    states a true fact, then the negation creates a false premise → SUPPRESS.

  Case C — yes/no question where the positive answer is "yes":
    POSITIVE: "Is the park located near the river?" (answer: yes, it is)
    NEGATIVE: "Is the park NOT located near the river?"
    → SUPPRESS (park IS near river; model must not be fooled by "not")

  KEY DISTINCTION from PRESERVE:
    "without [method/technique/approach]" → PRESERVE
    "without [concrete thing that would normally appear in the output]" → SUPPRESS
──────────────────────────────────────────────────────
SELECT: The negation asks the model to pick the WRONG/NEGATED/REVERSED option.
  The correct answer for NEGATIVE is DIFFERENT from POSITIVE.

  Case A — selection question asking for the wrong/invalid/excluded option:
    "Which step should NOT be used?" / "Which is NOT correct?" → SELECT
    "Which option is NOT valid?" / "Which answer is NOT right?" → SELECT

  Case B — "Which X does NOT Y?" (any selection question with NOT in the property):
    POSITIVE: "Which planet is the largest in our solar system?"
    NEGATIVE: "Which planet is NOT the largest in our solar system?"
    → SELECT (answer changes: Jupiter → any other planet)

    POSITIVE: "Which country primarily speaks Spanish?"
    NEGATIVE: "Which country does NOT primarily speak Spanish?"
    → SELECT (answer changes to a different country)

    RULE: "Which X does NOT Y?" ALWAYS leads to a different answer → always SELECT.

  Case C — yes/no question where the positive answer is "no" (negation IS correct):
    POSITIVE: "Is the library located north of the cafeteria?" (answer: no, it's east)
    NEGATIVE: "Is the library NOT located north of the cafeteria?"
    → SELECT (the negation is correct; model should confirm it is indeed NOT north)

  KEY DISTINCTIONS:
    "Which X does NOT Y?" → SELECT (answer changes to a different X)
    "Explain/Describe why X does NOT Y" (open-ended, false premise) → SUPPRESS
    "Is X not Y?" where positive answer was "yes" → SUPPRESS
    "Is X not Y?" where positive answer was "no" → SELECT
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
    parser.add_argument("--output", default="outputs/llm_router_v3_1_predictions.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
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
