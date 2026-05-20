"""
Candidate-aware LLM router v4.

This version tightens the E4-specific PRESERVE definition. In E4, some
out-of-scope negations appear in a condition or background clause; the intended
behavior is to preserve the positive answer when the question target is still
the same, even if a literal real-world answer might change.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.ai-gaochao.cn/v1/chat/completions"
API_KEY = os.environ.get("NEG_OPENAI_API_KEY", "")
if not API_KEY and Path("/tmp/.neg_api_key").exists():
    API_KEY = Path("/tmp/.neg_api_key").read_text().strip()


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}


SYSTEM = """You classify an E4 negation-ranking item into one behavior token.

You receive:
POSITIVE_PROMPT: the original prompt.
NEGATED_PROMPT: the negated prompt.
CANDIDATES: unlabeled possible continuations.

Return exactly one token: SUPPRESS, PRESERVE, or SELECT.

E4 behavior definitions:

SUPPRESS:
  The negation targets the answer itself or states a false/forbidden premise.
  The model should avoid the positive answer under the negated prompt.
  Examples:
  - POSITIVE: "Is a tiger a big cat?"
    NEGATED: "Is a tiger not a big cat?"
    -> SUPPRESS, because "tiger is not a big cat" is a false denial.
  - POSITIVE: "Who invented the telephone?"
    NEGATED: "Who did not invent the telephone?"
    -> SUPPRESS, because the positive inventor must be avoided.

SELECT:
  The negated prompt asks for a different, wrong, excluded, or reversed option.
  Examples:
  - "Which city is not the capital of Japan?" -> SELECT.
  - "What must you not do after completing the checklist?" -> SELECT.
  - "Which option is NOT valid/correct/allowed?" -> SELECT.

PRESERVE:
  The negation is out-of-scope for the answer target, so the intended E4 answer
  remains the same as the positive prompt.
  This includes:
  - Double negation: "not not", "not without", "not incorrect", "not uncommon",
    "not true that it never ..." -> PRESERVE.
  - A condition/background clause changes the scenario, but the question still
    asks for the same tool/object/attribute normally associated with the target.
    In E4 this is PRESERVE, even if the literal real-world answer could change.
    Examples:
      POSITIVE: "When you cut vegetables, what kitchen tool do you use?"
      NEGATED: "When you do not cut vegetables, what kitchen tool do you use?"
      -> PRESERVE.
      POSITIVE: "If it rains, what do people carry to stay dry?"
      NEGATED: "If it does not rain, what do people carry to stay dry?"
      -> PRESERVE.
      POSITIVE: "When the weather is sunny, what protects people's eyes?"
      NEGATED: "When the weather is not sunny, what protects people's eyes?"
      -> PRESERVE.

Decision rules:
1. If NEGATED_PROMPT contains a real double negation, choose PRESERVE.
2. If the negation is in an initial "when/if" background clause and the main
   question still asks for the same kind of tool/object/attribute, choose
   PRESERVE unless it explicitly asks which option is not/invalid/wrong.
3. If the main wh-question itself asks "which/what/who ... not ..." about the
   target answer, choose SELECT or SUPPRESS:
   - choose SELECT when it asks for a different valid option;
   - choose SUPPRESS when it is a false-premise factual denial.
4. Do not output explanations. Reply with exactly one word."""


def unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def candidate_pool(record: dict[str, Any]) -> list[str]:
    return unique(
        record.get("gold_pos", [])
        + record.get("gold_neg", [])
        + record.get("forbidden_neg", [])
        + record.get("candidate_pool_neg", [])
        + record.get("distractors", [])
    )


def write_predictions(path: Path, predictions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def classify(record: dict[str, Any], retries: int, timeout: float, default_token: str) -> str:
    user_msg = json.dumps(
        {
            "POSITIVE_PROMPT": record.get("prompt_pos", ""),
            "NEGATED_PROMPT": record.get("prompt_neg", ""),
            "CANDIDATES": candidate_pool(record),
        },
        ensure_ascii=False,
        indent=2,
    )
    payload = json.dumps(
        {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 10,
        }
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            text = data["choices"][0]["message"]["content"].strip().upper()
            if "SUPPRESS" in text:
                return "[SUPPRESS]"
            if "PRESERVE" in text:
                return "[PRESERVE]"
            if "SELECT" in text:
                return "[SELECT]"
            return default_token
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"API error for {record.get('id')}: {exc}", flush=True)
                return default_token
            time.sleep(2**attempt)
    return default_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/splits/router_calibration/calibration_clean_strict.jsonl")
    parser.add_argument("--output", default="outputs/llm_router_candidate_v4_calibration_predictions.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--default-token", default="[SUPPRESS]", choices=sorted(set(TOKEN_MAP.values())))
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.sample:
        records = records[: args.sample]

    out_path = Path(args.output)
    predictions: dict[str, str] = {}
    if args.resume and out_path.exists():
        try:
            predictions = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            predictions = {}
        print(f"Resuming from {len(predictions)} predictions", flush=True)

    correct = total = 0
    unsaved = 0
    for idx, record in enumerate(records, start=1):
        rid = record["id"]
        if rid not in predictions:
            predictions[rid] = classify(
                record,
                retries=args.retries,
                timeout=args.timeout,
                default_token=args.default_token,
            )
            unsaved += 1
            if unsaved >= max(1, args.save_every):
                write_predictions(out_path, predictions)
                unsaved = 0
                print(f"Saved {len(predictions)}/{len(records)} at record {idx}", flush=True)
            time.sleep(args.sleep)

        true = TOKEN_MAP[record["expected_neg_behavior"]]
        correct += int(predictions[rid] == true)
        total += 1

    write_predictions(out_path, predictions)
    print(f"Final accuracy: {correct}/{total} = {correct / total * 100:.1f}%", flush=True)
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(predictions.get(r["id"]) == token for r in subset)
        pct = n_correct / len(subset) * 100 if subset else 0.0
        print(f"  {behavior}: {n_correct}/{len(subset)} = {pct:.1f}%", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
