"""
Candidate-aware LLM router for E4 behavior tokens.

Unlike earlier routers, this sends the positive prompt, negated prompt, and the
unlabeled candidate pool. This is still non-oracle for candidate-ranking usage:
the router does not receive expected_neg_behavior or which candidate is gold.
The extra candidate context is necessary because prompt-only routing often
cannot distinguish false-premise suppression from negated-option selection.
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


SYSTEM = """You classify an E4 negation-ranking item into one behavior token.

You receive:
POSITIVE_PROMPT: an original prompt.
NEGATED_PROMPT: a prompt with negation.
CANDIDATES: unlabeled possible continuations.

Return exactly one token:

SUPPRESS:
  The negated prompt contains a false or forbidden premise. The positive answer
  should be suppressed/avoided under the negated prompt.
  Example: POSITIVE asks "Is the meeting in the morning?" and NEGATED asks
  "Is the meeting not in the morning?" while the candidate pool contains
  "The meeting is in the morning." plus alternatives. The behavior is SUPPRESS.

PRESERVE:
  The negation does not change the intended answer, usually because it is a true
  double negation ("not without", "cannot not", "not fail to", "not incorrect",
  "not uncommon") or because a "without X" modifier excludes a tool/procedure
  that is not the core answer.

SELECT:
  The negated prompt asks for a different / wrong / excluded / reversed option.
  Example: "Which city is not the capital of Japan?" or "Which method does not
  use dry heat?" asks for an option different from the positive answer.

Important rules:
- "Which/What/Select/Identify/Choose ... not ..." is usually SELECT unless it
  is a real double negation such as "not without" or "not incorrect".
- "Is/Does X not Y?" is SUPPRESS when the positive prompt asserts X Y and the
  negated prompt is a false-premise denial; it is SELECT only when the candidate
  pool clearly contains a reversed correct answer that should be selected.
- Do not preserve an answer that directly satisfies the property being negated.

Reply with exactly one word: SUPPRESS, PRESERVE, or SELECT."""


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}


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


def classify(
    record: dict[str, Any],
    retries: int = 3,
    timeout: float = 30.0,
    default_token: str = "[SUPPRESS]",
) -> str:
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
    parser.add_argument("--input", default="data/processed/validated_largetest_v2_clean.jsonl")
    parser.add_argument("--output", default="outputs/llm_router_candidate_predictions.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--default-token", default="[SUPPRESS]", choices=sorted(set(TOKEN_MAP.values())))
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
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
