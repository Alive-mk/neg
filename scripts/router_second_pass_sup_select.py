from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from rule_router_clean import predict_behavior


API_URL = "https://api.ai-gaochao.cn/v1/chat/completions"
API_KEY = os.environ.get("NEG_OPENAI_API_KEY", "")
if not API_KEY and Path("/tmp/.neg_api_key").exists():
    API_KEY = Path("/tmp/.neg_api_key").read_text().strip()


NARROW_PRESERVE_PATTERNS = [
    re.compile(r"\bif\b[^?.!]*\bdo not have\b", re.I),
    re.compile(r"\bif\b[^?.!]*\bnot available\b", re.I),
    re.compile(r"\bif\b[^?.!]*\bis not a\b", re.I),
    re.compile(r"\bif\b[^?.!]*\bis not an\b", re.I),
    re.compile(r"\bnot a poor\b", re.I),
    re.compile(r"\bdoes not precede\b", re.I),
    re.compile(r"^why\b.*\bnot often called\b", re.I),
    re.compile(r"^describe\b.*official language not spoken\b", re.I),
]


SYSTEM = """You are a second-pass router for an ambiguous negation branch.

You only answer one of two labels:
- SUPPRESS
- SELECT

Choose SUPPRESS when the negation simply denies, forbids, or rejects the original
proposition/action/attribute, so the original positive answer should be
suppressed rather than replaced by a new alternative.

Choose SELECT when the negated prompt is really asking for an alternative,
counterexample, excluded option, reversed choice, or another candidate that
fits the negated condition.

Important distinctions:
- "Is/Does X not Y?" is often SUPPRESS if it just denies the original claim.
- Imperatives like "Do not ..." are usually SUPPRESS.
- Questions asking which option is different / excluded / not correct are SELECT.
- If the prompt asks for another valid option because the original condition
  is negated, that is SELECT.

Reply with exactly one word: SUPPRESS or SELECT."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def classify(record: dict[str, Any], retries: int = 4, timeout: float = 60.0) -> str:
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
            "max_tokens": 8,
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
            if "SELECT" in text:
                return "[SELECT]"
            return "[SELECT]"
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"API error for {record.get('id')}: {exc}", flush=True)
                return "[SELECT]"
            time.sleep(2**attempt)
    return "[SELECT]"


def should_force_preserve(record: dict[str, Any], llm_token: str) -> bool:
    if llm_token != "[SELECT]":
        return False
    rule_token = predict_behavior(record)
    text = str(record.get("prompt_neg", ""))
    if rule_token == "[PRESERVE]":
        return True
    return rule_token == "[SUPPRESS]" and any(p.search(text) for p in NARROW_PRESERVE_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--llm-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    llm_predictions = load_json(Path(args.llm_predictions))
    out_path = Path(args.output)
    predictions: dict[str, str] = {}
    if args.resume and out_path.exists():
        try:
            predictions = load_json(out_path)
        except json.JSONDecodeError:
            predictions = {}

    target_records = []
    for record in records:
        rid = record["id"]
        llm_token = llm_predictions[rid]
        rule_token = predict_behavior(record)
        if should_force_preserve(record, llm_token):
            predictions[rid] = "[PRESERVE]"
            continue
        if llm_token == "[SELECT]" and rule_token == "[SUPPRESS]":
            target_records.append(record)
        else:
            predictions[rid] = "" if llm_token == "[SELECT]" else llm_token

    unsaved = 0
    for idx, record in enumerate(target_records, start=1):
        rid = record["id"]
        if rid not in predictions:
            predictions[rid] = "" if classify(record) == "[SELECT]" else "[SUPPRESS]"
            unsaved += 1
            if unsaved >= max(1, args.save_every):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
                unsaved = 0
                print(f"Saved {idx}/{len(target_records)} ambiguous records", flush=True)
            time.sleep(args.sleep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ambiguous branch size: {len(target_records)}", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
