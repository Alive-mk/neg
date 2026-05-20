"""
Conservative PRESERVE rescue over the candidate-aware LLM router.

Use the stable v3 candidate-aware router as the base. Only override SELECT to
PRESERVE when v4 also says PRESERVE and the prompt matches a narrow
out-of-scope instructional pattern that was safe on strict error analysis.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}

OPEN_IF_RE = re.compile(
    r"^\s*(explain|describe|list|outline|provide)\b.*\bif\b.*\b(?:not|do not|does not|don't|is not|are not)\b",
    re.I,
)
WITHOUT_METHOD_RE = re.compile(
    r"\bwithout using\b.*\b(standard|wizard|method|approach|technique)\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def behavior_label(token: str) -> str:
    return "[SELECT]" if token == "" else token


def should_rescue(record: dict[str, Any], base_token: str, v4_token: str) -> bool:
    if base_token != "[SELECT]" or v4_token != "[PRESERVE]":
        return False
    text = str(record.get("prompt_neg", ""))
    return bool(OPEN_IF_RE.search(text) or WITHOUT_METHOD_RE.search(text))


def summarize(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        pred = behavior_label(predictions[record["id"]])
        gold = TOKEN_MAP[record["expected_neg_behavior"]]
        correct += int(pred == gold)
        confusion[record["expected_neg_behavior"]][pred] += 1
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "changed": sum(1 for value in predictions.values() if value == "[PRESERVE]"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--v4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", default=None)
    parser.add_argument("--empty-select", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(Path(args.records))
    base = load_json(Path(args.base))
    v4 = load_json(Path(args.v4))

    predictions: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = record["id"]
        base_token = base[rid]
        v4_token = v4[rid]
        if should_rescue(record, base_token, v4_token):
            predictions[rid] = "[PRESERVE]"
            details[rid] = {
                "route": "preserve_rescue",
                "base": base_token,
                "v4": v4_token,
                "prompt_neg": record.get("prompt_neg", ""),
            }
        else:
            predictions[rid] = "" if args.empty_select and base_token == "[SELECT]" else base_token
            details[rid] = {"route": "base", "base": base_token, "v4": v4_token}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.details:
        Path(args.details).write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(records, predictions)
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy'] * 100:.1f}%")
    print("Confusion:", summary["confusion"])
    print(f"Rescued: {sum(1 for item in details.values() if item['route'] == 'preserve_rescue')}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
