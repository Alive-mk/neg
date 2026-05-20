"""
Rule router for the cleaned E4 split.

This version is paired with scripts/clean_e4_preserve_labels.py. It treats
direct "Which/What ... not ..." questions as SELECT instead of PRESERVE, while
reserving PRESERVE for double negation and clear out-of-scope "without" cases.
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


DOUBLE_NEGATION_RE = re.compile(
    r"\b("
    r"not\s+not|"
    r"not\s+(?:be\s+)?without|"
    r"cannot\s+not|can\s+not\s+not|"
    r"not\s+fail|never\s+fail|"
    r"not\s+(?:neglect|ignore|forget|skip|overlook|omit)|"
    r"not\s+(?:incorrect|untrue|uncommon)|"
    r"not\s+the\s+case\s+that.*never|"
    r"isn'?t\s+it\s+true.*cannot\s+never"
    r")\b",
)


def predict_behavior(record: dict[str, Any]) -> str:
    text = str(record.get("prompt_neg", "")).lower()
    candidate_text = " ".join(
        record.get("candidate_pool_neg", []) + record.get("distractors", [])
    ).lower()

    if DOUBLE_NEGATION_RE.search(text):
        return "[PRESERVE]"

    if " without " in f" {text} ":
        return "[PRESERVE]"

    if re.search(r"\b(which|what|select|identify|choose)\b.*\bnot\b", text):
        return "[SELECT]"

    if re.search(r"\b(incorrect|invalid|inappropriate|unsuitable|wrong)\b", text):
        return "[SELECT]"
    if re.search(
        r"\bnot\s+(?:the\s+)?"
        r"(?:correct|valid|good|effective|appropriate|suitable|best|largest|"
        r"smallest|fastest|slowest|oldest|newest|official|famous|known|used|"
        r"essential|necessary|required|needed)\b",
        text,
    ):
        return "[SELECT]"
    if re.search(r"\bshould\s+you\s+not\b|\bdo\s+you\s+not\b", text):
        return "[SELECT]"

    if re.search(r"\b(is|does|do|should|can|could|would|are)\b.*\bnot\b", text):
        comparative = re.search(
            r"\b(more|less|larger|smaller|higher|lower|better|worse|than|opposite|reverse)\b",
            text,
        )
        comparative_pool = re.search(
            r"\b(more|less|larger|smaller|higher|lower|than)\b",
            candidate_text,
        )
        if comparative and comparative_pool:
            return "[SELECT]"
        return "[SUPPRESS]"

    return "[SUPPRESS]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/validated_largetest_v2_clean.jsonl")
    parser.add_argument("--output", default="outputs/rule_router_clean_predictions.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    predictions = {record["id"]: predict_behavior(record) for record in records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    correct = sum(predictions[r["id"]] == TOKEN_MAP[r["expected_neg_behavior"]] for r in records)
    print(f"Accuracy: {correct}/{len(records)} = {correct / len(records) * 100:.1f}%")
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(predictions[r["id"]] == token for r in subset)
        print(f"  {behavior}: {n_correct}/{len(subset)} = {n_correct / len(subset) * 100:.1f}%")
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        confusion[record["expected_neg_behavior"]][predictions[record["id"]]] += 1
    print("Confusion:", {k: dict(v) for k, v in confusion.items()})
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
