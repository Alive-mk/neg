"""
Create a label-cleaned E4 JSONL by removing records whose expected behavior is
incompatible with the negated prompt form.

Root cause this catches:
    expected_neg_behavior=preserve_positive
    prompt_neg="Which cooking method does not involve dry heat?"
    gold_pos="Baking involves dry heat."

That is not a preserve/scope-control case; preserving the positive answer is
semantically wrong. Keeping such rows makes automatic routers look bad for
doing the right thing and contaminates preserve free-generation metrics.

The same issue can occur for suppress_target rows, for example:
    expected_neg_behavior=suppress_target
    prompt_neg="Which planet does not have rings?"
    gold_pos="Saturn has rings."

That prompt asks for an alternative, so treating it as suppression also
punishes a correct SELECT decision.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


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

DIRECT_NEGATED_SELECTION_RE = re.compile(
    r"\b(which|what|select|identify|choose)\b"
    r".*\b(?:does|do|is|are|was|were|should|would|could|can)?\s*not\b"
)

NEGATED_YESNO_RE = re.compile(
    r"^(does|do|is|are|was|were|should|would|could|can)\b.*\bnot\b"
)

NEGATIVE_IMPERATIVE_RE = re.compile(r"^(do not|never)\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def removal_reason(record: dict[str, Any]) -> str | None:
    behavior = record.get("expected_neg_behavior")
    prompt = str(record.get("prompt_neg", "")).strip().lower()
    if DOUBLE_NEGATION_RE.search(prompt):
        return None

    if behavior != "select_gold_neg" and DIRECT_NEGATED_SELECTION_RE.search(prompt):
        return "nonselect_direct_negated_selection"

    if behavior == "preserve_positive":
        if NEGATED_YESNO_RE.search(prompt):
            return "preserve_negated_yes_no"
        if NEGATIVE_IMPERATIVE_RE.search(prompt):
            return "preserve_negative_imperative"

    return None


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        reason = removal_reason(record)
        if reason:
            item = dict(record)
            item["_cleaning_reason"] = reason
            removed.append(item)
        else:
            kept.append(record)
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    kept, removed = clean_records(records)
    write_jsonl(Path(args.output), kept)

    reason_counts = Counter(r["_cleaning_reason"] for r in removed)
    behavior_before = Counter(r.get("expected_neg_behavior") for r in records)
    behavior_after = Counter(r.get("expected_neg_behavior") for r in kept)
    report = {
        "input": args.input,
        "output": args.output,
        "n_input": len(records),
        "n_output": len(kept),
        "n_removed": len(removed),
        "removed_by_reason": dict(reason_counts),
        "behavior_before": dict(behavior_before),
        "behavior_after": dict(behavior_after),
        "removed_examples": [
            {
                "id": r.get("id"),
                "reason": r.get("_cleaning_reason"),
                "prompt_neg": r.get("prompt_neg"),
                "gold_pos": r.get("gold_pos", [])[:1],
                "candidate_pool_neg": r.get("candidate_pool_neg", [])[:2],
            }
            for r in removed[:20]
        ],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input: {len(records)}")
    print(f"Output: {len(kept)}")
    print(f"Removed: {len(removed)} {dict(reason_counts)}")
    print(f"Behavior before: {dict(behavior_before)}")
    print(f"Behavior after:  {dict(behavior_after)}")
    print(f"Saved: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
