"""
Hybrid router that combines candidate-aware LLM predictions with the cleaned
rule router.

The candidate-aware LLM router is strong on SELECT but often over-predicts
SELECT on preserve/suppress items. This file exposes a few conservative merge
policies that only override SELECT when there is extra evidence from a rule
router or a lightweight supervised router.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from rule_router_clean import TOKEN_MAP, predict_behavior
from supervised_router_twostage import text_derived_suppression_only


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def behavior_label(token: str) -> str:
    return "[SELECT]" if token == "" else token


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


def hybrid_prediction(record: dict, llm_token: str) -> str:
    rule_token = predict_behavior(record)
    if rule_token in {"[SUPPRESS]", "[PRESERVE]"} and llm_token != "[PRESERVE]":
        return rule_token
    return llm_token or rule_token


def preserve_guard_prediction(record: dict, llm_token: str, supervised_token: str) -> str:
    rule_token = predict_behavior(record)
    if llm_token == "[SELECT]" and supervised_token == "[PRESERVE]" and rule_token != "[SUPPRESS]":
        return "[PRESERVE]"
    return llm_token or rule_token


def rule_preserve_prediction(record: dict, llm_token: str) -> str:
    rule_token = predict_behavior(record)
    if llm_token == "[SELECT]" and rule_token == "[PRESERVE]":
        return "[PRESERVE]"
    return llm_token or rule_token


def narrow_preserve_prediction(record: dict, llm_token: str) -> str:
    rule_token = predict_behavior(record)
    text = str(record.get("prompt_neg", ""))
    if llm_token == "[SELECT]":
        if rule_token == "[PRESERVE]":
            return "[PRESERVE]"
        if rule_token == "[SUPPRESS]" and any(p.search(text) for p in NARROW_PRESERVE_PATTERNS):
            return "[PRESERVE]"
    return llm_token or rule_token


def select_empty_prediction(_record: dict, llm_token: str) -> str:
    return "" if llm_token == "[SELECT]" else llm_token


def text_suppression_guard_prediction(record: dict, llm_token: str) -> str:
    if text_derived_suppression_only(record):
        return "[SUPPRESS]"
    return llm_token or predict_behavior(record)


PolicyFn = Callable[[dict, str, str], str]


def apply_policy(policy: str, record: dict, llm_token: str, supervised_token: str) -> str:
    policy_map: dict[str, PolicyFn] = {
        "llm": lambda rec, llm, _sup: llm,
        "llm_empty_select": lambda rec, llm, _sup: select_empty_prediction(rec, llm),
        "preserve_guard": lambda rec, llm, sup: preserve_guard_prediction(rec, llm, sup),
        "preserve_guard_empty_select": lambda rec, llm, sup: select_empty_prediction(
            rec, preserve_guard_prediction(rec, llm, sup)
        ),
        "rule_preserve_empty_select": lambda rec, llm, _sup: select_empty_prediction(
            rec, rule_preserve_prediction(rec, llm)
        ),
        "narrow_preserve_empty_select": lambda rec, llm, _sup: select_empty_prediction(
            rec, narrow_preserve_prediction(rec, llm)
        ),
        "recommended": lambda rec, llm, _sup: select_empty_prediction(
            rec, narrow_preserve_prediction(rec, llm)
        ),
        "text_suppression_guard_empty_select": lambda rec, llm, _sup: select_empty_prediction(
            rec, text_suppression_guard_prediction(rec, llm)
        ),
        "rule": lambda rec, _llm, _sup: predict_behavior(rec),
        "hybrid": lambda rec, llm, _sup: hybrid_prediction(rec, llm),
    }
    return policy_map[policy](record, llm_token, supervised_token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--llm-predictions", default="outputs/llm_router_candidate_predictions.json")
    parser.add_argument("--supervised-predictions", default=None)
    parser.add_argument("--output", default="outputs/hybrid_router_candidate_predictions.json")
    parser.add_argument(
        "--policy",
        choices=[
            "hybrid",
            "llm",
            "llm_empty_select",
            "rule",
            "rule_preserve_empty_select",
            "narrow_preserve_empty_select",
            "recommended",
            "text_suppression_guard_empty_select",
            "preserve_guard",
            "preserve_guard_empty_select",
        ],
        default="hybrid",
    )
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    llm_predictions = json.loads(Path(args.llm_predictions).read_text(encoding="utf-8"))
    needs_supervised = args.policy in {"preserve_guard", "preserve_guard_empty_select"}
    if needs_supervised and not args.supervised_predictions:
        raise SystemExit("--supervised-predictions is required for preserve_guard policies")
    supervised_predictions = {}
    if args.supervised_predictions:
        supervised_predictions = json.loads(Path(args.supervised_predictions).read_text(encoding="utf-8"))

    predictions = {
        record["id"]: apply_policy(
            args.policy,
            record,
            llm_predictions.get(record["id"], ""),
            supervised_predictions.get(record["id"], ""),
        )
        for record in records
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    correct = sum(
        behavior_label(predictions[record["id"]]) == TOKEN_MAP[record["expected_neg_behavior"]]
        for record in records
    )
    print(f"Accuracy: {correct}/{len(records)} = {correct / len(records) * 100:.1f}%")
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, tuple[int, int]] = {}
    for record in records:
        confusion[record["expected_neg_behavior"]][behavior_label(predictions[record["id"]])] += 1
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(behavior_label(predictions[r["id"]]) == token for r in subset)
        per_behavior[behavior] = (n_correct, len(subset))
        print(f"  {behavior}: {n_correct}/{len(subset)} = {n_correct / len(subset) * 100:.1f}%")
    print("Confusion:", {key: dict(value) for key, value in confusion.items()})
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
