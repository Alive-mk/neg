"""
Search lightweight routing policies over existing router predictions.

This is a diagnostic tool for the deployment bottleneck: it combines
candidate-aware LLM, supervised TF-IDF, and rule-router predictions without
using gold labels at inference time. Reported test accuracy is diagnostic only;
paper-facing numbers should avoid selecting a policy on the test set.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from rule_router_clean import TOKEN_MAP, predict_behavior


Policy = Callable[[dict, str, str, str], str]

OPEN_ENDED_RE = re.compile(r"^\s*(explain|describe|provide|list|outline|summarize|why|how)\b", re.I)
DIRECT_PROHIBITION_RE = re.compile(r"^\s*(do\s+not|don't)\b", re.I)
YES_NO_RE = re.compile(r"^\s*(is|does|do|should|can|could|would|are|will|was|were)\b", re.I)
CONDITIONAL_NEG_RE = re.compile(r"\bif\b[^?.!]*\bnot\b", re.I)
COMMUNICATION_WITHOUT_RE = re.compile(
    r"\bwithout\s+(?:mentioning|referring\s+to|stating|saying|naming|citing)\b",
    re.I,
)


def neg_text(record: dict) -> str:
    return str(record.get("prompt_neg", ""))


def is_open_ended_nonconditional(record: dict) -> bool:
    text = neg_text(record)
    if not OPEN_ENDED_RE.search(text):
        return False
    if DIRECT_PROHIBITION_RE.search(text) or YES_NO_RE.search(text):
        return False
    if CONDITIONAL_NEG_RE.search(text):
        return False
    return " not " in f" {text.lower()} "


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(records: list[dict], predictions: dict[str, str]) -> dict:
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_type: dict[str, dict] = {}
    for record in records:
        gold = TOKEN_MAP[record["expected_neg_behavior"]]
        pred = predictions[record["id"]]
        pred_for_behavior = "[SELECT]" if pred == "" else pred
        correct += int(pred_for_behavior == gold)
        confusion[record["expected_neg_behavior"]][pred] += 1
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(("[SELECT]" if predictions[r["id"]] == "" else predictions[r["id"]]) == token for r in subset)
        per_type[behavior] = {
            "correct": n_correct,
            "total": len(subset),
            "accuracy": n_correct / len(subset) if subset else 0.0,
        }
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "per_type": per_type,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "prediction_distribution": dict(Counter(predictions.values())),
    }


def policy_llm(_record: dict, llm: str, _sup: str, _rule: str) -> str:
    return llm


def policy_llm_empty_select(_record: dict, llm: str, _sup: str, _rule: str) -> str:
    return "" if llm == "[SELECT]" else llm


def policy_preserve_guard(_record: dict, llm: str, sup: str, rule: str) -> str:
    if llm == "[SELECT]" and sup == "[PRESERVE]" and rule != "[SUPPRESS]":
        return "[PRESERVE]"
    return llm


def policy_preserve_guard_empty_select(record: dict, llm: str, sup: str, rule: str) -> str:
    pred = policy_preserve_guard(record, llm, sup, rule)
    return "" if pred == "[SELECT]" else pred


def policy_consensus_preserve(_record: dict, llm: str, sup: str, rule: str) -> str:
    if llm == "[SELECT]" and sup == "[PRESERVE]" and rule == "[PRESERVE]":
        return "[PRESERVE]"
    return llm


def policy_consensus_preserve_empty_select(record: dict, llm: str, sup: str, rule: str) -> str:
    pred = policy_consensus_preserve(record, llm, sup, rule)
    return "" if pred == "[SELECT]" else pred


def policy_nonselect_consensus(_record: dict, llm: str, sup: str, rule: str) -> str:
    if llm == "[SELECT]" and sup == rule and sup in {"[SUPPRESS]", "[PRESERVE]"}:
        return sup
    return llm


def policy_nonselect_consensus_empty_select(record: dict, llm: str, sup: str, rule: str) -> str:
    pred = policy_nonselect_consensus(record, llm, sup, rule)
    return "" if pred == "[SELECT]" else pred


def policy_rule_preserve_empty_select(record: dict, llm: str, _sup: str, rule: str) -> str:
    pred = "[PRESERVE]" if llm == "[SELECT]" and rule == "[PRESERVE]" else llm
    return "" if pred == "[SELECT]" else pred


def policy_communication_without_empty_select(record: dict, llm: str, _sup: str, rule: str) -> str:
    text = neg_text(record)
    pred = llm
    if llm == "[SELECT]" and (rule == "[PRESERVE]" or COMMUNICATION_WITHOUT_RE.search(text)):
        pred = "[PRESERVE]"
    return "" if pred == "[SELECT]" else pred


def policy_open_ended_preserve_empty_select(record: dict, llm: str, sup: str, rule: str) -> str:
    pred = llm
    if llm == "[SELECT]" and rule != "[SELECT]":
        if sup == "[PRESERVE]" or is_open_ended_nonconditional(record):
            pred = "[PRESERVE]"
    return "" if pred == "[SELECT]" else pred


def policy_open_ended_rule_preserve_empty_select(record: dict, llm: str, _sup: str, rule: str) -> str:
    pred = llm
    if llm == "[SELECT]" and (rule == "[PRESERVE]" or is_open_ended_nonconditional(record)):
        pred = "[PRESERVE]"
    return "" if pred == "[SELECT]" else pred


POLICIES: dict[str, Policy] = {
    "llm": policy_llm,
    "llm_empty_select": policy_llm_empty_select,
    "preserve_guard": policy_preserve_guard,
    "preserve_guard_empty_select": policy_preserve_guard_empty_select,
    "consensus_preserve": policy_consensus_preserve,
    "consensus_preserve_empty_select": policy_consensus_preserve_empty_select,
    "nonselect_consensus": policy_nonselect_consensus,
    "nonselect_consensus_empty_select": policy_nonselect_consensus_empty_select,
    "rule_preserve_empty_select": policy_rule_preserve_empty_select,
    "communication_without_empty_select": policy_communication_without_empty_select,
    "open_ended_preserve_empty_select": policy_open_ended_preserve_empty_select,
    "open_ended_rule_preserve_empty_select": policy_open_ended_rule_preserve_empty_select,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--llm", default="outputs/llm_router_candidate_predictions.json")
    parser.add_argument("--supervised", default="outputs/supervised_router_tfidf_rule_predictions.json")
    parser.add_argument("--output-dir", default="outputs/router_policy_search")
    args = parser.parse_args()

    records = load_jsonl(Path(args.records))
    llm_predictions = json.loads(Path(args.llm).read_text(encoding="utf-8"))
    supervised_predictions = json.loads(Path(args.supervised).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for name, policy in POLICIES.items():
        predictions = {}
        for record in records:
            rid = record["id"]
            rule = predict_behavior(record)
            predictions[rid] = policy(
                record,
                llm_predictions.get(rid, ""),
                supervised_predictions.get(rid, ""),
                rule,
            )
        pred_path = out_dir / f"{name}_predictions.json"
        pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        report[name] = {
            "predictions": str(pred_path),
            "summary": summarize(records, predictions),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, row in report.items():
        summary = row["summary"]
        per = summary["per_type"]
        print(
            f"{name:34s} acc={summary['accuracy']*100:5.1f}% "
            f"sup={per['suppress_target']['accuracy']*100:5.1f}% "
            f"pre={per['preserve_positive']['accuracy']*100:5.1f}% "
            f"sel={per['select_gold_neg']['accuracy']*100:5.1f}%"
        )
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
