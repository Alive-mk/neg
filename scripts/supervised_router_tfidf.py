"""
Train a lightweight supervised E4 behavior router.

The router uses only the positive prompt, negated prompt, and unlabeled
candidate text. It is intended as a low-cost deployment baseline for the
behavior-token routing bottleneck.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from rule_router_clean import predict_behavior


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def featurize(record: dict[str, Any], include_rule: bool = False) -> str:
    candidates = " || ".join(candidate_pool(record))
    parts = [
        "POS: " + str(record.get("prompt_pos", "")),
        "NEG: " + str(record.get("prompt_neg", "")),
        "CAND: " + candidates,
    ]
    if include_rule:
        parts.append("RULE: " + predict_behavior(record))
    return "\n".join(parts)


def make_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=1,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    min_df=1,
                    lowercase=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
        random_state=0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def summarize(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    correct = sum(predictions[r["id"]] == TOKEN_MAP[r["expected_neg_behavior"]] for r in records)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, dict[str, Any]] = {}
    for record in records:
        confusion[record["expected_neg_behavior"]][predictions[record["id"]]] += 1
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(predictions[r["id"]] == token for r in subset)
        per_behavior[behavior] = {
            "correct": n_correct,
            "total": len(subset),
            "accuracy": n_correct / len(subset) if subset else 0.0,
        }
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "per_behavior": per_behavior,
        "confusion": {key: dict(value) for key, value in confusion.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/merged_excl_boost/train_clean_strict.jsonl")
    parser.add_argument("--test", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--output", default="outputs/supervised_router_tfidf_predictions.json")
    parser.add_argument("--report", default="outputs/supervised_router_tfidf_report.json")
    parser.add_argument("--include-rule", action="store_true")
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train))
    test_records = load_jsonl(Path(args.test))
    model = make_model()
    model.fit(
        [featurize(record, include_rule=args.include_rule) for record in train_records],
        [record["expected_neg_behavior"] for record in train_records],
    )

    labels = model.predict([featurize(record, include_rule=args.include_rule) for record in test_records])
    predictions = {
        record["id"]: TOKEN_MAP[label]
        for record, label in zip(test_records, labels, strict=True)
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "train": args.train,
        "test": args.test,
        "train_counts": dict(Counter(r["expected_neg_behavior"] for r in train_records)),
        "test_counts": dict(Counter(r["expected_neg_behavior"] for r in test_records)),
        "include_rule": args.include_rule,
        "summary": summarize(test_records, predictions),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy'] * 100:.1f}%")
    for behavior, row in summary["per_behavior"].items():
        print(f"  {behavior}: {row['correct']}/{row['total']} = {row['accuracy'] * 100:.1f}%")
    print("Confusion:", summary["confusion"])
    print(f"Saved: {out_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
