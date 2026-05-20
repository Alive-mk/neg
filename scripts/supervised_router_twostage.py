"""
Train a two-stage supervised behavior router for E4.

Stage 1 detects preserve_positive vs. non-preserve. Stage 2 separates
suppress_target from select_gold_neg for non-preserve records. This targets the
known failure mode where preserve examples are over-routed to SELECT.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from rule_router_clean import predict_behavior
from supervised_router_tfidf import candidate_pool


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}

TEXT_DERIVED_SUPPRESSION_ONLY_PATTERNS = [
    re.compile(
        r"^what is not the "
        r"(?:capital city|chemical symbol|official language|currency|administrative capital) "
        r"of [^?]+[?]?$",
        re.I,
    ),
    re.compile(r"^what is not the position of [^?]+ from the sun[?]?$", re.I),
    re.compile(r"^what is not the largest (?:planet in the solar system|ocean on earth)[?]?$", re.I),
    re.compile(r"^what is not a national language of [^?]+[?]?$", re.I),
    re.compile(r"^what is not a primary function of [^?]+[?]?$", re.I),
    re.compile(r"^what is not a result of [^?]+[?]?$", re.I),
    re.compile(r"^what is not the boiling point of water at sea level[?]?$", re.I),
    re.compile(r"^what is an? [a-z ]+ not classified as[?]?$", re.I),
    re.compile(r"^which is not the largest (?:ocean on earth|planet in the solar system)[?]?$", re.I),
    re.compile(r"^which is not earth'?s natural satellite[?]?$", re.I),
    re.compile(r"^which is not a primary function of [^?]+[?]?$", re.I),
    re.compile(r"^which is not a result of [^?]+[?]?$", re.I),
    re.compile(r"^which temperature is not the boiling point of water at sea level[?]?$", re.I),
    re.compile(r"^which base does not pair with adenine in dna[?]?$", re.I),
    re.compile(r"^which of the following does not travel at the speed of light in a vacuum[?]?$", re.I),
    re.compile(r"^which continent is [^?]+ not located in[?]?$", re.I),
    re.compile(r"^in which continent is [^?]+ not[?]?$", re.I),
    re.compile(r"^in which medium does [^?]+ not travel fastest[?]?$", re.I),
    re.compile(r"^in which mountain range is [^?]+ not[?]?$", re.I),
    re.compile(r"^what do [^?]+ not use as primary input for [^?]+[?]?$", re.I),
    re.compile(r"^what does gravity not cause objects to do[?]?$", re.I),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def featurize(record: dict[str, Any], include_rule: bool = False) -> str:
    candidates = " || ".join(candidate_pool(record))
    parts = [
        "POS: " + str(record.get("prompt_pos", "")),
        "NEG: " + str(record.get("prompt_neg", "")),
        "MODE: " + str(record.get("semantic_mode", "")),
        "SCOPE: " + str(record.get("scope_type", "")),
        "CAND: " + candidates,
    ]
    if include_rule:
        parts.append("RULE: " + predict_behavior(record))
    return "\n".join(parts)


def text_derived_suppression_only(record: dict[str, Any]) -> bool:
    text = " ".join(str(record.get("prompt_neg", "")).lower().strip().split())
    return any(pattern.search(text) for pattern in TEXT_DERIVED_SUPPRESSION_ONLY_PATTERNS)


def make_binary_model(class_weight: dict[int, float] | str = "balanced", c: float = 2.0) -> Pipeline:
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
        class_weight=class_weight,
        C=c,
        solver="lbfgs",
        random_state=0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def positive_proba(model: Pipeline, texts: list[str]) -> list[float]:
    classes = list(model.named_steps["classifier"].classes_)
    pos_index = classes.index(1)
    return [float(row[pos_index]) for row in model.predict_proba(texts)]


def summarize(records: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    correct = sum(labels[r["id"]] == r["expected_neg_behavior"] for r in records)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, dict[str, Any]] = {}
    for record in records:
        gold = record["expected_neg_behavior"]
        pred = labels[record["id"]]
        confusion[gold][pred] += 1
    for behavior in TOKEN_MAP:
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(labels[r["id"]] == behavior for r in subset)
        per_behavior[behavior] = {
            "correct": n_correct,
            "total": len(subset),
            "accuracy": n_correct / len(subset) if subset else 0.0,
        }
    preserve = per_behavior["preserve_positive"]["accuracy"]
    suppress = per_behavior["suppress_target"]["accuracy"]
    select = per_behavior["select_gold_neg"]["accuracy"]
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "balanced_behavior_accuracy": (preserve + suppress + select) / 3.0,
        "per_behavior": per_behavior,
        "confusion": {key: dict(value) for key, value in confusion.items()},
    }


def predict_labels(
    records: list[dict[str, Any]],
    preserve_model: Pipeline,
    branch_model: Pipeline,
    threshold: float,
    include_rule: bool,
    force_suppression_only: bool = False,
    text_suppression_guard: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    texts = [featurize(record, include_rule=include_rule) for record in records]
    preserve_probs = positive_proba(preserve_model, texts)
    branch_probs = positive_proba(branch_model, texts)
    labels: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for record, preserve_prob, suppress_prob in zip(records, preserve_probs, branch_probs, strict=True):
        metadata_guard_hit = force_suppression_only and record.get("semantic_mode") == "suppression_only"
        text_guard_hit = text_suppression_guard and text_derived_suppression_only(record)
        if metadata_guard_hit or text_guard_hit:
            label = "suppress_target"
        elif preserve_prob >= threshold:
            label = "preserve_positive"
        elif suppress_prob >= 0.5:
            label = "suppress_target"
        else:
            label = "select_gold_neg"
        labels[record["id"]] = label
        scores[record["id"]] = {
            "preserve_prob": preserve_prob,
            "suppress_prob_given_nonpreserve": suppress_prob,
            "metadata_suppression_guard_hit": metadata_guard_hit,
            "text_suppression_guard_hit": text_guard_hit,
        }
    return labels, scores


def tokens_from_labels(labels: dict[str, str], empty_select: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for record_id, label in labels.items():
        if empty_select and label == "select_gold_neg":
            out[record_id] = ""
        else:
            out[record_id] = TOKEN_MAP[label]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/router_calibration/train_fit_clean_strict.jsonl")
    parser.add_argument("--calibration", default="data/processed/splits/router_calibration/calibration_clean_strict.jsonl")
    parser.add_argument("--test", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--output-prefix", default="outputs/supervised_router_twostage")
    parser.add_argument("--include-rule", action="store_true")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--preserve-weight", type=float, default=3.0)
    parser.add_argument(
        "--force-suppression-only",
        action="store_true",
        help="Route records with semantic_mode=suppression_only to SUPPRESS before classifier decisions.",
    )
    parser.add_argument(
        "--text-suppression-guard",
        action="store_true",
        help="Route factual suppress-only prompts inferred from prompt text to SUPPRESS before classifier decisions.",
    )
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train))
    calibration_records = load_jsonl(Path(args.calibration))
    test_records = load_jsonl(Path(args.test))

    train_texts = [featurize(record, include_rule=args.include_rule) for record in train_records]
    y_preserve = [int(record["expected_neg_behavior"] == "preserve_positive") for record in train_records]
    preserve_model = make_binary_model(class_weight={0: 1.0, 1: args.preserve_weight})
    preserve_model.fit(train_texts, y_preserve)

    branch_train = [r for r in train_records if r["expected_neg_behavior"] != "preserve_positive"]
    branch_texts = [featurize(record, include_rule=args.include_rule) for record in branch_train]
    y_suppress = [int(record["expected_neg_behavior"] == "suppress_target") for record in branch_train]
    branch_model = make_binary_model(class_weight="balanced")
    branch_model.fit(branch_texts, y_suppress)

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    predictions_by_threshold: dict[str, dict[str, str]] = {}
    scores_by_threshold: dict[str, dict[str, dict[str, float]]] = {}
    for threshold in thresholds:
        labels, scores = predict_labels(
            calibration_records,
            preserve_model,
            branch_model,
            threshold,
            include_rule=args.include_rule,
            force_suppression_only=args.force_suppression_only,
            text_suppression_guard=args.text_suppression_guard,
        )
        summary = summarize(calibration_records, labels)
        preserve_recall = summary["per_behavior"]["preserve_positive"]["accuracy"]
        # Prefer preserve recall, but keep non-preserve branches from collapsing.
        objective = (
            0.50 * preserve_recall
            + 0.25 * summary["per_behavior"]["select_gold_neg"]["accuracy"]
            + 0.25 * summary["per_behavior"]["suppress_target"]["accuracy"]
        )
        rows.append({"threshold": threshold, "objective": objective, "summary": summary})
        predictions_by_threshold[f"{threshold:.2f}"] = labels
        scores_by_threshold[f"{threshold:.2f}"] = scores

    best = max(rows, key=lambda row: (row["objective"], row["summary"]["balanced_behavior_accuracy"]))
    best_threshold = float(best["threshold"])
    test_labels, test_scores = predict_labels(
        test_records,
        preserve_model,
        branch_model,
        best_threshold,
        include_rule=args.include_rule,
        force_suppression_only=args.force_suppression_only,
        text_suppression_guard=args.text_suppression_guard,
    )
    test_summary = summarize(test_records, test_labels)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(str(prefix) + "_strict_predictions.json")
    predictions_empty_path = Path(str(prefix) + "_strict_empty_select_predictions.json")
    report_path = Path(str(prefix) + "_report.json")
    scores_path = Path(str(prefix) + "_strict_scores.json")

    predictions_path.write_text(
        json.dumps(tokens_from_labels(test_labels, empty_select=False), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_empty_path.write_text(
        json.dumps(tokens_from_labels(test_labels, empty_select=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scores_path.write_text(json.dumps(test_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "train": args.train,
        "calibration": args.calibration,
        "test": args.test,
        "include_rule": args.include_rule,
        "force_suppression_only": args.force_suppression_only,
        "text_suppression_guard": args.text_suppression_guard,
        "preserve_weight": args.preserve_weight,
        "train_counts": dict(Counter(r["expected_neg_behavior"] for r in train_records)),
        "calibration_counts": dict(Counter(r["expected_neg_behavior"] for r in calibration_records)),
        "test_counts": dict(Counter(r["expected_neg_behavior"] for r in test_records)),
        "threshold_search": rows,
        "best_threshold": best_threshold,
        "test_summary": test_summary,
        "test_label_predictions": test_labels,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best threshold: {best_threshold:.2f}")
    print(
        "Calibration: "
        f"acc={best['summary']['accuracy'] * 100:.1f}% "
        f"balanced={best['summary']['balanced_behavior_accuracy'] * 100:.1f}%"
    )
    print(
        "Strict labels: "
        f"acc={test_summary['accuracy'] * 100:.1f}% "
        f"balanced={test_summary['balanced_behavior_accuracy'] * 100:.1f}%"
    )
    for behavior, row in test_summary["per_behavior"].items():
        print(f"  {behavior}: {row['correct']}/{row['total']} = {row['accuracy'] * 100:.1f}%")
    print("Confusion:", test_summary["confusion"])
    print(f"Saved: {predictions_path}")
    print(f"Saved: {predictions_empty_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
