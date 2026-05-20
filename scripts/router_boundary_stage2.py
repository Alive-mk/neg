"""
Second-stage boundary router for v4 candidate-aware predictions.

The first-stage v4 router improves PRESERVE recall but over-predicts PRESERVE
on query-like SELECT items. This script trains a lightweight router-only
classifier with audited boundary sentinels, then only revisits records that the
first stage predicted as PRESERVE.
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


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}

QUERY_LIKE_RE = re.compile(
    r"^\s*(what|which|who|where|when|select|choose|identify)\b[^?!.]*\bnot\b",
    re.I,
)
DIRECT_PROHIBITION_RE = re.compile(r"^\s*(do\s+not|don't|avoid)\b", re.I)


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


def token_for(record: dict[str, Any]) -> str:
    return TOKEN_MAP[record["expected_neg_behavior"]]


def featurize(record: dict[str, Any]) -> str:
    text = str(record.get("prompt_neg", ""))
    lower = text.lower()
    meta = [
        "RULE=" + predict_behavior(record),
        "QUERY_LIKE=" + str(bool(QUERY_LIKE_RE.search(text))),
        "DIRECT_PROHIBITION=" + str(bool(DIRECT_PROHIBITION_RE.search(text))),
        "HAS_QMARK=" + str("?" in text),
        "START_WH=" + str(bool(re.match(r"^\s*(what|which|who|where|when)\b", text, re.I))),
        "START_AUX=" + str(bool(re.match(r"^\s*(is|does|do|did|should|can|are|was|were|will)\b", text, re.I))),
        "HAS_DOUBLE_NEG=" + str(
            bool(
                re.search(
                    r"\bnot\s+not\b|\bnot true that\b|\bnot without\b|\bnot incorrect\b|\bnot uncommon\b|\bnot fail\b|\bdid(?:n't| not) fail\b",
                    text,
                    re.I,
                )
            )
        ),
        "HAS_WITHOUT=" + str(" without " in f" {lower} "),
    ]
    return "\n".join(
        [
            "POS: " + str(record.get("prompt_pos", "")),
            "NEG: " + text,
            "CAND: " + " || ".join(candidate_pool(record)),
            "META: " + " ".join(meta),
        ]
    )


def make_model(c_value: float) -> Pipeline:
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
        C=c_value,
        solver="lbfgs",
        random_state=0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def router_only_audit_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        bucket = str((record.get("metadata") or {}).get("bucket") or record.get("bucket") or "")
        text = str(record.get("prompt_neg", ""))
        copied = dict(record)
        metadata = dict(copied.get("metadata") or {})
        metadata["router_only_relabel"] = True
        metadata["router_only_original_behavior"] = copied.get("expected_neg_behavior")
        copied["metadata"] = metadata

        if bucket in {"audit_preserve_query_like", "audit_suppress_query_like"} and QUERY_LIKE_RE.search(text):
            copied["expected_neg_behavior"] = "select_gold_neg"
            examples.append(copied)
        elif DIRECT_PROHIBITION_RE.search(text):
            copied["expected_neg_behavior"] = "suppress_target"
            examples.append(copied)
    return examples


def summarize(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, dict[str, Any]] = {}
    for record in records:
        pred = "[SELECT]" if predictions[record["id"]] == "" else predictions[record["id"]]
        gold = token_for(record)
        correct += int(pred == gold)
        confusion[record["expected_neg_behavior"]][pred] += 1
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(
            ("[SELECT]" if predictions[r["id"]] == "" else predictions[r["id"]]) == token
            for r in subset
        )
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
        "prediction_distribution": dict(Counter(predictions.values())),
    }


def predict_stage2(
    records: list[dict[str, Any]],
    first_stage: dict[str, str],
    model: Pipeline,
    preserve_threshold: float,
    empty_select: bool,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    predictions: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}
    classes = list(model.named_steps["classifier"].classes_)
    preserve_idx = classes.index("[PRESERVE]")
    for record in records:
        rid = record["id"]
        first = first_stage[rid]
        if first != "[PRESERVE]":
            predictions[rid] = "" if empty_select and first == "[SELECT]" else first
            details[rid] = {"route": "keep_first_stage", "first_stage": first}
            continue

        probs = model.predict_proba([featurize(record)])[0]
        pred = str(model.predict([featurize(record)])[0])
        preserve_prob = float(probs[preserve_idx])
        if pred != "[PRESERVE]" and preserve_prob < preserve_threshold:
            predictions[rid] = "" if empty_select and pred == "[SELECT]" else pred
            details[rid] = {
                "route": "stage2_override",
                "first_stage": first,
                "stage2_pred": pred,
                "preserve_prob": preserve_prob,
            }
        else:
            predictions[rid] = "[PRESERVE]"
            details[rid] = {
                "route": "keep_preserve",
                "first_stage": first,
                "stage2_pred": pred,
                "preserve_prob": preserve_prob,
            }
    return predictions, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/router_targeted_retraining_pack/train_clean_strict_plus_routerfix19.jsonl")
    parser.add_argument("--audit", default="data/processed/splits/router_targeted_retraining_pack/audit_boundary_candidates.jsonl")
    parser.add_argument("--calibration", default="data/processed/splits/router_calibration/calibration_clean_strict.jsonl")
    parser.add_argument("--strict", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--first-calibration", default="outputs/llm_router_candidate_v4_calibration_predictions.json")
    parser.add_argument("--first-strict", default="outputs/llm_router_candidate_v4_strict_predictions.json")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--output-dir", default="outputs/router_boundary_stage2")
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train))
    audit_records = load_jsonl(Path(args.audit))
    router_audit = router_only_audit_examples(audit_records)
    fit_records = train_records + router_audit

    calibration_records = load_jsonl(Path(args.calibration))
    strict_records = load_jsonl(Path(args.strict))
    first_calibration = load_json(Path(args.first_calibration))
    first_strict = load_json(Path(args.first_strict))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "train": args.train,
        "audit": args.audit,
        "fit_counts": dict(Counter(r["expected_neg_behavior"] for r in fit_records)),
        "router_audit_added": len(router_audit),
        "settings": {},
    }

    for c_value in args.c_values:
        model = make_model(c_value)
        model.fit([featurize(r) for r in fit_records], [token_for(r) for r in fit_records])
        for threshold in args.thresholds:
            key = f"c{c_value:g}_t{threshold:.2f}"
            cal_pred, cal_details = predict_stage2(
                calibration_records,
                first_calibration,
                model,
                preserve_threshold=threshold,
                empty_select=True,
            )
            strict_pred, strict_details = predict_stage2(
                strict_records,
                first_strict,
                model,
                preserve_threshold=threshold,
                empty_select=True,
            )
            cal_path = out_dir / f"calibration_predictions_{key}.json"
            strict_path = out_dir / f"strict_predictions_{key}.json"
            cal_path.write_text(json.dumps(cal_pred, ensure_ascii=False, indent=2), encoding="utf-8")
            strict_path.write_text(json.dumps(strict_pred, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / f"calibration_details_{key}.json").write_text(
                json.dumps(cal_details, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out_dir / f"strict_details_{key}.json").write_text(
                json.dumps(strict_details, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["settings"][key] = {
                "c_value": c_value,
                "preserve_threshold": threshold,
                "calibration_summary": summarize(calibration_records, cal_pred),
                "strict_summary": summarize(strict_records, strict_pred),
                "calibration_predictions": str(cal_path),
                "strict_predictions": str(strict_path),
            }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, row in report["settings"].items():
        cal = row["calibration_summary"]
        strict = row["strict_summary"]
        print(
            f"{key:12s} cal={cal['accuracy']*100:5.1f}% "
            f"(sup={cal['per_behavior']['suppress_target']['accuracy']*100:5.1f} "
            f"pre={cal['per_behavior']['preserve_positive']['accuracy']*100:5.1f} "
            f"sel={cal['per_behavior']['select_gold_neg']['accuracy']*100:5.1f}) "
            f"strict={strict['accuracy']*100:5.1f}% "
            f"(sup={strict['per_behavior']['suppress_target']['accuracy']*100:5.1f} "
            f"pre={strict['per_behavior']['preserve_positive']['accuracy']*100:5.1f} "
            f"sel={strict['per_behavior']['select_gold_neg']['accuracy']*100:5.1f})"
        )
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
