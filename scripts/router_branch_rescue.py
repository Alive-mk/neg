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


def featurize(record: dict[str, Any], llm_token: str) -> str:
    candidates = " || ".join(candidate_pool(record))
    parts = [
        "POS: " + str(record.get("prompt_pos", "")),
        "NEG: " + str(record.get("prompt_neg", "")),
        "CAND: " + candidates,
        "RULE: " + predict_behavior(record),
        "LLM: " + llm_token,
    ]
    return "\n".join(parts)


def make_model(C: float = 2.0) -> Pipeline:
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
        C=C,
        solver="lbfgs",
        random_state=0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def summarize(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    correct = sum(
        ("[SELECT]" if predictions[r["id"]] == "" else predictions[r["id"]]) == TOKEN_MAP[r["expected_neg_behavior"]]
        for r in records
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_behavior: dict[str, dict[str, Any]] = {}
    for record in records:
        confusion[record["expected_neg_behavior"]][predictions[record["id"]]] += 1
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
    }


def branch_predict(
    records: list[dict[str, Any]],
    llm_predictions: dict[str, str],
    select_model: Pipeline,
    nonselect_model: Pipeline,
    threshold: float,
) -> tuple[dict[str, str], dict[str, dict[str, float | str]]]:
    predictions: dict[str, str] = {}
    details: dict[str, dict[str, float | str]] = {}
    for record in records:
        rid = record["id"]
        llm_token = llm_predictions[rid]
        if llm_token != "[SELECT]":
            predictions[rid] = llm_token
            details[rid] = {
                "llm_token": llm_token,
                "keep_select_prob": 0.0,
                "route": "keep_llm_nonselect",
            }
            continue

        feat = featurize(record, llm_token)
        keep_select_prob = float(select_model.predict_proba([feat])[0][1])
        if keep_select_prob >= threshold:
            predictions[rid] = ""
            details[rid] = {
                "llm_token": llm_token,
                "keep_select_prob": keep_select_prob,
                "route": "keep_select_empty_prefix",
            }
            continue

        nonselect_label = nonselect_model.predict([feat])[0]
        predictions[rid] = nonselect_label
        details[rid] = {
            "llm_token": llm_token,
            "keep_select_prob": keep_select_prob,
            "route": "override_nonselect",
            "override_label": nonselect_label,
        }
    return predictions, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/splits/merged_excl_boost/train_clean_strict.jsonl")
    parser.add_argument("--dev", default="data/processed/splits/balanced/dev_clean_strict.jsonl")
    parser.add_argument("--test", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--llm-train", required=True)
    parser.add_argument("--llm-dev", required=True)
    parser.add_argument("--llm-test", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    parser.add_argument("--output-dir", default="outputs/router_branch_rescue")
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train))
    dev_records = load_jsonl(Path(args.dev))
    test_records = load_jsonl(Path(args.test))
    llm_train = load_json(Path(args.llm_train))
    llm_dev = load_json(Path(args.llm_dev))
    llm_test = load_json(Path(args.llm_test))

    train_select_branch = [r for r in train_records if llm_train[r["id"]] == "[SELECT]"]
    select_model = make_model(C=1.5)
    select_model.fit(
        [featurize(r, llm_train[r["id"]]) for r in train_select_branch],
        [int(r["expected_neg_behavior"] == "select_gold_neg") for r in train_select_branch],
    )

    train_nonselect = [r for r in train_records if r["expected_neg_behavior"] != "select_gold_neg"]
    nonselect_model = make_model(C=2.0)
    nonselect_model.fit(
        [featurize(r, llm_train.get(r["id"], "[SELECT]")) for r in train_nonselect],
        [TOKEN_MAP[r["expected_neg_behavior"]] for r in train_nonselect],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "train_branch_size": len(train_select_branch),
        "train_branch_gold": dict(Counter(r["expected_neg_behavior"] for r in train_select_branch)),
        "thresholds": {},
    }

    for threshold in args.thresholds:
        key = f"{threshold:.2f}"
        dev_predictions, dev_details = branch_predict(dev_records, llm_dev, select_model, nonselect_model, threshold)
        test_predictions, test_details = branch_predict(test_records, llm_test, select_model, nonselect_model, threshold)

        dev_path = out_dir / f"dev_predictions_t{key}.json"
        test_path = out_dir / f"test_predictions_t{key}.json"
        dev_path.write_text(json.dumps(dev_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        test_path.write_text(json.dumps(test_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"dev_details_t{key}.json").write_text(
            json.dumps(dev_details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"test_details_t{key}.json").write_text(
            json.dumps(test_details, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        report["thresholds"][key] = {
            "dev_summary": summarize(dev_records, dev_predictions),
            "test_summary": summarize(test_records, test_predictions),
            "dev_predictions": str(dev_path),
            "test_predictions": str(test_path),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, row in report["thresholds"].items():
        dev = row["dev_summary"]
        test = row["test_summary"]
        print(
            f"t={key} "
            f"dev={dev['accuracy']*100:5.1f}% "
            f"(sup={dev['per_behavior']['suppress_target']['accuracy']*100:5.1f} "
            f"pre={dev['per_behavior']['preserve_positive']['accuracy']*100:5.1f} "
            f"sel={dev['per_behavior']['select_gold_neg']['accuracy']*100:5.1f}) "
            f"test={test['accuracy']*100:5.1f}% "
            f"(sup={test['per_behavior']['suppress_target']['accuracy']*100:5.1f} "
            f"pre={test['per_behavior']['preserve_positive']['accuracy']*100:5.1f} "
            f"sel={test['per_behavior']['select_gold_neg']['accuracy']*100:5.1f})"
        )
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
