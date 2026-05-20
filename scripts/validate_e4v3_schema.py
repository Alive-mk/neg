from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "behavior",
    "domain",
    "template_id",
    "entity_family",
    "positive_prompt",
    "negative_prompt",
    "positive_gold",
    "negative_gold_single",
    "negative_gold_multi",
    "candidate_pool",
    "valid_negative_candidates",
    "invalid_negative_candidates",
    "ambiguity_flag",
    "split",
    "holdout_group",
}

BEHAVIORS = {"suppress", "preserve", "select"}


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(record))
    for field in missing:
        errors.append(f"missing field: {field}")
    if missing:
        return errors

    behavior = str(record.get("behavior", ""))
    if behavior not in BEHAVIORS:
        errors.append(f"invalid behavior={behavior}")

    for field in (
        "id",
        "domain",
        "template_id",
        "entity_family",
        "positive_prompt",
        "negative_prompt",
        "negative_gold_single",
        "split",
        "holdout_group",
    ):
        if not str(record.get(field, "")).strip():
            errors.append(f"{field} must be non-empty")

    positive_gold = as_list(record.get("positive_gold"))
    negative_multi = as_list(record.get("negative_gold_multi"))
    candidates = as_list(record.get("candidate_pool"))
    valid_negatives = as_list(record.get("valid_negative_candidates"))
    invalid_negatives = as_list(record.get("invalid_negative_candidates"))
    negative_single = str(record.get("negative_gold_single", "")).strip()

    if not positive_gold:
        errors.append("positive_gold must be non-empty")
    if behavior == "select" and not negative_single:
        errors.append("select requires negative_gold_single")
    if behavior == "select" and not negative_multi:
        errors.append("select requires negative_gold_multi")
    if behavior == "select" and not valid_negatives:
        errors.append("select requires valid_negative_candidates")
    if not candidates:
        errors.append("candidate_pool must be non-empty")

    norm_pos = {normalize(item) for item in positive_gold}
    norm_multi = {normalize(item) for item in negative_multi}
    norm_candidates = {normalize(item) for item in candidates}
    norm_valid = {normalize(item) for item in valid_negatives}
    norm_invalid = {normalize(item) for item in invalid_negatives}
    norm_single = normalize(negative_single)

    if norm_single and norm_single not in norm_multi:
        errors.append("negative_gold_multi must include negative_gold_single")
    if norm_single and norm_single not in norm_candidates:
        errors.append("candidate_pool must include negative_gold_single")
    if not norm_valid.issubset(norm_candidates):
        errors.append("valid_negative_candidates must be subset of candidate_pool")
    if not norm_valid.issubset(norm_multi):
        errors.append("negative_gold_multi must include valid_negative_candidates")
    if not norm_invalid.issubset(norm_candidates):
        errors.append("invalid_negative_candidates must be subset of candidate_pool")
    if norm_valid & norm_invalid:
        errors.append("valid_negative_candidates overlap invalid_negative_candidates")
    if norm_pos & norm_valid:
        errors.append("positive_gold overlaps valid_negative_candidates")

    ambiguity_flag = bool(record.get("ambiguity_flag"))
    if behavior == "select":
        is_ambiguous = len(norm_multi) > 1 or len(norm_valid) > 1
        if ambiguity_flag != is_ambiguous:
            errors.append("ambiguity_flag does not match multi-answer labels")

    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-invalid-samples", type=int, default=50)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    errors_by_type: Counter[str] = Counter()
    invalid_samples = []
    invalid_count = 0
    ids: Counter[str] = Counter()
    split_behavior: dict[str, Counter[str]] = {}
    templates_by_split: dict[str, set[str]] = {}
    holdout_by_split: dict[str, set[str]] = {}

    for row in rows:
        record_id = str(row.get("id", ""))
        ids[record_id] += 1
        split = str(row.get("split", ""))
        behavior = str(row.get("behavior", ""))
        split_behavior.setdefault(split, Counter())[behavior] += 1
        templates_by_split.setdefault(split, set()).add(str(row.get("template_id", "")))
        holdout_by_split.setdefault(split, set()).add(str(row.get("holdout_group", "")))

        errors = validate_record(row)
        if ids[record_id] > 1:
            errors.append(f"duplicate id: {record_id}")
        for error in errors:
            errors_by_type[error] += 1
        if errors:
            invalid_count += 1
            if len(invalid_samples) < args.max_invalid_samples:
                invalid_samples.append({"id": record_id, "errors": errors})

    train_templates = templates_by_split.get("train", set())
    test_templates = templates_by_split.get("test", set())
    train_holdout = holdout_by_split.get("train", set())
    test_holdout = holdout_by_split.get("test", set())
    report = {
        "input": args.input,
        "rows": len(rows),
        "valid_rows": len(rows) - invalid_count,
        "invalid_rows": invalid_count,
        "error_counts": dict(errors_by_type),
        "invalid_samples": invalid_samples,
        "split_behavior_counts": {
            split: dict(counts) for split, counts in split_behavior.items()
        },
        "split_template_counts": {
            split: len(values) for split, values in templates_by_split.items()
        },
        "template_overlap_train_test": len(train_templates & test_templates),
        "holdout_overlap_train_test": len(train_holdout & test_holdout),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(rows)}")
    print(f"Error types: {len(errors_by_type)}")
    print("Split/behavior:", report["split_behavior_counts"])
    print(f"Train/test template overlap: {report['template_overlap_train_test']}")
    print(f"Train/test holdout overlap: {report['holdout_overlap_train_test']}")
    print(f"Saved: {args.report}")
    if errors_by_type:
        raise SystemExit("E4 v3 schema validation failed")


if __name__ == "__main__":
    main()
