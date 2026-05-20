from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


CHECK_FIELDS = (
    "record_id",
    "source_record_id",
    "prompt_pair",
    "template_id",
    "family_id",
    "entity_id",
    "holdout_group",
)
DEFAULT_FAIL_FIELDS = {"record_id", "source_record_id", "prompt_pair", "holdout_group"}
SPLIT_ORDER = {"train": 0, "dev": 1, "seen_eval": 2, "test": 3, "unseen_test": 4}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def clean_scalar(value: Any) -> str:
    text = str(value or "").strip()
    return text


def clean_values(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            out.extend(clean_values(list(value)))
            continue
        text = clean_scalar(value)
        if text:
            out.append(text)
    return out


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_scalar(row.get(key))
        if value:
            return value
    return ""


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def row_values(row: dict[str, Any]) -> dict[str, list[str]]:
    metadata = row_metadata(row)
    prompt_pos = first_present(row, ("prompt_pos", "positive_prompt"))
    prompt_neg = first_present(row, ("prompt_neg", "negative_prompt"))
    source_ids = clean_values(
        [
            metadata.get("source_id"),
            metadata.get("release_id_renamed_from"),
            row.get("source_id"),
            row.get("source_record_id"),
        ]
    )
    if not source_ids:
        source_ids = clean_values([row.get("id")])
    return {
        "record_id": clean_values([row.get("id")]),
        "source_record_id": source_ids,
        "prompt_pair": clean_values(
            [f"{normalize_text(prompt_pos)} || {normalize_text(prompt_neg)}"]
            if prompt_pos or prompt_neg
            else []
        ),
        "template_id": clean_values([row.get("template_id")]),
        "family_id": clean_values([first_present(row, ("family_id", "entity_family"))]),
        "entity_id": clean_values([row.get("entity_id")]),
        "holdout_group": clean_values([row.get("holdout_group")]),
    }


def split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--split-input must be SPLIT=PATH")
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("split name must be non-empty")
    return split, Path(path)


def parse_fields(value: str) -> set[str]:
    if value == "all":
        return set(CHECK_FIELDS)
    if value == "none":
        return set()
    fields = {field.strip() for field in value.split(",") if field.strip()}
    unknown = sorted(fields - set(CHECK_FIELDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown fail-on fields: {', '.join(unknown)}")
    return fields


def load_split_rows(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: dict[str, str] = {}

    if args.input:
        inputs["combined"] = str(args.input)
        for row in load_jsonl(args.input):
            split = clean_scalar(row.get("split"))
            if not split:
                raise SystemExit(f"{args.input}: row {row.get('id', '<missing id>')} has no split")
            split_rows[split].append(row)

    for split, path in args.split_input or []:
        inputs[split] = str(path)
        split_rows[split].extend(load_jsonl(path))

    return dict(split_rows), inputs


def collect_field_values(
    split_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, list[str]]]], dict[str, dict[str, dict[str, int]]]]:
    values_by_split: dict[str, dict[str, dict[str, list[str]]]] = {}
    coverage: dict[str, dict[str, dict[str, int]]] = {}

    for split, rows in split_rows.items():
        split_values: dict[str, dict[str, list[str]]] = {
            field: defaultdict(list) for field in CHECK_FIELDS
        }
        split_coverage: dict[str, dict[str, int]] = {}
        for row in rows:
            record_id = clean_scalar(row.get("id")) or "<missing id>"
            extracted = row_values(row)
            for field in CHECK_FIELDS:
                field_values = extracted[field]
                if field_values:
                    for value in field_values:
                        split_values[field][value].append(record_id)
                else:
                    split_coverage.setdefault(field, {"present": 0, "missing": 0})["missing"] += 1
                    continue
                split_coverage.setdefault(field, {"present": 0, "missing": 0})["present"] += 1

        for field in CHECK_FIELDS:
            split_coverage.setdefault(field, {"present": 0, "missing": 0})
        values_by_split[split] = {
            field: dict(value_map) for field, value_map in split_values.items()
        }
        coverage[split] = split_coverage

    return values_by_split, coverage


def find_overlaps(
    values_by_split: dict[str, dict[str, dict[str, list[str]]]],
    split_names: list[str],
    max_samples: int,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    overlap_counts: dict[str, dict[str, int]] = {}
    overlap_samples: list[dict[str, Any]] = []

    for left, right in combinations(split_names, 2):
        pair_key = f"{left}_vs_{right}"
        overlap_counts[pair_key] = {}
        for field in CHECK_FIELDS:
            left_values = values_by_split[left][field]
            right_values = values_by_split[right][field]
            overlap = sorted(set(left_values) & set(right_values))
            overlap_counts[pair_key][field] = len(overlap)
            for value in overlap[:max_samples]:
                overlap_samples.append(
                    {
                        "split_pair": pair_key,
                        "field": field,
                        "value": value,
                        left: left_values[value][:max_samples],
                        right: right_values[value][:max_samples],
                    }
                )

    return overlap_counts, overlap_samples


def build_report(
    split_rows: dict[str, list[dict[str, Any]]],
    inputs: dict[str, str],
    fail_on: set[str],
    max_samples: int,
) -> dict[str, Any]:
    split_names = sorted(split_rows, key=lambda name: (SPLIT_ORDER.get(name, 99), name))
    values_by_split, coverage = collect_field_values(split_rows)
    overlap_counts, overlap_samples = find_overlaps(values_by_split, split_names, max_samples)

    failed_overlaps: list[dict[str, Any]] = []
    for pair_key, field_counts in overlap_counts.items():
        for field, count in field_counts.items():
            if field in fail_on and count:
                failed_overlaps.append(
                    {"split_pair": pair_key, "field": field, "count": count}
                )

    failed_coverage: list[dict[str, Any]] = []
    for split in split_names:
        for field in fail_on:
            if coverage[split][field]["present"] == 0:
                failed_coverage.append(
                    {
                        "split": split,
                        "field": field,
                        "reason": "no covered rows for fail-on field",
                    }
                )

    return {
        "inputs": inputs,
        "split_counts": {split: len(split_rows[split]) for split in split_names},
        "checked_fields": list(CHECK_FIELDS),
        "fail_on": sorted(fail_on),
        "field_coverage": coverage,
        "overlap_counts": overlap_counts,
        "overlap_samples": overlap_samples,
        "failed_overlaps": failed_overlaps,
        "failed_coverage": failed_coverage,
        "ok": not failed_overlaps and not failed_coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check train/dev/test leakage for E4 split files or E4v3 release JSONL."
    )
    parser.add_argument("--input", type=Path, help="Single JSONL with a split field.")
    parser.add_argument(
        "--split-input",
        action="append",
        type=split_arg,
        help="Explicit split file as SPLIT=PATH. Can be repeated.",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--fail-on",
        type=parse_fields,
        default=DEFAULT_FAIL_FIELDS,
        help=(
            "Comma-separated checked fields that should fail on overlap. "
            "Use 'all' or 'none'. Default: record_id,source_record_id,prompt_pair,holdout_group."
        ),
    )
    args = parser.parse_args()

    if not args.input and not args.split_input:
        raise SystemExit("provide --input or at least one --split-input")

    split_rows, inputs = load_split_rows(args)
    if len(split_rows) < 2:
        raise SystemExit("need at least two splits to check leakage")

    report = build_report(split_rows, inputs, args.fail_on, args.max_samples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Split counts:", report["split_counts"])
    print("Fail-on fields:", ", ".join(report["fail_on"]) if report["fail_on"] else "none")
    if report["failed_coverage"]:
        for failed in report["failed_coverage"]:
            print(f"MISSING {failed['split']} {failed['field']}: {failed['reason']}")
    if report["failed_overlaps"]:
        for failed in report["failed_overlaps"]:
            print(f"LEAK {failed['split_pair']} {failed['field']}: {failed['count']}")
    print(f"Saved report: {args.report}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
