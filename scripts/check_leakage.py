from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.schema import normalize_text  # noqa: E402


CHECK_FIELDS = (
    "record_id",
    "prompt_pair",
    "template_id",
    "family_id",
    "entity_id",
    "holdout_group",
)
DEFAULT_FAIL_ON = "record_id,prompt_pair,holdout_group"


def split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--split-input must be SPLIT=PATH")
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("split name must be non-empty")
    return split, Path(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    return normalize_text(str(value))


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_scalar(row.get(key))
        if value:
            return value
    return ""


def record_values(row: dict[str, Any]) -> dict[str, list[str]]:
    pos_prompt = first_present(row, ("prompt_pos", "positive_prompt"))
    neg_prompt = first_present(row, ("prompt_neg", "negative_prompt"))
    prompt_pair = f"{pos_prompt} || {neg_prompt}" if pos_prompt or neg_prompt else ""

    values = {
        "record_id": [clean_scalar(row.get("id"))],
        "prompt_pair": [prompt_pair],
        "template_id": [clean_scalar(row.get("template_id"))],
        "family_id": [first_present(row, ("family_id", "entity_family"))],
        "entity_id": [clean_scalar(row.get("entity_id"))],
        "holdout_group": [clean_scalar(row.get("holdout_group"))],
    }
    return {
        field: [value for value in field_values if value]
        for field, field_values in values.items()
    }


def load_split_rows(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: dict[str, str] = {}

    if args.input:
        path = Path(args.input)
        inputs["release_input"] = str(path)
        for row in load_jsonl(path):
            split = str(row.get("split", "")).strip()
            if not split:
                raise SystemExit(f"{path}: row {row.get('id', '<missing id>')} has no split")
            split_rows[split].append(row)

    for split, path in args.split_input or []:
        inputs[split] = str(path)
        split_rows[split].extend(load_jsonl(path))

    return dict(split_rows), inputs


def parse_fail_on(value: str) -> set[str]:
    if value.strip().lower() in {"", "none"}:
        return set()
    if value.strip().lower() == "all":
        return set(CHECK_FIELDS)
    fields = {item.strip() for item in value.split(",") if item.strip()}
    unknown = fields - set(CHECK_FIELDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown --fail-on field(s): {', '.join(sorted(unknown))}"
        )
    return fields


def collect_values(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    by_split: dict[str, dict[str, dict[str, list[str]]]] = {}
    for split, rows in split_rows.items():
        field_map: dict[str, dict[str, list[str]]] = {
            field: defaultdict(list) for field in CHECK_FIELDS
        }
        for row in rows:
            record_id = str(row.get("id", "<missing id>"))
            for field, values in record_values(row).items():
                for value in values:
                    field_map[field][value].append(record_id)
        by_split[split] = {field: dict(values) for field, values in field_map.items()}
    return by_split


def sample_overlaps(
    left_values: dict[str, list[str]],
    right_values: dict[str, list[str]],
    left_split: str,
    right_split: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for value in sorted(set(left_values) & set(right_values))[:max_samples]:
        samples.append(
            {
                "value": value[:240],
                left_split: left_values[value][:5],
                right_split: right_values[value][:5],
            }
        )
    return samples


def build_report(
    split_rows: dict[str, list[dict[str, Any]]],
    inputs: dict[str, str],
    fail_on: set[str],
    max_samples: int,
) -> dict[str, Any]:
    values = collect_values(split_rows)
    split_order = {"train": 0, "dev": 1, "seen_eval": 2, "test": 3, "unseen_test": 4}
    split_names = sorted(split_rows, key=lambda name: (split_order.get(name, 99), name))
    overlaps: dict[str, dict[str, Any]] = {}
    failed_checks: list[dict[str, Any]] = []

    for left, right in combinations(split_names, 2):
        pair_key = f"{left}_vs_{right}"
        pair_report: dict[str, Any] = {}
        for field in CHECK_FIELDS:
            left_values = values[left][field]
            right_values = values[right][field]
            common = set(left_values) & set(right_values)
            field_report = {
                "count": len(common),
                "samples": sample_overlaps(
                    left_values,
                    right_values,
                    left,
                    right,
                    max_samples,
                ),
            }
            pair_report[field] = field_report
            if field in fail_on and common:
                failed_checks.append(
                    {
                        "split_pair": pair_key,
                        "field": field,
                        "count": len(common),
                        "samples": field_report["samples"],
                    }
                )
        overlaps[pair_key] = pair_report

    return {
        "inputs": inputs,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "checked_fields": list(CHECK_FIELDS),
        "fail_on": sorted(fail_on),
        "overlaps": overlaps,
        "failed_checks": failed_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check train/dev/test leakage for E4 split files or E4v3 release JSONL."
    )
    parser.add_argument(
        "--input",
        help="Single JSONL with a split field, e.g. an E4v3 release file.",
    )
    parser.add_argument(
        "--split-input",
        action="append",
        type=split_arg,
        default=[],
        help="Explicit split file as SPLIT=PATH. Can be repeated.",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--fail-on",
        default=DEFAULT_FAIL_ON,
        help=(
            "Comma-separated fields that make the check fail on overlap. "
            f"Known fields: {', '.join(CHECK_FIELDS)}. "
            "Use 'all' or 'none'. Default: record_id,prompt_pair,holdout_group."
        ),
    )
    args = parser.parse_args()

    if not args.input and not args.split_input:
        raise SystemExit("provide --input or at least one --split-input")

    try:
        fail_on = parse_fail_on(args.fail_on)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    split_rows, inputs = load_split_rows(args)
    if len(split_rows) < 2:
        raise SystemExit("need at least two splits to check leakage")

    report = build_report(split_rows, inputs, fail_on, args.max_samples)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Split counts:", report["split_counts"])
    print("Fail-on fields:", ", ".join(report["fail_on"]) or "none")
    for failed in report["failed_checks"]:
        print(
            f"LEAK {failed['split_pair']} {failed['field']}: {failed['count']}"
        )
    print(f"Report: {args.report}")

    if report["failed_checks"]:
        raise SystemExit("Leakage check failed")


if __name__ == "__main__":
    main()
