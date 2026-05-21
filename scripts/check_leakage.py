"""Check overlap and field coverage across E4/E4v3 splits.

The checker accepts either separate E4-style split files via --split-input or a
single E4v3 release JSONL via --input. Fields listed in --fail-on are strict:
any missing value in any split fails, and any cross-split overlap fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.schema import normalize_text  # noqa: E402


FIELDS = {
    "prompt_pair",
    "template_id",
    "family_id",
    "entity_id",
    "holdout_group",
    "record_id",
    "source_record_id",
}
DEFAULT_FAIL_ON = ["prompt_pair"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--split-input must be SPLIT=PATH")
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("split name must be non-empty")
    return split, Path(path)


def nested_get(row: dict[str, Any], path: list[str]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_text(row: dict[str, Any], paths: list[list[str]]) -> str:
    for path in paths:
        value = nested_get(row, path)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def record_value(row: dict[str, Any], field: str) -> str:
    if field == "prompt_pair":
        pos = first_text(row, [["prompt_pos"], ["positive_prompt"]])
        neg = first_text(row, [["prompt_neg"], ["negative_prompt"]])
        if not pos or not neg:
            return ""
        return f"{normalize_text(pos)} || {normalize_text(neg)}"
    if field == "family_id":
        return first_text(row, [["family_id"], ["entity_family"]])
    if field == "entity_id":
        return first_text(
            row,
            [
                ["entity_id"],
                ["metadata", "source_metadata", "entity_id"],
                ["metadata", "entity_id"],
            ],
        )
    if field == "record_id":
        return first_text(row, [["id"]])
    if field == "source_record_id":
        return first_text(
            row,
            [
                ["metadata", "source_id"],
                ["metadata", "release_id_renamed_from"],
                ["id"],
            ],
        )
    return first_text(row, [[field]])


def load_split_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    if args.input and args.split_input:
        raise SystemExit("Use either --input or --split-input, not both")
    if not args.input and not args.split_input:
        raise SystemExit("Provide --input or at least two --split-input values")

    if args.input:
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for row in read_jsonl(Path(args.input)):
            split = str(row.get("split") or "all").strip() or "all"
            rows_by_split.setdefault(split, []).append(row)
        return rows_by_split

    rows_by_split = {}
    for split, path in args.split_input:
        rows_by_split.setdefault(split, []).extend(read_jsonl(path))
    return rows_by_split


def parse_fail_on(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_FAIL_ON)
    fields: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item == "all":
                fields.update(FIELDS)
            elif item in FIELDS:
                fields.add(item)
            else:
                raise SystemExit(f"unsupported --fail-on field: {item}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Combined release JSONL with a split field.")
    parser.add_argument("--split-input", action="append", type=split_input)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--fail-on",
        action="append",
        help=(
            "Field that must be complete and non-overlapping across splits. "
            "Repeat, comma-separate, or use all. Default: prompt_pair."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    fail_on = parse_fail_on(args.fail_on)
    rows_by_split = load_split_rows(args)
    if len(rows_by_split) < 2:
        raise SystemExit("Need at least two splits to check leakage")

    field_coverage: dict[str, dict[str, dict[str, int]]] = {}
    values: dict[str, dict[str, dict[str, list[str]]]] = {}
    failed_coverage: list[dict[str, Any]] = []

    for split, rows in rows_by_split.items():
        field_coverage[split] = {}
        values[split] = {}
        for field in sorted(FIELDS):
            counter: dict[str, list[str]] = {}
            missing = 0
            for row in rows:
                value = record_value(row, field)
                if not value:
                    missing += 1
                    continue
                counter.setdefault(value, []).append(str(row.get("id", "")))
            present = len(rows) - missing
            field_coverage[split][field] = {"present": present, "missing": missing}
            values[split][field] = counter
            if field in fail_on and missing:
                failed_coverage.append(
                    {
                        "split": split,
                        "field": field,
                        "missing": missing,
                        "present": present,
                    }
                )

    overlaps: list[dict[str, Any]] = []
    failed_overlaps: list[dict[str, Any]] = []
    for left, right in combinations(sorted(rows_by_split), 2):
        for field in sorted(FIELDS):
            shared = sorted(set(values[left][field]) & set(values[right][field]))
            item = {
                "left": left,
                "right": right,
                "field": field,
                "count": len(shared),
                "examples": [
                    {
                        "value": value,
                        "left_ids": values[left][field][value][: args.max_examples],
                        "right_ids": values[right][field][value][: args.max_examples],
                    }
                    for value in shared[: args.max_examples]
                ],
            }
            overlaps.append(item)
            if field in fail_on and shared:
                failed_overlaps.append(item)

    report = {
        "input": args.input,
        "split_inputs": {
            split: str(path) for split, path in (args.split_input or [])
        },
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "fail_on": sorted(fail_on),
        "field_coverage": field_coverage,
        "overlaps": overlaps,
        "failed_coverage": failed_coverage,
        "failed_overlaps": failed_overlaps,
        "ok": not failed_coverage and not failed_overlaps,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Splits:", report["splits"])
    print("Fail-on:", ", ".join(report["fail_on"]) or "(none)")
    print("Failed coverage:", len(failed_coverage))
    print("Failed overlaps:", len(failed_overlaps))
    print(f"Saved: {out_path}")
    if not report["ok"]:
        raise SystemExit("Leakage check failed")


if __name__ == "__main__":
    main()
