from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import load_records  # noqa: E402
from neg_blindness.schema import ExperimentRecord, normalize_text  # noqa: E402


BEHAVIOR_MAP = {
    "suppress_target": "suppress",
    "preserve_positive": "preserve",
    "select_gold_neg": "select",
}


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def without(items: list[str], blocked: list[str]) -> list[str]:
    blocked_norm = {normalize_text(item) for item in blocked}
    return [item for item in unique(items) if normalize_text(item) not in blocked_norm]


def split_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--split-input must be SPLIT=PATH")
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError("split name must be non-empty")
    return split, Path(path)


def candidate_pool(record: ExperimentRecord) -> list[str]:
    return unique(
        record.gold_pos
        + record.gold_neg
        + record.valid_negatives
        + record.forbidden_neg
        + record.candidate_pool_neg
        + record.distractors
    )


def negative_labels(record: ExperimentRecord, pool: list[str]) -> tuple[str, list[str], list[str]]:
    if record.expected_neg_behavior == "select_gold_neg":
        multi = unique(record.gold_neg + record.valid_negatives)
        single = multi[0] if multi else ""
        return single, multi, multi

    if record.expected_neg_behavior == "preserve_positive":
        single = record.gold_pos[0] if record.gold_pos else (pool[0] if pool else "")
        return single, unique([single]), []

    valid_suppressions = without(record.candidate_pool_neg, record.forbidden_neg or record.gold_pos)
    if not valid_suppressions:
        valid_suppressions = without(pool, record.forbidden_neg or record.gold_pos)
    single = valid_suppressions[0] if valid_suppressions else (pool[0] if pool else "")
    return single, unique(valid_suppressions or [single]), []


def export_record(record: ExperimentRecord, split: str, holdout_field: str) -> dict[str, Any]:
    behavior = BEHAVIOR_MAP[record.expected_neg_behavior]
    pool = candidate_pool(record)
    single, multi, valid = negative_labels(record, pool)
    if single and normalize_text(single) not in {normalize_text(item) for item in pool}:
        pool = unique(pool + [single])

    if behavior == "select":
        invalid = without(pool, valid)
    elif behavior == "preserve":
        invalid = without(pool, record.gold_pos)
    else:
        invalid = unique(record.forbidden_neg + record.distractors)

    holdout_group = str(getattr(record, holdout_field))
    if split == "test" and not holdout_group:
        holdout_group = record.template_id

    return {
        "id": record.id,
        "behavior": behavior,
        "domain": record.domain,
        "template_id": record.template_id,
        "entity_family": record.family_id,
        "positive_prompt": record.prompt_pos,
        "negative_prompt": record.prompt_neg,
        "positive_gold": record.gold_pos,
        "negative_gold_single": single,
        "negative_gold_multi": multi,
        "candidate_pool": pool,
        "valid_negative_candidates": valid,
        "invalid_negative_candidates": invalid,
        "ambiguity_flag": behavior == "select" and len({normalize_text(item) for item in multi}) > 1,
        "split": split,
        "holdout_group": holdout_group,
        "metadata": {
            "source_schema": "e4",
            "source_id": record.id,
            "source_behavior": record.expected_neg_behavior,
            "source_scope_type": record.scope_type,
            "source_semantic_mode": record.semantic_mode,
            "source_metadata": record.metadata,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_unique_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    renamed: list[dict[str, str]] = []
    output: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)
        original = str(item["id"])
        counts[original] += 1
        candidate = original
        if candidate in seen:
            split = str(item.get("split", "split"))
            suffix_index = counts[original]
            candidate = f"{original}__{split}{suffix_index:03d}"
            while candidate in seen:
                suffix_index += 1
                candidate = f"{original}__{split}{suffix_index:03d}"
            metadata = dict(item.get("metadata") or {})
            metadata["release_id_renamed_from"] = original
            item["metadata"] = metadata
            item["id"] = candidate
            renamed.append({"from": original, "to": candidate})
        seen.add(candidate)
        output.append(item)

    return output, renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-input", action="append", type=split_arg, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--holdout-field",
        choices=["template_id", "family_id", "entity_id"],
        default="template_id",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    for split, path in args.split_input:
        records = load_records(path)
        for record in records:
            row = export_record(record, split, args.holdout_field)
            rows.append(row)
            split_counts[split] += 1
            behavior_counts[row["behavior"]] += 1

    rows, renamed_ids = ensure_unique_ids(rows)
    write_jsonl(Path(args.output), rows)

    report = {
        "output": args.output,
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "behavior_counts": dict(behavior_counts),
        "holdout_field": args.holdout_field,
        "renamed_duplicate_ids": len(renamed_ids),
        "renamed_duplicate_id_samples": renamed_ids[:50],
        "multi_answer_select_rows": sum(
            1 for row in rows
            if row["behavior"] == "select" and len(row["negative_gold_multi"]) > 1
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(rows)}")
    print("Split counts:", dict(split_counts))
    print("Behavior counts:", dict(behavior_counts))
    print(f"Saved: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
