from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, write_json
from neg_blindness.schema import ExperimentRecord, normalize_text
from neg_blindness.validators import validate_record


BEHAVIOR_ALIASES = {
    "suppress": "suppress_target",
    "suppression": "suppress_target",
    "suppress_target": "suppress_target",
    "select": "select_gold_neg",
    "select_gold_neg": "select_gold_neg",
    "preserve": "preserve_positive",
    "preserve_positive": "preserve_positive",
}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value).strip()]
    return []


def first_value(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def without(items: list[str], forbidden: list[str]) -> list[str]:
    forbidden_norm = {normalize_text(item) for item in forbidden}
    return [item for item in unique(items) if normalize_text(item) not in forbidden_norm]


def behavior_from_row(row: dict[str, Any], default_behavior: str) -> str:
    value = first_value(row, ["expected_neg_behavior", "behavior"], default_behavior)
    behavior = BEHAVIOR_ALIASES.get(str(value).strip(), str(value).strip())
    if behavior not in {"suppress_target", "select_gold_neg", "preserve_positive"}:
        raise ValueError(f"Unsupported behavior: {value!r}")
    return behavior


def convert_row(
    row: dict[str, Any],
    index: int,
    benchmark_name: str,
    default_behavior: str,
    default_domain: str,
    id_prefix: str,
) -> ExperimentRecord:
    behavior = behavior_from_row(row, default_behavior)
    record_id = str(first_value(row, ["id", "uid"], f"{id_prefix}_{index:05d}"))
    if id_prefix and not record_id.startswith(f"{id_prefix}_"):
        record_id = f"{id_prefix}_{record_id}"

    prompt_pos = str(first_value(
        row,
        ["prompt_pos", "positive_prompt", "positive", "prompt_positive"],
        "",
    ))
    prompt_neg = str(first_value(
        row,
        ["prompt_neg", "negative_prompt", "negated_prompt", "prompt"],
        "",
    ))

    gold_pos = as_list(first_value(
        row,
        ["gold_pos", "positive_gold", "forbidden_target", "forbidden_targets", "answer"],
        [],
    ))
    forbidden_neg = as_list(first_value(
        row,
        ["forbidden_neg", "forbidden_target", "forbidden_targets", "positive_gold", "gold_pos"],
        gold_pos,
    ))
    gold_neg = as_list(first_value(
        row,
        ["gold_neg", "negative_gold", "negative_gold_single"],
        [],
    ))
    valid_negatives = as_list(first_value(
        row,
        ["valid_negatives", "valid_negative_candidates", "negative_gold_multi"],
        [],
    ))
    candidate_pool = as_list(first_value(
        row,
        ["candidate_pool_neg", "candidate_pool", "candidates", "options"],
        [],
    ))
    distractors = as_list(first_value(row, ["distractors", "hard_distractors"], []))

    if behavior == "suppress_target":
        candidate_pool_neg = without(candidate_pool + valid_negatives, forbidden_neg or gold_pos)
        gold_neg = []
    elif behavior == "select_gold_neg":
        gold_neg = unique(gold_neg or valid_negatives[:1])
        valid_negatives = unique(without(valid_negatives, gold_neg))
        candidate_pool_neg = without(candidate_pool, forbidden_neg or gold_pos)
    else:
        candidate_pool_neg = without(candidate_pool, gold_pos)
        gold_neg = []
        valid_negatives = []

    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_benchmark": benchmark_name,
            "source_row": {
                key: value
                for key, value in row.items()
                if key
                in {
                    "benchmark",
                    "category",
                    "relation",
                    "subject",
                    "object",
                    "source",
                    "split",
                }
            },
        }
    )

    category = str(first_value(row, ["category", "relation"], benchmark_name))
    subject = str(first_value(row, ["subject", "entity", "topic"], record_id))
    relation = str(first_value(row, ["relation", "predicate"], category))

    return ExperimentRecord(
        id=record_id,
        neg_type=str(first_value(row, ["neg_type"], "sentential")),
        scope_type=str(first_value(row, ["scope_type"], "in_scope")),
        semantic_mode=str(first_value(row, ["semantic_mode"], "suppression_only"))
        if behavior == "suppress_target"
        else str(first_value(row, ["semantic_mode"], "exclusive_choice")),
        expected_neg_behavior=behavior,
        domain=str(first_value(row, ["domain"], default_domain)),
        prompt_pos=prompt_pos,
        prompt_neg=prompt_neg,
        gold_pos=gold_pos,
        gold_neg=gold_neg,
        forbidden_neg=forbidden_neg or gold_pos,
        candidate_pool_neg=candidate_pool_neg,
        valid_negatives=valid_negatives,
        distractors=distractors,
        template_id=str(first_value(row, ["template_id"], f"{benchmark_name}_{category}")),
        family_id=str(first_value(row, ["family_id", "entity_family"], f"{benchmark_name}_{relation}")),
        entity_id=str(first_value(row, ["entity_id"], f"{benchmark_name}_{subject}")),
        metadata=metadata,
    ).with_defaults()


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--benchmark-name", default="public_negation")
    parser.add_argument(
        "--default-behavior",
        choices=["suppress_target", "select_gold_neg", "preserve_positive"],
        default="suppress_target",
    )
    parser.add_argument("--default-domain", default="factual")
    parser.add_argument("--id-prefix", default="pubneg")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--require-distractors",
        action="store_true",
        help="Enforce E4's non-empty distractor rule for external benchmark rows.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    records: list[ExperimentRecord] = []
    invalid_samples: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        try:
            record = convert_row(
                row=row,
                index=index,
                benchmark_name=args.benchmark_name,
                default_behavior=args.default_behavior,
                default_domain=args.default_domain,
                id_prefix=args.id_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            error_counts[f"convert_error: {exc}"] += 1
            invalid_samples.append({"index": index, "errors": [str(exc)]})
            continue
        validation = validate_record(record)
        errors = list(validation.errors)
        if not args.require_distractors:
            errors = [
                error for error in errors
                if error != "distractors must be non-empty"
            ]
        if not errors:
            records.append(record)
            continue
        for error in errors:
            error_counts[error] += 1
        invalid_samples.append(
            {
                "index": index,
                "id": record.id,
                "errors": errors,
            }
        )

    dump_records(args.output, records)
    report = {
        "input": args.input,
        "output": args.output,
        "benchmark_name": args.benchmark_name,
        "input_rows": len(rows),
        "valid_records": len(records),
        "invalid_records": len(invalid_samples),
        "behavior_counts": dict(Counter(record.expected_neg_behavior for record in records)),
        "domain_counts": dict(Counter(record.domain for record in records)),
        "template_count": len({record.template_id for record in records}),
        "error_counts": dict(error_counts),
        "invalid_samples": invalid_samples[:50],
    }
    write_json(args.report, report)

    print(f"Input rows: {len(rows)}")
    print(f"Valid records: {len(records)}")
    print(f"Invalid records: {len(invalid_samples)}")
    print("Behavior counts:", report["behavior_counts"])
    print(f"Saved: {args.output}")
    print(f"Saved report: {args.report}")
    if invalid_samples:
        raise SystemExit("Some rows could not be converted; inspect the report.")


if __name__ == "__main__":
    main()
