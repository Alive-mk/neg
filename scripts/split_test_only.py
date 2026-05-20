"""
Create a test-only split from a validated pool.

Loads an optional reference train set to compute entity/family holdout metrics.
Deduplicates prompts against the reference set and the existing test splits.
Outputs:
  <output-dir>/test.jsonl          — the new large test set
  <report>                          — split report with holdout stats
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, load_records, write_json
from neg_blindness.schema import ExperimentRecord, normalize_text


def _prompt_pair(r: ExperimentRecord) -> tuple[str, str]:
    return normalize_text(r.prompt_pos), normalize_text(r.prompt_neg)


def _distribution(records: list[ExperimentRecord], attr: str) -> dict[str, int]:
    return dict(Counter(getattr(r, attr) for r in records))


def _holdout_report(
    ref: list[ExperimentRecord],
    test: list[ExperimentRecord],
    attr: str,
) -> dict:
    ref_values = {getattr(r, attr) for r in ref}
    unseen = [r for r in test if getattr(r, attr) not in ref_values]
    return {
        "unseen_records": len(unseen),
        "total_records": len(test),
        "unseen_ratio": round(len(unseen) / len(test), 4) if test else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="validated pool (.jsonl)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--ref-train",
        nargs="+",
        default=[],
        help="Reference train/dev .jsonl files to compute holdout metrics and exclude prompt overlaps",
    )
    parser.add_argument(
        "--existing-tests",
        nargs="+",
        default=[],
        help="Existing test .jsonl files — prompts already in these will be excluded",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"[split_test_only] loaded {len(records)} records from {args.input}")

    ref_records: list[ExperimentRecord] = []
    for path in args.ref_train:
        ref_records.extend(load_records(path))
    print(f"[split_test_only] reference train/dev records: {len(ref_records)}")

    excluded_pairs: set[tuple[str, str]] = set()
    for path in args.ref_train + args.existing_tests:
        for r in load_records(path):
            excluded_pairs.add(_prompt_pair(r))

    deduped = [r for r in records if _prompt_pair(r) not in excluded_pairs]
    print(f"[split_test_only] after prompt dedup: {len(deduped)} records")

    if args.max_samples and len(deduped) > args.max_samples:
        deduped = deduped[: args.max_samples]
        print(f"[split_test_only] capped to {args.max_samples} records")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_records(output_dir / "test.jsonl", deduped)
    print(f"[split_test_only] wrote {len(deduped)} records -> {output_dir}/test.jsonl")

    report = {
        "input_records": len(records),
        "excluded_by_prompt_overlap": len(records) - len(deduped),
        "final_test_records": len(deduped),
        "distributions": {
            "semantic_mode": _distribution(deduped, "semantic_mode"),
            "neg_type": _distribution(deduped, "neg_type"),
            "scope_type": _distribution(deduped, "scope_type"),
            "domain": _distribution(deduped, "domain"),
            "expected_neg_behavior": _distribution(deduped, "expected_neg_behavior"),
        },
    }
    if ref_records:
        report["holdout_vs_train"] = {
            "family_id": _holdout_report(ref_records, deduped, "family_id"),
            "entity_id": _holdout_report(ref_records, deduped, "entity_id"),
            "template_id": _holdout_report(ref_records, deduped, "template_id"),
        }

    write_json(args.report, report)
    print(f"[split_test_only] report written -> {args.report}")

    if ref_records:
        fam = report["holdout_vs_train"]["family_id"]
        ent = report["holdout_vs_train"]["entity_id"]
        print(
            f"[split_test_only] holdout: family={fam['unseen_ratio']:.1%}  entity={ent['unseen_ratio']:.1%}"
        )


if __name__ == "__main__":
    main()
