"""Materialize reviewed SELECT multi-answer labels into an E4 JSONL file."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_reviewed_audit(path: Path) -> dict[str, set[str]]:
    valid: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("audit_decision") == "valid_multi_answer":
                valid.setdefault(row["id"], set()).add(row["neg_best"])
    return valid


def unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit-tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    audited_valid = load_reviewed_audit(Path(args.audit_tsv))
    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line
    ]

    updated: list[dict[str, Any]] = []
    rescue_count = 0
    ambiguous_count = 0
    behavior_counts = Counter(record.get("expected_neg_behavior") for record in records)
    for record in records:
        item = dict(record)
        existing_valid = list(item.get("valid_negatives") or [])
        valid_negatives = unique(existing_valid + sorted(audited_valid.get(item["id"], set())))
        if valid_negatives:
            item["valid_negatives"] = valid_negatives
            metadata = dict(item.get("metadata") or {})
            metadata["select_multianswer_audit"] = {
                "source": args.audit_tsv,
                "valid_negative_count": len(valid_negatives),
            }
            item["metadata"] = metadata
        rescue_count += len(set(valid_negatives) - set(existing_valid))
        if item.get("expected_neg_behavior") == "select_gold_neg":
            valid_set = set(item.get("gold_neg", []) + valid_negatives)
            if len(valid_set) > len(set(item.get("gold_neg", []))):
                ambiguous_count += 1
        updated.append(item)

    write_jsonl(Path(args.output), updated)
    select_total = behavior_counts.get("select_gold_neg", 0)
    report = {
        "input": args.input,
        "audit_tsv": args.audit_tsv,
        "output": args.output,
        "n_records": len(records),
        "behavior_counts": dict(behavior_counts),
        "records_with_added_valid_negatives": len(audited_valid),
        "added_valid_negative_labels": rescue_count,
        "select_n": select_total,
        "select_ambiguous": ambiguous_count,
        "select_ambiguous_rate": ambiguous_count / select_total if select_total else 0.0,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Records: {len(records)}")
    print(f"SELECT ambiguous: {ambiguous_count}/{select_total} = {report['select_ambiguous_rate'] * 100:.1f}%")
    print(f"Added valid negatives: {rescue_count}")
    print(f"Saved: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
