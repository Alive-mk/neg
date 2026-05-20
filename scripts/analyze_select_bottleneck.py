"""
Analyze select_gold_neg failures for two evaluated model outputs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def load_output(path: Path, model: str | None) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model is None:
        if len(payload) != 1:
            raise ValueError(f"{path} has multiple models; pass the model name")
        model = next(iter(payload))
    return {record["id"]: record for record in payload[model].get("per_record", [])}


def pct(num: int, den: int) -> str:
    return "nan" if den == 0 else f"{100.0 * num / den:.1f}"


def summarize_subset(name: str, records: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> None:
    select_ids = [
        rid
        for rid, record in records.items()
        if record["expected_neg_behavior"] == "select_gold_neg" and rid in outputs
    ]
    ok = [rid for rid in select_ids if outputs[rid].get("neg_rank_correct")]
    bad = [rid for rid in select_ids if not outputs[rid].get("neg_rank_correct")]
    print(f"\n[{name}] select n={len(select_ids)} ok={len(ok)} acc={pct(len(ok), len(select_ids))}")

    for field in ["semantic_mode", "scope_type", "domain", "template_id"]:
        counts = Counter(records[rid].get(field, "") for rid in bad)
        print(f"  miss by {field}:")
        for key, count in counts.most_common(12):
            print(f"    {key}: {count}")
    wrong_type = Counter()
    examples = []
    for rid in bad:
        record = records[rid]
        neg_best = outputs[rid].get("neg_best")
        if neg_best in record.get("forbidden_neg", []):
            label = "forbidden_neg"
        elif neg_best in record.get("gold_pos", []):
            label = "gold_pos"
        elif neg_best in record.get("candidate_pool_neg", []):
            label = "candidate_pool_neg"
        elif neg_best in record.get("distractors", []):
            label = "distractor"
        else:
            label = "other"
        wrong_type[label] += 1
        if len(examples) < 10:
            examples.append((rid, label, record, neg_best))
    print(f"  wrong best type: {dict(wrong_type.most_common())}")
    for rid, label, record, neg_best in examples:
        print(
            f"    miss {rid}: type={label} mode={record.get('semantic_mode')} "
            f"domain={record.get('domain')} prompt={record.get('prompt_neg')} "
            f"gold_neg={record.get('gold_neg')} neg_best={neg_best}"
        )


def summarize_deltas(
    records: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    reference_ids: set[str] | None,
) -> None:
    select_ids = [
        rid
        for rid, record in records.items()
        if record["expected_neg_behavior"] == "select_gold_neg" and rid in base and rid in candidate
    ]
    if reference_ids is not None:
        select_ids = [rid for rid in select_ids if rid in reference_ids]
    gains = [
        rid
        for rid in select_ids
        if not base[rid].get("neg_rank_correct") and candidate[rid].get("neg_rank_correct")
    ]
    losses = [
        rid
        for rid in select_ids
        if base[rid].get("neg_rank_correct") and not candidate[rid].get("neg_rank_correct")
    ]
    print(f"\n[deltas] select n={len(select_ids)} gains={len(gains)} losses={len(losses)} net={len(gains)-len(losses)}")
    for label, ids in [("gains", gains), ("losses", losses)]:
        print(f"  {label}:")
        for field in ["semantic_mode", "scope_type", "domain", "template_id"]:
            counts = Counter(records[rid].get(field, "") for rid in ids)
            print(f"    by {field}: {dict(counts.most_common(8))}")
        for rid in ids[:8]:
            record = records[rid]
            print(
                f"    example {rid}: template={record.get('template_id')} "
                f"mode={record.get('semantic_mode')} prompt={record.get('prompt_neg')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--candidate-model")
    parser.add_argument("--reference-output")
    parser.add_argument("--reference-model")
    args = parser.parse_args()

    records = load_jsonl(Path(args.clean_input))
    base = load_output(Path(args.base_output), args.base_model)
    candidate = load_output(Path(args.candidate_output), args.candidate_model)
    reference_ids = None
    if args.reference_output:
        reference_ids = set(load_output(Path(args.reference_output), args.reference_model))

    summarize_subset("base", records, base)
    summarize_subset("candidate", records, candidate)
    summarize_deltas(records, base, candidate, reference_ids)
    if reference_ids is not None:
        missing_records = {rid: rec for rid, rec in records.items() if rid not in reference_ids}
        print(f"\n[reference missing] records={len(missing_records)}")
        summarize_subset("base missing-reference", missing_records, base)
        summarize_subset("candidate missing-reference", missing_records, candidate)


if __name__ == "__main__":
    main()
