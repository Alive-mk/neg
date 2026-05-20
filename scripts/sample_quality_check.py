"""Sample 200 records from training data for manual quality inspection.

Stratified by semantic_mode and scope_type, output a TSV/JSON for easy human review.

Usage:
    python scripts/sample_quality_check.py \
        --inputs data/processed/splits/balanced/train.jsonl \
                 data/processed/splits/balanced/dev.jsonl \
                 data/processed/splits/balanced/test.jsonl \
        --n 200 \
        --seed 77 \
        --output outputs/quality_check_200.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_all(paths: list[str]) -> list[dict]:
    records = []
    for p in paths:
        records.extend(json.loads(l) for l in open(p))
    return records


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Group by (semantic_mode, expected_neg_behavior, scope_type)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (
            r.get("semantic_mode", "?"),
            r.get("expected_neg_behavior", "?"),
            r.get("scope_type", "?"),
        )
        buckets[key].append(r)

    n_buckets = len(buckets)
    per_bucket = max(1, n // n_buckets)
    sampled: list[dict] = []
    for key, bucket in sorted(buckets.items()):
        k = min(per_bucket, len(bucket))
        sampled.extend(rng.sample(bucket, k))

    # Fill remainder from unsampled
    sampled_ids = {r["id"] for r in sampled}
    remaining = [r for r in records if r["id"] not in sampled_ids]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, n - len(sampled))])
    return sampled[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_all(args.inputs)
    print(f"Total records loaded: {len(records)}")

    sample = stratified_sample(records, args.n, args.seed)
    print(f"Sampled {len(sample)} records")

    from collections import Counter
    print("Distribution:")
    for k, v in Counter(r.get("semantic_mode","?") for r in sample).items():
        print(f"  {k}: {v}")
    for k, v in Counter(r.get("expected_neg_behavior","?") for r in sample).items():
        print(f"  {k}: {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write TSV for easy spreadsheet review
    fields = [
        "id", "semantic_mode", "scope_type", "expected_neg_behavior",
        "prompt_pos", "prompt_neg", "gold_pos_0", "gold_neg_0",
        "candidate_pool_neg_preview",
        "quality_ok",  # blank column for human annotation (Y/N)
        "quality_notes",  # blank column for notes
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in sample:
            row = {
                "id": r.get("id", ""),
                "semantic_mode": r.get("semantic_mode", ""),
                "scope_type": r.get("scope_type", ""),
                "expected_neg_behavior": r.get("expected_neg_behavior", ""),
                "prompt_pos": r.get("prompt_pos", "")[:200],
                "prompt_neg": r.get("prompt_neg", "")[:200],
                "gold_pos_0": (r.get("gold_pos") or [""])[0][:100],
                "gold_neg_0": (r.get("gold_neg") or [""])[0][:100],
                "candidate_pool_neg_preview": "|".join((r.get("candidate_pool_neg") or [])[:3])[:150],
                "quality_ok": "",
                "quality_notes": "",
            }
            w.writerow(row)

    # Also write JSON for programmatic access
    json_out = args.output.replace(".tsv", ".json")
    json.dump(
        [{"idx": i + 1, **r} for i, r in enumerate(sample)],
        open(json_out, "w"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"Saved TSV: {args.output}")
    print(f"Saved JSON: {json_out}")


if __name__ == "__main__":
    main()
