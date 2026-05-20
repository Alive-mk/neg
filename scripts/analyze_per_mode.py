"""Per-semantic-mode breakdown of evaluation metrics.

Produces a table showing how each method performs on each semantic mode
(suppression_only, contrastive_resolution, exclusive_choice).
Useful for understanding where MGNM vs Vanilla SFT differ.

Usage:
    python scripts/analyze_per_mode.py \
        --inputs outputs/eval_bal_full_3ep_largetest.json \
                 outputs/eval_vanilla_3ep_largetest.json \
        --labels "MGNM" "Vanilla SFT" \
        --output outputs/per_mode_breakdown.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    ("PosAcc",       "pos_correct",      None),
    ("NegSuppRate",  "neg_suppressed",   "suppress_target"),
    ("FlipAcc",      "flip_correct",     None),          # filtered by flip_required
    ("ScopeCtrl",    "preserve_positive","preserve_positive"),
    ("OverNeg",      "over_negation",    "preserve_positive"),
]


def load_per_record(path: str) -> list[dict]:
    d = json.load(open(path))
    sub = d[list(d.keys())[-1]]
    return sub.get("per_record", [])


def compute_metric_by_mode(records: list[dict], field: str, behavior_filter: str | None) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if behavior_filter == "suppress_target" and r.get("expected_neg_behavior") != "suppress_target":
            continue
        if behavior_filter == "preserve_positive" and r.get("expected_neg_behavior") != "preserve_positive":
            continue
        if field == "flip_correct" and not r.get("flip_required", False):
            continue
        if field not in r:
            continue
        mode = r.get("semantic_mode", "unknown")
        buckets[mode].append(float(r[field]))
    return {k: mean(v) for k, v in buckets.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    all_records = [(label, load_per_record(path)) for label, path in zip(args.labels, args.inputs)]

    # Find all modes
    modes = sorted({r.get("semantic_mode","?") for _, recs in all_records for r in recs})

    results = {}
    for label, metric_field, behavior in METRICS:
        print(f"\n=== {label} ===")
        header = f"{'Method':<20}" + "".join(f"  {m:<24}" for m in modes) + "  Overall"
        print(header)
        print("-" * len(header))
        metric_results = {}
        for method, records in all_records:
            by_mode = compute_metric_by_mode(records, metric_field, behavior)
            overall = mean(list(by_mode.values())) if by_mode else 0.0
            row = f"{method:<20}"
            for m in modes:
                v = by_mode.get(m, float("nan"))
                row += f"  {v*100:>6.1f}%{' ':17}"
            row += f"  {overall*100:>6.1f}%"
            print(row)
            metric_results[method] = {**{m: by_mode.get(m, None) for m in modes}, "overall": overall}
        results[label] = metric_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
