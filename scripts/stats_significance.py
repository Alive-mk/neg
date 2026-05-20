"""Statistical significance tests comparing MGNM vs Vanilla SFT on large_test.

Tests applied:
- McNemar's test (exact) for each binary metric
- Bootstrap 95% CI difference for each metric
- Overall summary table

Usage:
    python scripts/stats_significance.py \
        --mgnm    outputs/eval_bal_full_3ep_largetest.json \
        --vanilla outputs/eval_vanilla_3ep_largetest.json \
        --output  outputs/stats_significance.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


METRICS = [
    ("PosAcc",          "pos_correct",      None),
    ("NegSuppRate",     "neg_suppressed",   lambda r: r.get("expected_neg_behavior") == "suppress_target"),
    ("FlipAcc",         "flip_correct",     lambda r: r.get("flip_required", False)),
    ("ScopeControlAcc", "preserve_positive",lambda r: r.get("expected_neg_behavior") == "preserve_positive"),
    ("OverNegationRate","over_negation",    lambda r: r.get("expected_neg_behavior") == "preserve_positive"),
    ("DoubleNegAcc",    "preserve_positive",lambda r: r.get("scope_type") == "double_negation"),
]


def load_per_record(path: str, model_key: str | None = None) -> list[dict]:
    d = json.load(open(path))
    if model_key:
        data = d[model_key]
    else:
        data = d[list(d.keys())[-1]]
    if "aggregate" in data:
        return data["aggregate"]["per_record"]
    return data["per_record"]


def load_keep_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {json.loads(line)["id"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided McNemar exact binomial test. Returns p-value."""
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    target = min(b, c)
    for k in range(target + 1):
        binom = math.comb(n, k) * (0.5 ** n)
        p += binom
    p *= 2
    p = min(p, 1.0)
    return p


def bootstrap_diff(vals_a: list[float], vals_b: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    rng = random.Random(seed)
    n = len(vals_a)
    diffs = []
    for _ in range(n_boot):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        diff = sum(vals_a[i] - vals_b[i] for i in indices) / n
        diffs.append(diff)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    obs = sum(a - b for a, b in zip(vals_a, vals_b)) / n
    return obs, lo, hi


def get_metric_pairs(
    mgnm_records: list[dict],
    vanilla_records: list[dict],
    field: str,
    filter_fn=None,
) -> tuple[list[float], list[float]]:
    mgnm_by_id = {r["id"]: r for r in mgnm_records}
    vanilla_by_id = {r["id"]: r for r in vanilla_records}
    common_ids = sorted(set(mgnm_by_id) & set(vanilla_by_id))

    a_vals, b_vals = [], []
    for rid in common_ids:
        r_m = mgnm_by_id[rid]
        r_v = vanilla_by_id[rid]
        if filter_fn and not filter_fn(r_m):
            continue
        if field not in r_m or field not in r_v:
            continue
        a_vals.append(float(r_m[field]))
        b_vals.append(float(r_v[field]))
    return a_vals, b_vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgnm", required=True)
    parser.add_argument("--vanilla", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mgnm-key")
    parser.add_argument("--vanilla-key")
    parser.add_argument("--keep-ids")
    args = parser.parse_args()

    mgnm_records = load_per_record(args.mgnm, args.mgnm_key)
    vanilla_records = load_per_record(args.vanilla, args.vanilla_key)
    keep_ids = load_keep_ids(args.keep_ids)
    if keep_ids is not None:
        mgnm_records = [record for record in mgnm_records if record["id"] in keep_ids]
        vanilla_records = [record for record in vanilla_records if record["id"] in keep_ids]

    print(f"MGNM records: {len(mgnm_records)}, Vanilla records: {len(vanilla_records)}")

    results = {}
    print(f"\n{'Metric':<22} {'MGNM':>8} {'Vanilla':>10} {'Δ':>8} {'95% CI (Δ)':>22} {'p (McNemar)':>14} {'sig':>5}")
    print("-" * 95)

    for label, field, filter_fn in METRICS:
        a_vals, b_vals = get_metric_pairs(mgnm_records, vanilla_records, field, filter_fn)
        if not a_vals:
            print(f"{label:<22} {'N/A':>8}")
            continue

        n = len(a_vals)
        mgnm_mean = sum(a_vals) / n
        vanilla_mean = sum(b_vals) / n
        obs_diff, lo, hi = bootstrap_diff(a_vals, b_vals)

        # McNemar contingency: b = vanilla yes, mgnm no; c = vanilla no, mgnm yes
        b = sum(1 for a, bv in zip(a_vals, b_vals) if bv > 0.5 and a <= 0.5)
        c = sum(1 for a, bv in zip(a_vals, b_vals) if bv <= 0.5 and a > 0.5)
        p = mcnemar_exact(b, c)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        print(
            f"{label:<22} {mgnm_mean:>8.3f} {vanilla_mean:>10.3f} "
            f"{obs_diff:>+8.3f} [{lo:>+.3f}, {hi:>+.3f}] "
            f"{p:>14.4f} {sig:>5}  (n={n})"
        )
        results[label] = {
            "n": n,
            "mgnm_mean": mgnm_mean,
            "vanilla_mean": vanilla_mean,
            "obs_diff": obs_diff,
            "bootstrap_95ci": [lo, hi],
            "mcnemar_p": p,
            "significant_05": p < 0.05,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
