"""Extract qualitative case studies from per-record eval results.

Finds examples where MGNM and Vanilla SFT disagree, useful for paper qualitative analysis.

Usage:
    python scripts/extract_case_studies.py \
        --mgnm    outputs/eval_lsup2_largetest.json \
        --vanilla outputs/eval_vanilla_3ep_largetest.json \
        --data    data/processed/splits/large_test/test.jsonl \
        --output  outputs/case_studies.json \
        --n       10
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def load_records_by_id(eval_path: str) -> dict[str, dict]:
    d = json.load(open(eval_path))
    sub = d[list(d.keys())[-1]]
    return {r["id"]: r for r in sub.get("per_record", [])}


def load_data_by_id(data_path: str) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(data_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgnm", required=True)
    parser.add_argument("--vanilla", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mgnm = load_records_by_id(args.mgnm)
    vanilla = load_records_by_id(args.vanilla)
    data = load_data_by_id(args.data)
    common = set(mgnm) & set(vanilla) & set(data)

    # Case type 1: suppress_target — MGNM correct, Vanilla correct (both good)
    # Case type 2: suppress_target — MGNM wrong, Vanilla correct (MGNM weakness)
    # Case type 3: preserve_positive — MGNM correct, Vanilla wrong (MGNM strength)
    # Case type 4: preserve_positive — both wrong (hard cases)

    categories: dict[str, list] = {
        "sup_mgnm_right_van_wrong": [],
        "sup_mgnm_wrong_van_right": [],
        "pre_mgnm_right_van_wrong": [],
        "pre_mgnm_wrong_van_right": [],
    }

    for rid in sorted(common):
        m = mgnm[rid]
        v = vanilla[rid]
        beh = m.get("expected_neg_behavior", "")

        if beh == "suppress_target":
            m_ok = m.get("neg_suppressed", False)
            v_ok = v.get("neg_suppressed", False)
            if m_ok and not v_ok:
                categories["sup_mgnm_right_van_wrong"].append(rid)
            elif not m_ok and v_ok:
                categories["sup_mgnm_wrong_van_right"].append(rid)

        elif beh == "preserve_positive":
            m_ok = m.get("preserve_positive", False)
            v_ok = v.get("preserve_positive", False)
            if m_ok and not v_ok:
                categories["pre_mgnm_right_van_wrong"].append(rid)
            elif not m_ok and v_ok:
                categories["pre_mgnm_wrong_van_right"].append(rid)

    rng = random.Random(args.seed)
    results = {}
    per_cat = max(1, args.n // len(categories))

    for cat, ids in categories.items():
        sample = rng.sample(ids, min(per_cat, len(ids)))
        cases = []
        for rid in sample:
            rec = data[rid]
            cases.append({
                "id": rid,
                "category": cat,
                "semantic_mode": rec.get("semantic_mode"),
                "scope_type": rec.get("scope_type"),
                "expected_neg_behavior": rec.get("expected_neg_behavior"),
                "prompt_pos": rec.get("prompt_pos", ""),
                "prompt_neg": rec.get("prompt_neg", ""),
                "gold_pos": rec.get("gold_pos", []),
                "gold_neg": rec.get("gold_neg", []),
                "mgnm_neg_suppressed": mgnm[rid].get("neg_suppressed"),
                "mgnm_preserve_pos": mgnm[rid].get("preserve_positive"),
                "mgnm_neg_best": mgnm[rid].get("neg_best", ""),
                "vanilla_neg_suppressed": vanilla[rid].get("neg_suppressed"),
                "vanilla_preserve_pos": vanilla[rid].get("preserve_positive"),
                "vanilla_neg_best": vanilla[rid].get("neg_best", ""),
            })
        results[cat] = {"n": len(ids), "sample": cases}
        print(f"{cat}: {len(ids)} cases, sampled {len(sample)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
