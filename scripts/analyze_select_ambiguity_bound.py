"""
Estimate how much select_gold_neg accuracy is capped by single-answer labels.

This does not replace audited metrics. It reports an upper bound where a model
choice from candidate_pool_neg is counted as potentially valid for select items.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def load_output(path: Path, model_name: str | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_name is None:
        if len(payload) != 1:
            raise ValueError(f"{path} has multiple models; pass --model")
        model_name = next(iter(payload))
    return payload[model_name].get("per_record", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-tsv")
    args = parser.parse_args()

    records = load_records(Path(args.clean_input))
    outputs = load_output(Path(args.eval_output), args.model)
    total = hard = soft_all = soft_exclusive = 0
    candidate_pool_misses: list[dict[str, Any]] = []
    wrong_type = Counter()
    bucket = Counter()

    for item in outputs:
        record = records.get(item["id"])
        if not record or record["expected_neg_behavior"] != "select_gold_neg":
            continue
        total += 1
        is_hard = bool(item.get("neg_rank_correct"))
        hard += int(is_hard)
        neg_best = item.get("neg_best")
        in_candidate_pool = neg_best in record.get("candidate_pool_neg", [])
        soft_all += int(is_hard or in_candidate_pool)
        soft_exclusive += int(
            is_hard
            or (record.get("semantic_mode") == "exclusive_choice" and in_candidate_pool)
        )
        if not is_hard:
            if in_candidate_pool:
                label = "candidate_pool_neg"
                candidate_pool_misses.append(
                    {
                        "id": item["id"],
                        "semantic_mode": record.get("semantic_mode"),
                        "domain": record.get("domain"),
                        "template_id": record.get("template_id"),
                        "prompt_neg": record.get("prompt_neg"),
                        "gold_neg": record.get("gold_neg"),
                        "candidate_pool_neg": record.get("candidate_pool_neg", []),
                        "neg_best": neg_best,
                    }
                )
                bucket[(record.get("semantic_mode", ""), record.get("domain", ""))] += 1
            elif neg_best in record.get("forbidden_neg", []):
                label = "forbidden_neg"
            elif neg_best in record.get("gold_pos", []):
                label = "gold_pos"
            elif neg_best in record.get("distractors", []):
                label = "distractor"
            else:
                label = "other"
            wrong_type[label] += 1

    report = {
        "model": args.model,
        "select_n": total,
        "hard_neg_rank": {
            "correct": hard,
            "accuracy": hard / total if total else 0.0,
        },
        "candidate_pool_upper_bound": {
            "correct": soft_all,
            "accuracy": soft_all / total if total else 0.0,
            "note": "Counts any candidate_pool_neg choice as potentially valid; diagnostic upper bound only.",
        },
        "exclusive_choice_candidate_pool_upper_bound": {
            "correct": soft_exclusive,
            "accuracy": soft_exclusive / total if total else 0.0,
            "note": "Counts candidate_pool_neg choices only for exclusive_choice items.",
        },
        "wrong_best_type": dict(wrong_type.most_common()),
        "candidate_pool_miss_buckets": {
            f"{mode}|{domain}": count
            for (mode, domain), count in bucket.most_common()
        },
        "candidate_pool_miss_examples": candidate_pool_misses,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.audit_tsv:
        audit_path = Path(args.audit_tsv)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "semantic_mode",
                    "domain",
                    "template_id",
                    "prompt_neg",
                    "gold_neg",
                    "neg_best",
                    "candidate_pool_neg",
                    "audit_decision",
                    "audit_note",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for row in candidate_pool_misses:
                writer.writerow(
                    {
                        **row,
                        "gold_neg": " || ".join(row.get("gold_neg") or []),
                        "candidate_pool_neg": " || ".join(row.get("candidate_pool_neg") or []),
                        "audit_decision": "",
                        "audit_note": "",
                    }
                )

    print(
        f"select n={total} "
        f"hard={100 * report['hard_neg_rank']['accuracy']:.1f} "
        f"candidate_pool_upper={100 * report['candidate_pool_upper_bound']['accuracy']:.1f} "
        f"exclusive_choice_upper={100 * report['exclusive_choice_candidate_pool_upper_bound']['accuracy']:.1f}"
    )
    print(f"wrong_best_type={report['wrong_best_type']}")
    print(f"candidate_pool_miss_buckets={report['candidate_pool_miss_buckets']}")
    print(f"Saved: {out_path}")
    if args.audit_tsv:
        print(f"Saved audit TSV: {args.audit_tsv}")


if __name__ == "__main__":
    main()
