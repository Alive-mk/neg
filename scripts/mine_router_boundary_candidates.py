from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from clean_e4_preserve_labels import removal_reason


TARGET_BUCKETS = {
    ("preserve_positive", "double_negation", "exclusive_choice"): "preserve_double_exclusive",
    ("preserve_positive", "double_negation", "contrastive_resolution"): "preserve_double_contrastive",
    ("preserve_positive", "out_of_scope", "contrastive_resolution"): "preserve_outscope_contrastive",
    ("select_gold_neg", "in_scope", "exclusive_choice"): "select_inscope_exclusive",
    ("select_gold_neg", "in_scope", "contrastive_resolution"): "select_inscope_contrastive",
}

AUDIT_BUCKETS = {
    ("preserve_positive", "out_of_scope", "exclusive_choice"): "audit_preserve_query_like",
    ("suppress_target", "in_scope", "suppression_only"): "audit_suppress_query_like",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def content_key(record: dict) -> str:
    return json.dumps(
        {
            key: record.get(key)
            for key in [
                "expected_neg_behavior",
                "scope_type",
                "semantic_mode",
                "prompt_pos",
                "prompt_neg",
                "gold_pos",
                "gold_neg",
                "forbidden_neg",
                "candidate_pool_neg",
                "distractors",
                "family_id",
                "entity_id",
                "template_id",
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def dedupe_records(records: list[dict]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for record in records:
        deduped.setdefault(content_key(record), record)
    return list(deduped.values())


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "bucket",
        "expected_neg_behavior",
        "scope_type",
        "semantic_mode",
        "family_id",
        "entity_id",
        "template_id",
        "priority",
        "source",
        "cleaning_reason",
        "prompt_neg",
        "gold_pos_0",
        "gold_neg_0",
        "candidate_pool_neg_preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "bucket": record["bucket"],
                    "expected_neg_behavior": record["expected_neg_behavior"],
                    "scope_type": record["scope_type"],
                    "semantic_mode": record["semantic_mode"],
                    "family_id": record["family_id"],
                    "entity_id": record["entity_id"],
                    "template_id": record["template_id"],
                    "priority": record["priority"],
                    "source": record["source"],
                    "cleaning_reason": record.get("cleaning_reason", ""),
                    "prompt_neg": record["prompt_neg"],
                    "gold_pos_0": (record.get("gold_pos") or [""])[0],
                    "gold_neg_0": (record.get("gold_neg") or [""])[0],
                    "candidate_pool_neg_preview": " | ".join((record.get("candidate_pool_neg") or [])[:3]),
                }
            )


def prioritize(record: dict, used_families: set[str]) -> tuple[int, int, str, str]:
    family_seen = int(record["family_id"] in used_families)
    bucket_rank = {
        "preserve_double_exclusive": 0,
        "preserve_double_contrastive": 1,
        "preserve_outscope_contrastive": 2,
        "select_inscope_exclusive": 3,
        "select_inscope_contrastive": 4,
        "audit_preserve_query_like": 5,
        "audit_suppress_query_like": 6,
    }.get(record["bucket"], 99)
    text = record.get("prompt_neg", "").lower()
    lexical_bonus = int("which" in text or "what" in text or "who" in text)
    return (bucket_rank, family_seen, -lexical_bonus, record["prompt_neg"])


def select_diverse(records: list[dict], max_per_bucket: int, max_per_family: int) -> list[dict]:
    selected: list[dict] = []
    by_bucket: dict[str, int] = Counter()
    by_family: dict[str, int] = Counter()
    for record in records:
        if by_bucket[record["bucket"]] >= max_per_bucket:
            continue
        if by_family[record["family_id"]] >= max_per_family:
            continue
        selected.append(record)
        by_bucket[record["bucket"]] += 1
        by_family[record["family_id"]] += 1
    return selected


def add_metadata(records: list[dict], bucket_map: dict[tuple[str, str, str], str], source: str, used_families: set[str]) -> list[dict]:
    enriched = []
    for idx, record in enumerate(records, start=1):
        item = dict(record)
        item["bucket"] = bucket_map[(record["expected_neg_behavior"], record["scope_type"], record["semantic_mode"])]
        item["source"] = source
        item["priority"] = "new_family" if record["family_id"] not in used_families else "seen_family"
        item["candidate_id"] = f"{item['bucket']}__{idx:03d}"
        enriched.append(item)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/generated_balanced.jsonl")
    parser.add_argument(
        "--used",
        nargs="+",
        default=[
            "data/processed/splits/merged_excl_boost/train_clean_strict.jsonl",
            "data/processed/splits/balanced/dev_clean_strict.jsonl",
            "data/processed/validated_largetest_v2_clean_strict.jsonl",
        ],
    )
    parser.add_argument("--clean-output", default="outputs/router_boundary_clean_candidates.jsonl")
    parser.add_argument("--clean-tsv", default="outputs/router_boundary_clean_candidates.tsv")
    parser.add_argument("--audit-output", default="outputs/router_boundary_audit_candidates.jsonl")
    parser.add_argument("--audit-tsv", default="outputs/router_boundary_audit_candidates.tsv")
    parser.add_argument("--report", default="outputs/router_boundary_candidates_report.json")
    parser.add_argument("--max-clean-per-bucket", type=int, default=24)
    parser.add_argument("--max-audit-per-bucket", type=int, default=24)
    parser.add_argument("--max-per-family", type=int, default=2)
    args = parser.parse_args()

    raw_records = dedupe_records(load_jsonl(Path(args.raw)))
    used_records = dedupe_records([record for path in args.used for record in load_jsonl(Path(path))])
    used_keys = {content_key(record) for record in used_records}
    used_families = {str(record.get("family_id") or "") for record in used_records}

    leftover_records = [record for record in raw_records if content_key(record) not in used_keys]
    clean_candidates = []
    audit_candidates = []
    removed_reason_counts: Counter[str] = Counter()

    for record in leftover_records:
        combo = (record["expected_neg_behavior"], record["scope_type"], record["semantic_mode"])
        reason = removal_reason(record)
        if combo in TARGET_BUCKETS and reason is None:
            clean_candidates.append(record)
        elif combo in AUDIT_BUCKETS and reason is not None:
            item = dict(record)
            item["cleaning_reason"] = reason
            audit_candidates.append(item)
            removed_reason_counts[reason] += 1

    clean_candidates = add_metadata(
        sorted(clean_candidates, key=lambda record: prioritize({**record, "bucket": TARGET_BUCKETS[(record["expected_neg_behavior"], record["scope_type"], record["semantic_mode"])]}, used_families)),
        TARGET_BUCKETS,
        "leftover_clean_pool",
        used_families,
    )
    audit_candidates = add_metadata(
        sorted(audit_candidates, key=lambda record: prioritize({**record, "bucket": AUDIT_BUCKETS[(record["expected_neg_behavior"], record["scope_type"], record["semantic_mode"])]}, used_families)),
        AUDIT_BUCKETS,
        "leftover_audit_pool",
        used_families,
    )

    clean_selected = select_diverse(clean_candidates, args.max_clean_per_bucket, args.max_per_family)
    audit_selected = select_diverse(audit_candidates, args.max_audit_per_bucket, args.max_per_family)

    write_jsonl(Path(args.clean_output), clean_selected)
    write_tsv(Path(args.clean_tsv), clean_selected)
    write_jsonl(Path(args.audit_output), audit_selected)
    write_tsv(Path(args.audit_tsv), audit_selected)

    report = {
        "raw_unique_records": len(raw_records),
        "used_unique_records": len(used_records),
        "leftover_unique_records": len(leftover_records),
        "clean_candidate_pool": {
            "total": len(clean_candidates),
            "selected": len(clean_selected),
            "by_bucket_pool": dict(Counter(record["bucket"] for record in clean_candidates)),
            "by_bucket_selected": dict(Counter(record["bucket"] for record in clean_selected)),
            "new_family_selected": sum(record["priority"] == "new_family" for record in clean_selected),
        },
        "audit_candidate_pool": {
            "total": len(audit_candidates),
            "selected": len(audit_selected),
            "by_bucket_pool": dict(Counter(record["bucket"] for record in audit_candidates)),
            "by_bucket_selected": dict(Counter(record["bucket"] for record in audit_selected)),
            "cleaning_reasons": dict(removed_reason_counts),
        },
        "outputs": {
            "clean_jsonl": args.clean_output,
            "clean_tsv": args.clean_tsv,
            "audit_jsonl": args.audit_output,
            "audit_tsv": args.audit_tsv,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"clean_selected={len(clean_selected)} "
        f"{dict(Counter(record['bucket'] for record in clean_selected))}"
    )
    print(
        f"audit_selected={len(audit_selected)} "
        f"{dict(Counter(record['bucket'] for record in audit_selected))}"
    )
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
