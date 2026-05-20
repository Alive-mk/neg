from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


CLEAN_QUOTAS = {
    "preserve_double_exclusive": 10,
    "preserve_outscope_contrastive": 8,
    "preserve_double_contrastive": 4,
    "select_inscope_exclusive": 5,
    "select_inscope_contrastive": 3,
}

AUDIT_QUOTAS = {
    "audit_preserve_query_like": 8,
    "audit_suppress_query_like": 4,
}

PREFERRED_CANDIDATE_IDS = {
    "preserve_double_exclusive": [
        "preserve_double_exclusive__019",
        "preserve_double_exclusive__009",
        "preserve_double_exclusive__015",
        "preserve_double_exclusive__016",
        "preserve_double_exclusive__024",
        "preserve_double_exclusive__007",
        "preserve_double_exclusive__008",
        "preserve_double_exclusive__013",
        "preserve_double_exclusive__014",
        "preserve_double_exclusive__030",
        "preserve_double_exclusive__025",
    ],
    "preserve_outscope_contrastive": [
        "preserve_outscope_contrastive__099",
        "preserve_outscope_contrastive__097",
        "preserve_outscope_contrastive__087",
        "preserve_outscope_contrastive__090",
        "preserve_outscope_contrastive__101",
        "preserve_outscope_contrastive__088",
        "preserve_outscope_contrastive__098",
        "preserve_outscope_contrastive__110",
    ],
    "preserve_double_contrastive": [
        "preserve_double_contrastive__074",
        "preserve_double_contrastive__075",
        "preserve_double_contrastive__065",
        "preserve_double_contrastive__066",
    ],
    "select_inscope_exclusive": [
        "select_inscope_exclusive__120",
        "select_inscope_exclusive__121",
        "select_inscope_exclusive__128",
        "select_inscope_exclusive__133",
        "select_inscope_exclusive__116",
        "select_inscope_exclusive__129",
    ],
    "select_inscope_contrastive": [
        "select_inscope_contrastive__147",
        "select_inscope_contrastive__151",
        "select_inscope_contrastive__163",
    ],
    "audit_preserve_query_like": [
        "audit_preserve_query_like__007",
        "audit_preserve_query_like__006",
        "audit_preserve_query_like__021",
        "audit_preserve_query_like__010",
        "audit_preserve_query_like__008",
        "audit_preserve_query_like__018",
        "audit_preserve_query_like__027",
        "audit_preserve_query_like__029",
    ],
    "audit_suppress_query_like": [
        "audit_suppress_query_like__055",
        "audit_suppress_query_like__056",
        "audit_suppress_query_like__057",
        "audit_suppress_query_like__060",
    ],
}

EXCLUDED_CANDIDATE_IDS = {
    "preserve_double_exclusive__017",
    "preserve_double_exclusive__018",
    "preserve_double_exclusive__035",
    "preserve_double_contrastive__060",
    "preserve_double_contrastive__069",
    "preserve_double_contrastive__071",
    "preserve_double_contrastive__085",
}

DOMAIN_PRIORITY = {
    "instructional": 0,
    "commonsense": 1,
    "factual": 2,
    "lexical": 3,
}

QUALITY_PENALTIES = [
    re.compile(r"\bdon'?t\s+never\b", re.I),
    re.compile(r"\bdoesn'?t\s+never\b", re.I),
    re.compile(r"\bnot\s+without\b", re.I),
    re.compile(r"\bnot\s+uncommon\b", re.I),
    re.compile(r"\bnot\s+unusual\b", re.I),
    re.compile(r"\bcannot\s+avoid\s+not\b", re.I),
    re.compile(r"\bcannot\s+not\s+consider\b", re.I),
    re.compile(r"\bcannot\s+be\s+without\s+not\b", re.I),
    re.compile(r"\bdo\s+not\s+need\s+to\s+use\b", re.I),
    re.compile(r"\bwhen\s+it\s+doesn'?t\s+rain\b", re.I),
    re.compile(r"\bif\s+you\s+do\s+not\s+fail\s+to\s+avoid\b", re.I),
    re.compile(r"\bis\s+it\s+(?:false|incorrect)\s+that\b", re.I),
    re.compile(r"\bafter\s+a\s+heavy\s+rain\b", re.I),
    re.compile(r"\bduring\s+a\s+snowstorm\b", re.I),
    re.compile(r"\bor any other city\b", re.I),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def count_penalties(text: str) -> int:
    return sum(int(pattern.search(text) is not None) for pattern in QUALITY_PENALTIES)


def clean_rank(record: dict) -> tuple:
    text = record.get("prompt_neg", "")
    domain = record.get("domain", "")
    topic = (record.get("metadata") or {}).get("topic", "")
    return (
        count_penalties(text),
        DOMAIN_PRIORITY.get(domain, 9),
        0 if record.get("priority") == "new_family" else 1,
        len(text),
        topic,
        text,
    )


def audit_rank(record: dict) -> tuple:
    text = record.get("prompt_neg", "")
    reason = record.get("cleaning_reason", "")
    return (
        0 if record.get("priority") == "new_family" else 1,
        0 if reason == "nonselect_direct_negated_selection" else 1,
        DOMAIN_PRIORITY.get(record.get("domain", ""), 9),
        len(text),
        text,
    )


def select_with_quotas(records: list[dict], quotas: dict[str, int], rank_fn) -> list[dict]:
    records = [record for record in records if record["candidate_id"] not in EXCLUDED_CANDIDATE_IDS]
    selected: list[dict] = []
    family_counts: Counter[str] = Counter()
    by_bucket = {}
    by_id = {record["candidate_id"]: record for record in records}
    for bucket in quotas:
        by_bucket[bucket] = sorted(
            [record for record in records if record["bucket"] == bucket],
            key=rank_fn,
        )

    for bucket, quota in quotas.items():
        bucket_selected = 0
        for candidate_id in PREFERRED_CANDIDATE_IDS.get(bucket, []):
            if bucket_selected >= quota:
                break
            record = by_id.get(candidate_id)
            if record is None or record["bucket"] != bucket:
                continue
            if record in selected:
                continue
            if family_counts[record["family_id"]] >= 1:
                continue
            selected.append(record)
            family_counts[record["family_id"]] += 1
            bucket_selected += 1

        for record in by_bucket[bucket]:
            if bucket_selected >= quota:
                break
            if family_counts[record["family_id"]] >= 1:
                continue
            selected.append(record)
            family_counts[record["family_id"]] += 1
            bucket_selected += 1

        if bucket_selected < quota:
            for record in by_bucket[bucket]:
                if bucket_selected >= quota:
                    break
                if record in selected:
                    continue
                selected.append(record)
                family_counts[record["family_id"]] += 1
                bucket_selected += 1

    return selected


def annotate_clean(records: list[dict]) -> list[dict]:
    tier_map = {
        "preserve_double_exclusive": "tier1_router_fix",
        "preserve_outscope_contrastive": "tier1_router_fix",
        "preserve_double_contrastive": "tier2_support",
        "select_inscope_exclusive": "tier2_support",
        "select_inscope_contrastive": "tier2_support",
    }
    rationale_map = {
        "preserve_double_exclusive": "targets preserve->SELECT double-negation errors",
        "preserve_outscope_contrastive": "targets preserve/query-like out-of-scope confusion",
        "preserve_double_contrastive": "broadens preserve contrastive coverage",
        "select_inscope_exclusive": "adds exclusive-choice select boundary cases",
        "select_inscope_contrastive": "adds contrastive select audit coverage",
    }
    out = []
    for record in records:
        item = dict(record)
        item["review_tier"] = tier_map[item["bucket"]]
        item["recommended_action"] = "candidate_for_targeted_retraining"
        item["review_rationale"] = rationale_map[item["bucket"]]
        out.append(item)
    return out


def annotate_audit(records: list[dict]) -> list[dict]:
    rationale_map = {
        "audit_preserve_query_like": "query-like preserve boundary; audit semantics before training",
        "audit_suppress_query_like": "query-like suppress boundary; audit label direction before training",
    }
    out = []
    for record in records:
        item = dict(record)
        item["review_tier"] = "audit_sentinel"
        item["recommended_action"] = "audit_only_do_not_train"
        item["review_rationale"] = rationale_map[item["bucket"]]
        out.append(item)
    return out


def write_tsv(path: Path, records: list[dict]) -> None:
    fields = [
        "candidate_id",
        "bucket",
        "review_tier",
        "recommended_action",
        "review_rationale",
        "priority",
        "cleaning_reason",
        "domain",
        "family_id",
        "entity_id",
        "template_id",
        "prompt_neg",
        "gold_pos_0",
        "gold_neg_0",
        "candidate_pool_neg_preview",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "bucket": record["bucket"],
                    "review_tier": record["review_tier"],
                    "recommended_action": record["recommended_action"],
                    "review_rationale": record["review_rationale"],
                    "priority": record.get("priority", ""),
                    "cleaning_reason": record.get("cleaning_reason", ""),
                    "domain": record.get("domain", ""),
                    "family_id": record.get("family_id", ""),
                    "entity_id": record.get("entity_id", ""),
                    "template_id": record.get("template_id", ""),
                    "prompt_neg": record.get("prompt_neg", ""),
                    "gold_pos_0": (record.get("gold_pos") or [""])[0],
                    "gold_neg_0": (record.get("gold_neg") or [""])[0],
                    "candidate_pool_neg_preview": " | ".join((record.get("candidate_pool_neg") or [])[:3]),
                }
            )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", default="outputs/router_boundary_clean_candidates.jsonl")
    parser.add_argument("--audit-input", default="outputs/router_boundary_audit_candidates.jsonl")
    parser.add_argument("--clean-shortlist-jsonl", default="outputs/router_boundary_priority_shortlist.jsonl")
    parser.add_argument("--clean-shortlist-tsv", default="outputs/router_boundary_priority_shortlist.tsv")
    parser.add_argument("--audit-shortlist-jsonl", default="outputs/router_boundary_audit_sentinels.jsonl")
    parser.add_argument("--audit-shortlist-tsv", default="outputs/router_boundary_audit_sentinels.tsv")
    parser.add_argument("--report", default="outputs/router_boundary_shortlist_report.json")
    args = parser.parse_args()

    clean_candidates = load_jsonl(Path(args.clean_input))
    audit_candidates = load_jsonl(Path(args.audit_input))

    clean_shortlist = annotate_clean(select_with_quotas(clean_candidates, CLEAN_QUOTAS, clean_rank))
    audit_shortlist = annotate_audit(select_with_quotas(audit_candidates, AUDIT_QUOTAS, audit_rank))

    write_jsonl(Path(args.clean_shortlist_jsonl), clean_shortlist)
    write_tsv(Path(args.clean_shortlist_tsv), clean_shortlist)
    write_jsonl(Path(args.audit_shortlist_jsonl), audit_shortlist)
    write_tsv(Path(args.audit_shortlist_tsv), audit_shortlist)

    report = {
        "clean_shortlist": {
            "total": len(clean_shortlist),
            "by_bucket": dict(Counter(record["bucket"] for record in clean_shortlist)),
            "new_family": sum(record.get("priority") == "new_family" for record in clean_shortlist),
        },
        "audit_shortlist": {
            "total": len(audit_shortlist),
            "by_bucket": dict(Counter(record["bucket"] for record in audit_shortlist)),
            "new_family": sum(record.get("priority") == "new_family" for record in audit_shortlist),
        },
        "outputs": {
            "clean_jsonl": args.clean_shortlist_jsonl,
            "clean_tsv": args.clean_shortlist_tsv,
            "audit_jsonl": args.audit_shortlist_jsonl,
            "audit_tsv": args.audit_shortlist_tsv,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"clean_shortlist={len(clean_shortlist)} {dict(Counter(r['bucket'] for r in clean_shortlist))}")
    print(f"audit_shortlist={len(audit_shortlist)} {dict(Counter(r['bucket'] for r in audit_shortlist))}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
