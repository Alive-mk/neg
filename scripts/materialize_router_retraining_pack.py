from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(ROOT / "src"))

from neg_blindness.schema import ExperimentRecord
from neg_blindness.validators import dedupe_records, ensure_unique_record_ids, validate_record


PROMPT_NEG_REWRITES = {
    "preserve_double_exclusive__019": "You do not want to delay the office workflow, and you must never omit any required approval. What is the key to maintaining smooth workflow?",
    "preserve_double_exclusive__015": "You do not fail to wash your hands before cooking, and you never skip that step. What should you do first?",
    "preserve_double_exclusive__016": "You do not fail to wear a jacket when it is cold outside, and you never skip doing so. What should you wear?",
    "preserve_double_exclusive__007": "You do not fail to use a knife when cutting vegetables, and you never skip that tool. What tool do you use?",
    "preserve_double_exclusive__008": "It is not true that you do not brush your teeth in the morning, and you never skip that step. What should you do?",
    "preserve_double_exclusive__013": "It is not true that Alexander Graham Bell never invented the telephone. Who invented the telephone?",
    "preserve_double_exclusive__014": "It is not true that Isaac Newton never discovered gravity. Who discovered gravity?",
    "preserve_double_exclusive__005": "It is not true that the item never remains fragile. Which conclusion is correct?",
    "preserve_double_contrastive__065": "It is not true that a robin is never a bird.",
    "select_inscope_contrastive__151": "What practices are not safe or correct when using office staircases?",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[ExperimentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def counts(records: list[ExperimentRecord], field: str) -> dict[str, int]:
    return dict(Counter(getattr(record, field) for record in records))


def normalize_training_record(record_dict: dict, split_name: str) -> ExperimentRecord:
    item = {
        key: record_dict[key]
        for key in [
            "id",
            "neg_type",
            "scope_type",
            "semantic_mode",
            "expected_neg_behavior",
            "domain",
            "prompt_pos",
            "prompt_neg",
            "gold_pos",
            "gold_neg",
            "forbidden_neg",
            "candidate_pool_neg",
            "distractors",
            "template_id",
            "family_id",
            "entity_id",
            "metadata",
        ]
    }
    item["id"] = f"routerfix__{record_dict['candidate_id']}"
    metadata = dict(item.get("metadata") or {})
    metadata.update(
        {
            "targeted_router_fix": True,
            "targeted_source": split_name,
            "candidate_id": record_dict["candidate_id"],
            "review_tier": record_dict.get("review_tier"),
            "review_rationale": record_dict.get("review_rationale"),
            "review_decision": record_dict.get("decision"),
            "review_note": record_dict.get("decision_note"),
            "priority": record_dict.get("priority"),
            "bucket": record_dict.get("bucket"),
        }
    )
    item["metadata"] = metadata
    return ExperimentRecord.from_dict(item)


def apply_safe_fixes(record: ExperimentRecord) -> tuple[ExperimentRecord, list[str]]:
    fixes: list[str] = []
    candidate_id = str((record.metadata or {}).get("candidate_id") or "")
    if candidate_id in PROMPT_NEG_REWRITES:
        record.prompt_neg = PROMPT_NEG_REWRITES[candidate_id]
        fixes.append("prompt_neg_rewritten_for_schema")
    missing = [item for item in record.gold_pos if item not in record.forbidden_neg]
    if missing:
        record.forbidden_neg = list(record.forbidden_neg) + missing
        fixes.append("forbidden_neg_extended_with_gold_pos")
    return record, fixes


def split_valid_records(records: list[ExperimentRecord]) -> tuple[list[ExperimentRecord], list[dict]]:
    valid: list[ExperimentRecord] = []
    invalid: list[dict] = []
    for record in records:
        fixed, fixes = apply_safe_fixes(record)
        result = validate_record(fixed)
        if result.ok:
            if fixes:
                fixed.metadata = {
                    **fixed.metadata,
                    "safe_fixes": fixes,
                }
            valid.append(fixed)
        else:
            invalid.append(
                {
                    "record": fixed,
                    "errors": result.errors,
                    "safe_fixes": fixes,
                }
            )
    return valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="data/processed/splits/merged_excl_boost/train_clean_strict.jsonl")
    parser.add_argument("--priority-pass", default="outputs/router_boundary_priority_shortlist_pass.jsonl")
    parser.add_argument("--priority-reviewed", default="outputs/router_boundary_priority_shortlist_reviewed.jsonl")
    parser.add_argument("--audit-reviewed", default="outputs/router_boundary_audit_sentinels_reviewed.jsonl")
    parser.add_argument("--pack-dir", default="data/processed/splits/router_targeted_retraining_pack")
    parser.add_argument("--report", default="outputs/router_targeted_retraining_pack_report.json")
    args = parser.parse_args()

    base_train = [ExperimentRecord.from_dict(record) for record in load_jsonl(Path(args.base_train))]
    priority_pass_dicts = load_jsonl(Path(args.priority_pass))
    priority_reviewed_dicts = load_jsonl(Path(args.priority_reviewed))
    audit_reviewed_dicts = load_jsonl(Path(args.audit_reviewed))

    pass_records = [
        normalize_training_record(record, "priority_pass")
        for record in priority_pass_dicts
    ]
    moved_to_audit_records = [
        normalize_training_record(record, "priority_move_to_audit")
        for record in priority_reviewed_dicts
        if record.get("decision") == "move_to_audit"
    ]
    audit_records = moved_to_audit_records + [
        normalize_training_record(record, "audit_sentinel")
        for record in audit_reviewed_dicts
    ]

    pass_records, pass_unique_report = ensure_unique_record_ids(pass_records)
    audit_records, audit_unique_report = ensure_unique_record_ids(audit_records)
    valid_pass_records, invalid_pass_records = split_valid_records(pass_records)
    valid_audit_records, invalid_audit_records = split_valid_records(audit_records)

    base_keys = {record.canonical_key() for record in base_train}
    pass_deduped = [record for record in valid_pass_records if record.canonical_key() not in base_keys]
    dropped_overlap = [record.id for record in valid_pass_records if record.canonical_key() in base_keys]
    pass_deduped, pass_dupe_report = dedupe_records(pass_deduped)
    valid_pass_records = pass_deduped
    merged_train = base_train + pass_deduped
    merged_train, merged_unique_report = ensure_unique_record_ids(merged_train)
    merged_train, merged_dupe_report = dedupe_records(merged_train)

    pack_dir = Path(args.pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    pass_path = pack_dir / "priority_pass_train.jsonl"
    rewrite_path = pack_dir / "priority_pass_needs_rewrite.jsonl"
    audit_path = pack_dir / "audit_boundary_candidates.jsonl"
    merged_path = pack_dir / "train_clean_strict_plus_routerfix19.jsonl"

    write_jsonl(pass_path, valid_pass_records)
    rewrite_records = []
    for item in invalid_pass_records:
        record = item["record"]
        record.metadata = {
            **record.metadata,
            "rewrite_needed": True,
            "validation_errors": item["errors"],
            "safe_fixes": item["safe_fixes"],
        }
        rewrite_records.append(record)
    write_jsonl(rewrite_path, rewrite_records)
    audit_output_records = list(valid_audit_records)
    for item in invalid_audit_records:
        record = item["record"]
        record.metadata = {
            **record.metadata,
            "audit_only": True,
            "validation_errors": item["errors"],
            "safe_fixes": item["safe_fixes"],
        }
        audit_output_records.append(record)
    write_jsonl(audit_path, audit_output_records)
    write_jsonl(merged_path, merged_train)

    report = {
        "base_train": {
            "path": args.base_train,
            "total": len(base_train),
            "by_behavior": counts(base_train, "expected_neg_behavior"),
            "by_scope": counts(base_train, "scope_type"),
        },
        "priority_pass_input": {
            "total": len(priority_pass_dicts),
            "strict_valid_total": len(valid_pass_records),
            "needs_rewrite_total": len(rewrite_records),
            "dropped_overlap_with_base_train": dropped_overlap,
            "by_behavior": counts(valid_pass_records, "expected_neg_behavior"),
            "by_scope": counts(valid_pass_records, "scope_type"),
            "id_uniqueness": pass_unique_report,
            "dedupe_report": pass_dupe_report,
            "rewrite_candidates": [
                {
                    "id": item["record"].id,
                    "candidate_id": item["record"].metadata.get("candidate_id"),
                    "errors": item["errors"],
                    "safe_fixes": item["safe_fixes"],
                    "prompt_neg": item["record"].prompt_neg,
                }
                for item in invalid_pass_records
            ],
        },
        "audit_pack": {
            "total": len(audit_output_records),
            "strict_valid_total": len(valid_audit_records),
            "invalid_but_kept_for_audit_total": len(invalid_audit_records),
            "by_behavior": counts(audit_output_records, "expected_neg_behavior"),
            "by_scope": counts(audit_output_records, "scope_type"),
            "id_uniqueness": audit_unique_report,
            "invalid_examples": [
                {
                    "id": item["record"].id,
                    "candidate_id": item["record"].metadata.get("candidate_id"),
                    "errors": item["errors"],
                    "prompt_neg": item["record"].prompt_neg,
                }
                for item in invalid_audit_records[:20]
            ],
        },
        "merged_train": {
            "total": len(merged_train),
            "delta_vs_base": len(merged_train) - len(base_train),
            "by_behavior": counts(merged_train, "expected_neg_behavior"),
            "by_scope": counts(merged_train, "scope_type"),
            "id_uniqueness": merged_unique_report,
            "dedupe_report": merged_dupe_report,
        },
        "outputs": {
            "priority_pass_train": str(pass_path),
            "priority_pass_needs_rewrite": str(rewrite_path),
            "audit_boundary_candidates": str(audit_path),
            "merged_train": str(merged_path),
        },
        "recommended_train_command": (
            "python scripts/train_mgnm.py "
            "--models configs/model_config.json "
            "--model-name qwen2_5_7b "
            f"--train-input {merged_path} "
            "--eval-input data/processed/splits/router_calibration/calibration_clean_strict.jsonl "
            "--output-dir outputs/routerfix_qwen_targeted_phase1 "
            "--epochs 3 --learning-rate 2e-4 --weight-decay 0.01 "
            "--lambda-preserve 1.5 --lambda-rank 1.5 --lambda-rank-select 3.0 "
            "--lambda-ret 0.05 --behavior-token --scope-oversample 1 --dn-oversample 2"
        ),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"priority_pass_train={len(valid_pass_records)}")
    print(f"priority_pass_needs_rewrite={len(rewrite_records)}")
    print(f"audit_boundary_candidates={len(audit_output_records)}")
    print(f"merged_train={len(merged_train)} (delta={len(merged_train)-len(base_train)})")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
