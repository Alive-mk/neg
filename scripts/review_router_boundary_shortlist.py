from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PRIORITY_REVIEWS = {
    "preserve_double_exclusive__019": ("accept_train", "natural double-negation preserve case; strong office-workflow boundary"),
    "preserve_double_exclusive__015": ("accept_train", "natural daily-action preserve case; directly targets preserve->SELECT confusion"),
    "preserve_double_exclusive__016": ("accept_train", "natural weather/action preserve case; clean exclusive-choice structure"),
    "preserve_double_exclusive__024": ("accept_train", "clean preserve statement with realistic commonsense continuation"),
    "preserve_double_exclusive__007": ("accept_train", "slightly marked wording but still a valid preserve double-negation tool-choice case"),
    "preserve_double_exclusive__008": ("accept_train", "marked but usable morning-routine preserve case"),
    "preserve_double_exclusive__013": ("accept_train", "useful factual preserve question with compact candidate set"),
    "preserve_double_exclusive__014": ("accept_train", "useful factual preserve question with compact candidate set"),
    "preserve_double_exclusive__030": ("reject_unnatural", "adverbial double negation is too awkward for high-priority retraining"),
    "preserve_double_exclusive__005": ("accept_train", "clean lexical preserve item; useful low-variance control example"),
    "preserve_outscope_contrastive__099": ("move_to_audit", "negated prompt asks for unsafe behavior; current gold keeps safe behavior, so train semantics are mismatched"),
    "preserve_outscope_contrastive__097": ("move_to_audit", "prompt asks for unsuccessful completion; keeping successful steps should be audited, not trained"),
    "preserve_outscope_contrastive__087": ("move_to_audit", "instruction 'avoid outlining' conflicts with preserved positive answer"),
    "preserve_outscope_contrastive__090": ("move_to_audit", "prompt asks for ignored safety instructions; preserved positive answer should be audited as a boundary case"),
    "preserve_outscope_contrastive__101": ("move_to_audit", "question semantics are contrastive/argumentative rather than clean preserve supervision"),
    "preserve_outscope_contrastive__088": ("move_to_audit", "prompt shifts topic away from apples; preserved answer is likely distribution-specific rather than clean supervision"),
    "preserve_outscope_contrastive__098": ("move_to_audit", "explicit exclusion of dolphin conflicts with preserved positive answer containing dolphin"),
    "preserve_outscope_contrastive__110": ("move_to_audit", "explicit exclusion of 'sad' conflicts with preserved positive answer containing 'sad'"),
    "preserve_double_contrastive__074": ("accept_train", "good factual preserve-contrastive item; meaning is stable"),
    "preserve_double_contrastive__075": ("accept_train", "good factual preserve-contrastive item; meaning is stable"),
    "preserve_double_contrastive__065": ("accept_train", "simple lexical preserve-contrastive item; useful low-noise support"),
    "preserve_double_contrastive__072": ("reject_semantic_mismatch", "prompt implies uncommon winter snow but gold explains snow as common"),
    "select_inscope_exclusive__120": ("reject_semantic_mismatch", "prompt means 'must do', but label is framed as select over a forbidden alternative"),
    "select_inscope_exclusive__128": ("accept_train", "clean factual select example with stable negative answer"),
    "select_inscope_exclusive__133": ("accept_train", "clean factual select example with stable negative answer"),
    "select_inscope_exclusive__116": ("accept_train", "natural commonsense select example; directly useful for negated action choice"),
    "select_inscope_exclusive__129": ("accept_train", "reasonable task-step select boundary case"),
    "select_inscope_contrastive__147": ("accept_train", "useful instructional select contrastive example"),
    "select_inscope_contrastive__151": ("accept_train", "useful safety-domain select contrastive example"),
    "select_inscope_contrastive__168": ("accept_train", "negated yes/no lexical item is still semantically coherent and worth keeping as support"),
}


AUDIT_REVIEWS = {
    "audit_preserve_query_like__007": ("keep_audit", "query-like preserve boundary; likely should become select-style after audit"),
    "audit_preserve_query_like__006": ("keep_audit", "query-like preserve boundary; likely should become select-style after audit"),
    "audit_preserve_query_like__010": ("keep_audit", "query-like preserve boundary; explicit alternative choice semantics"),
    "audit_preserve_query_like__008": ("keep_audit", "query-like preserve boundary in factual domain; likely relabel candidate"),
    "audit_preserve_query_like__018": ("keep_audit", "clean label-boundary sentinel; likely relabel to select"),
    "audit_preserve_query_like__027": ("keep_audit", "instruction-following boundary; should remain audit-only until task definition is fixed"),
    "audit_preserve_query_like__029": ("keep_audit", "instruction-following boundary; should remain audit-only until task definition is fixed"),
    "audit_preserve_query_like__022": ("keep_audit", "query-like preserve boundary for task-order semantics"),
    "audit_suppress_query_like__056": ("keep_audit", "classic suppress-vs-select confusion; strong audit sentinel"),
    "audit_suppress_query_like__057": ("keep_audit", "classic suppress-vs-select confusion; strong audit sentinel"),
    "audit_suppress_query_like__064": ("keep_audit", "instructional query-like suppress boundary; likely relabel candidate"),
    "audit_suppress_query_like__068": ("keep_audit", "classic suppress-vs-select confusion; strong audit sentinel"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "decision",
        "decision_note",
        "bucket",
        "review_tier",
        "domain",
        "family_id",
        "template_id",
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
                    "decision": record["decision"],
                    "decision_note": record["decision_note"],
                    "bucket": record["bucket"],
                    "review_tier": record["review_tier"],
                    "domain": record.get("domain", ""),
                    "family_id": record.get("family_id", ""),
                    "template_id": record.get("template_id", ""),
                    "prompt_neg": record.get("prompt_neg", ""),
                    "gold_pos_0": (record.get("gold_pos") or [""])[0],
                    "gold_neg_0": (record.get("gold_neg") or [""])[0],
                    "candidate_pool_neg_preview": " | ".join((record.get("candidate_pool_neg") or [])[:3]),
                }
            )


def annotate(records: list[dict], decisions: dict[str, tuple[str, str]]) -> list[dict]:
    annotated = []
    missing = []
    for record in records:
        if record["candidate_id"] not in decisions:
            missing.append(record["candidate_id"])
            continue
        decision, note = decisions[record["candidate_id"]]
        item = dict(record)
        item["decision"] = decision
        item["decision_note"] = note
        annotated.append(item)
    if missing:
        raise ValueError(f"Missing review decisions for: {missing}")
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority-input", default="outputs/router_boundary_priority_shortlist.jsonl")
    parser.add_argument("--audit-input", default="outputs/router_boundary_audit_sentinels.jsonl")
    parser.add_argument("--priority-reviewed-jsonl", default="outputs/router_boundary_priority_shortlist_reviewed.jsonl")
    parser.add_argument("--priority-reviewed-tsv", default="outputs/router_boundary_priority_shortlist_reviewed.tsv")
    parser.add_argument("--priority-pass-jsonl", default="outputs/router_boundary_priority_shortlist_pass.jsonl")
    parser.add_argument("--priority-pass-tsv", default="outputs/router_boundary_priority_shortlist_pass.tsv")
    parser.add_argument("--audit-reviewed-jsonl", default="outputs/router_boundary_audit_sentinels_reviewed.jsonl")
    parser.add_argument("--audit-reviewed-tsv", default="outputs/router_boundary_audit_sentinels_reviewed.tsv")
    parser.add_argument("--report", default="outputs/router_boundary_review_report.json")
    args = parser.parse_args()

    priority_records = annotate(load_jsonl(Path(args.priority_input)), PRIORITY_REVIEWS)
    audit_records = annotate(load_jsonl(Path(args.audit_input)), AUDIT_REVIEWS)

    priority_pass = [record for record in priority_records if record["decision"] == "accept_train"]

    write_jsonl(Path(args.priority_reviewed_jsonl), priority_records)
    write_tsv(Path(args.priority_reviewed_tsv), priority_records)
    write_jsonl(Path(args.priority_pass_jsonl), priority_pass)
    write_tsv(Path(args.priority_pass_tsv), priority_pass)
    write_jsonl(Path(args.audit_reviewed_jsonl), audit_records)
    write_tsv(Path(args.audit_reviewed_tsv), audit_records)

    report = {
        "priority_total": len(priority_records),
        "priority_decisions": dict(Counter(record["decision"] for record in priority_records)),
        "priority_pass_total": len(priority_pass),
        "priority_pass_by_bucket": dict(Counter(record["bucket"] for record in priority_pass)),
        "audit_total": len(audit_records),
        "audit_decisions": dict(Counter(record["decision"] for record in audit_records)),
        "outputs": {
            "priority_reviewed_tsv": args.priority_reviewed_tsv,
            "priority_pass_tsv": args.priority_pass_tsv,
            "audit_reviewed_tsv": args.audit_reviewed_tsv,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"priority_decisions={report['priority_decisions']}")
    print(f"priority_pass_by_bucket={report['priority_pass_by_bucket']}")
    print(f"audit_decisions={report['audit_decisions']}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
