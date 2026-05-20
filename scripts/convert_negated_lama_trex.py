from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, write_json  # noqa: E402
from neg_blindness.schema import ExperimentRecord, normalize_text  # noqa: E402
from neg_blindness.validators import validate_record  # noqa: E402


BROKEN_NEGATED_TEMPLATES = {"P103", "P190"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_relations(path: Path) -> dict[str, dict[str, Any]]:
    return {row["relation"]: row for row in load_jsonl(path)}


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def render_prefix(template: str, subject: str) -> str:
    if "[Y]" not in template:
        raise ValueError(f"template lacks [Y]: {template}")
    before_y = template.split("[Y]", 1)[0]
    if "[X]" not in before_y:
        raise ValueError(f"template must place [X] before [Y]: {template}")
    return before_y.replace("[X]", subject).strip()


def stable_sample(
    values: list[str],
    n: int,
    seed: int,
    key: str,
) -> list[str]:
    values = unique(values)
    rng = random.Random(f"{seed}:{key}")
    rng.shuffle(values)
    return values[:n]


def source_files(input_root: Path, relations: set[str] | None) -> list[Path]:
    trex_root = input_root / "TREx"
    files = sorted(trex_root.glob("*.jsonl"))
    if relations:
        files = [path for path in files if path.stem in relations]
    return files


def build_record(
    row: dict[str, Any],
    relation_info: dict[str, Any],
    candidate_pool: list[str],
    id_prefix: str,
) -> ExperimentRecord:
    relation = str(row["predicate_id"])
    subject = str(row.get("sub_label") or row.get("sub_uri") or row["uuid"])
    gold = str(row["obj_label"]).strip()
    prompt_pos = render_prefix(str(relation_info["template"]), subject)
    prompt_neg = render_prefix(str(relation_info["template_negated"]), subject)
    record_id = f"{id_prefix}_{relation}_{sha1(str(row['uuid']).encode('utf-8')).hexdigest()[:12]}"

    metadata = {
        "source_benchmark": "negated_lama",
        "source_subset": "TREx",
        "source_uuid": row.get("uuid"),
        "relation": relation,
        "relation_label": relation_info.get("label"),
        "relation_type": relation_info.get("type"),
    }

    return ExperimentRecord(
        id=record_id,
        neg_type="sentential",
        scope_type="in_scope",
        semantic_mode="suppression_only",
        expected_neg_behavior="suppress_target",
        domain="factual",
        prompt_pos=prompt_pos,
        prompt_neg=prompt_neg,
        gold_pos=[gold],
        gold_neg=[],
        forbidden_neg=[gold],
        candidate_pool_neg=candidate_pool,
        valid_negatives=[],
        distractors=[],
        template_id=f"negated_lama_trex_{relation}",
        family_id=f"negated_lama_trex_{relation}",
        entity_id=f"negated_lama_trex_{row.get('sub_uri') or subject}",
        metadata=metadata,
    ).with_defaults()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--id-prefix", default="nlamatrex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-records-per-relation", type=int, default=100)
    parser.add_argument("--relations", nargs="*")
    parser.add_argument(
        "--include-broken-template-relations",
        action="store_true",
        help="Include relations with known inconsistent negated templates.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    relation_filter = set(args.relations) if args.relations else None
    relations = load_relations(input_root / "relations.jsonl")

    rows_by_relation: dict[str, list[dict[str, Any]]] = {}
    skipped_relations: dict[str, str] = {}
    for path in source_files(input_root, relation_filter):
        relation = path.stem
        if relation not in relations:
            skipped_relations[relation] = "missing relation metadata"
            continue
        if relation in BROKEN_NEGATED_TEMPLATES and not args.include_broken_template_relations:
            skipped_relations[relation] = "known inconsistent template_negated"
            continue
        rows_by_relation[relation] = load_jsonl(path)

    object_pool_by_relation = {
        relation: unique([str(row.get("obj_label", "")).strip() for row in rows])
        for relation, rows in rows_by_relation.items()
    }

    records: list[ExperimentRecord] = []
    validation_errors: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for relation, rows in sorted(rows_by_relation.items()):
        selected = stable_sample(
            [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows],
            args.max_records_per_relation,
            args.seed,
            relation,
        )
        selected_rows = [json.loads(item) for item in selected]
        selected_counts[relation] = len(selected_rows)
        relation_pool = object_pool_by_relation[relation]
        for row in selected_rows:
            gold = str(row.get("obj_label", "")).strip()
            candidates = stable_sample(
                [item for item in relation_pool if normalize_text(item) != normalize_text(gold)],
                args.max_candidates,
                args.seed,
                str(row.get("uuid", "")),
            )
            if not gold or not candidates:
                validation_errors["missing gold or candidates"] += 1
                continue
            try:
                record = build_record(row, relations[relation], candidates, args.id_prefix)
            except Exception as exc:  # noqa: BLE001
                validation_errors[f"convert_error: {exc}"] += 1
                continue
            errors = [
                error for error in validate_record(record).errors
                if error != "distractors must be non-empty"
            ]
            if errors:
                for error in errors:
                    validation_errors[error] += 1
                continue
            records.append(record)

    dump_records(args.output, records)
    report = {
        "input_root": args.input_root,
        "output": args.output,
        "rows": len(records),
        "relations": len(selected_counts),
        "selected_counts": dict(selected_counts),
        "skipped_relations": skipped_relations,
        "max_candidates": args.max_candidates,
        "max_records_per_relation": args.max_records_per_relation,
        "behavior_counts": dict(Counter(record.expected_neg_behavior for record in records)),
        "template_count": len({record.template_id for record in records}),
        "validation_errors": dict(validation_errors),
    }
    write_json(args.report, report)

    print(f"Converted records: {len(records)}")
    print(f"Relations: {len(selected_counts)}")
    print(f"Skipped relations: {len(skipped_relations)}")
    print(f"Saved: {args.output}")
    print(f"Report: {args.report}")
    if validation_errors:
        print("Validation skips:", dict(validation_errors))


if __name__ == "__main__":
    main()
