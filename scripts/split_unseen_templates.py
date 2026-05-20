from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, load_records, write_json
from neg_blindness.schema import ExperimentRecord, normalize_text
from neg_blindness.validators import dedupe_records, ensure_unique_record_ids


def behavior_counts(records: Iterable[ExperimentRecord]) -> dict[str, int]:
    return dict(Counter(record.expected_neg_behavior for record in records))


def domain_counts(records: Iterable[ExperimentRecord]) -> dict[str, int]:
    return dict(Counter(record.domain for record in records))


def primary_behavior(records: list[ExperimentRecord]) -> str:
    counts = Counter(record.expected_neg_behavior for record in records)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def split_template_ids(
    template_to_records: dict[str, list[ExperimentRecord]],
    test_template_ratio: float,
    dev_template_ratio: float,
    min_test_templates_per_behavior: int,
    rng: random.Random,
) -> tuple[set[str], set[str], set[str]]:
    behavior_to_templates: dict[str, list[str]] = defaultdict(list)
    for template_id, records in template_to_records.items():
        behavior_to_templates[primary_behavior(records)].append(template_id)

    test_templates: set[str] = set()
    dev_templates: set[str] = set()

    for behavior, template_ids in behavior_to_templates.items():
        shuffled = list(template_ids)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_template_ratio))
        if shuffled and test_template_ratio > 0:
            n_test = max(min_test_templates_per_behavior, n_test)
        n_test = min(n_test, max(0, len(shuffled) - 1))
        selected_test = set(shuffled[:n_test])
        test_templates.update(selected_test)

        remaining = [template_id for template_id in shuffled if template_id not in selected_test]
        n_dev = int(round(len(remaining) * dev_template_ratio))
        n_dev = min(n_dev, max(0, len(remaining) - 1))
        dev_templates.update(remaining[:n_dev])

    all_templates = set(template_to_records)
    train_templates = all_templates - test_templates - dev_templates
    return train_templates, dev_templates, test_templates


def build_seen_eval_split(
    records: list[ExperimentRecord],
    seen_eval_ratio: float,
    rng: random.Random,
) -> tuple[list[ExperimentRecord], list[ExperimentRecord]]:
    if seen_eval_ratio <= 0:
        return records, []

    template_to_records: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        template_to_records[record.template_id].append(record)

    train_records: list[ExperimentRecord] = []
    seen_eval_records: list[ExperimentRecord] = []

    for template_id in sorted(template_to_records):
        items = list(template_to_records[template_id])
        rng.shuffle(items)
        if len(items) < 2:
            train_records.extend(items)
            continue
        n_eval = int(round(len(items) * seen_eval_ratio))
        n_eval = min(max(1, n_eval), len(items) - 1)
        seen_eval_records.extend(items[:n_eval])
        train_records.extend(items[n_eval:])

    return train_records, seen_eval_records


def prompt_pairs(records: Iterable[ExperimentRecord]) -> set[tuple[str, str]]:
    return {
        (normalize_text(record.prompt_pos), normalize_text(record.prompt_neg))
        for record in records
    }


def overlap_report(train: list[ExperimentRecord], eval_records: list[ExperimentRecord]) -> dict:
    train_templates = {record.template_id for record in train}
    eval_templates = {record.template_id for record in eval_records}
    train_families = {record.family_id for record in train}
    eval_families = {record.family_id for record in eval_records}
    train_entities = {record.entity_id for record in train}
    eval_entities = {record.entity_id for record in eval_records}
    return {
        "template_overlap": len(train_templates & eval_templates),
        "family_overlap": len(train_families & eval_families),
        "entity_overlap": len(train_entities & eval_entities),
        "prompt_pair_overlap": len(prompt_pairs(train) & prompt_pairs(eval_records)),
        "eval_unseen_template_records": sum(
            record.template_id not in train_templates for record in eval_records
        ),
        "eval_records": len(eval_records),
    }


def remove_prompt_overlaps(
    train: list[ExperimentRecord],
    eval_records: list[ExperimentRecord],
) -> tuple[list[ExperimentRecord], list[ExperimentRecord]]:
    train_pairs = prompt_pairs(train)
    kept: list[ExperimentRecord] = []
    removed: list[ExperimentRecord] = []
    for record in eval_records:
        if (normalize_text(record.prompt_pos), normalize_text(record.prompt_neg)) in train_pairs:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def select_multianswer_stats(records: list[ExperimentRecord]) -> dict:
    select_records = [
        record for record in records if record.expected_neg_behavior == "select_gold_neg"
    ]
    with_valid = [record for record in select_records if record.valid_negatives]
    ambiguous = []
    for record in select_records:
        hard = set(record.gold_neg)
        multi = set(record.gold_neg + record.valid_negatives)
        if len(multi) > len(hard):
            ambiguous.append(record)
    return {
        "select_records": len(select_records),
        "select_with_valid_negatives": len(with_valid),
        "select_ambiguous": len(ambiguous),
        "select_ambiguous_rate": len(ambiguous) / len(select_records)
        if select_records
        else 0.0,
    }


def split_summary(records: list[ExperimentRecord]) -> dict:
    return {
        "records": len(records),
        "templates": len({record.template_id for record in records}),
        "families": len({record.family_id for record in records}),
        "entities": len({record.entity_id for record in records}),
        "behavior_counts": behavior_counts(records),
        "domain_counts": domain_counts(records),
        "select_multianswer": select_multianswer_stats(records),
    }


def load_inputs(paths: list[str]) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for path in paths:
        records.extend(load_records(path))
    return records


def apply_dedupe(
    records: list[ExperimentRecord],
    mode: str,
) -> tuple[list[ExperimentRecord], dict]:
    if mode == "none":
        return records, {"mode": mode}
    if mode == "canonical":
        records, dedupe_report = dedupe_records(records)
        records, unique_report = ensure_unique_record_ids(records)
        return records, {
            "mode": mode,
            "canonical_dedupe": dedupe_report,
            "unique_ids": unique_report,
        }
    if mode == "id":
        records, unique_report = ensure_unique_record_ids(records)
        return records, {"mode": mode, "unique_ids": unique_report}
    raise ValueError(f"Unsupported dedupe mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-template-ratio", type=float, default=0.3)
    parser.add_argument("--dev-template-ratio", type=float, default=0.0)
    parser.add_argument("--seen-eval-ratio", type=float, default=0.15)
    parser.add_argument("--min-test-templates-per-behavior", type=int, default=1)
    parser.add_argument(
        "--keep-prompt-overlap",
        action="store_true",
        help="Keep exact prompt overlaps between train and eval splits.",
    )
    parser.add_argument(
        "--dedupe",
        choices=["canonical", "id", "none"],
        default="canonical",
        help="canonical removes duplicate prompt/label records before splitting.",
    )
    args = parser.parse_args()

    if not 0 <= args.test_template_ratio < 1:
        raise ValueError("--test-template-ratio must be in [0, 1)")
    if not 0 <= args.dev_template_ratio < 1:
        raise ValueError("--dev-template-ratio must be in [0, 1)")
    if not 0 <= args.seen_eval_ratio < 1:
        raise ValueError("--seen-eval-ratio must be in [0, 1)")

    rng = random.Random(args.seed)
    loaded_records = load_inputs(args.inputs)
    records, dedupe_report = apply_dedupe(loaded_records, args.dedupe)

    template_to_records: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        template_to_records[record.template_id].append(record)

    train_templates, dev_templates, test_templates = split_template_ids(
        template_to_records=template_to_records,
        test_template_ratio=args.test_template_ratio,
        dev_template_ratio=args.dev_template_ratio,
        min_test_templates_per_behavior=args.min_test_templates_per_behavior,
        rng=rng,
    )

    train_pool = [
        record for record in records if record.template_id in train_templates
    ]
    dev_records = [
        record for record in records if record.template_id in dev_templates
    ]
    unseen_test_records = [
        record for record in records if record.template_id in test_templates
    ]
    train_records, seen_eval_records = build_seen_eval_split(
        records=train_pool,
        seen_eval_ratio=args.seen_eval_ratio,
        rng=rng,
    )
    removed_prompt_overlaps: dict[str, list[ExperimentRecord]] = {
        "seen_eval": [],
        "dev": [],
        "unseen_test": [],
    }
    if not args.keep_prompt_overlap:
        seen_eval_records, removed_prompt_overlaps["seen_eval"] = remove_prompt_overlaps(
            train_records,
            seen_eval_records,
        )
        dev_records, removed_prompt_overlaps["dev"] = remove_prompt_overlaps(
            train_records,
            dev_records,
        )
        unseen_test_records, removed_prompt_overlaps["unseen_test"] = remove_prompt_overlaps(
            train_records,
            unseen_test_records,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_records(output_dir / "train.jsonl", train_records)
    dump_records(output_dir / "unseen_test.jsonl", unseen_test_records)
    dump_records(output_dir / "seen_eval.jsonl", seen_eval_records)
    if dev_records:
        dump_records(output_dir / "dev.jsonl", dev_records)
    removed_records = [
        record
        for split_records in removed_prompt_overlaps.values()
        for record in split_records
    ]
    if removed_records:
        dump_records(output_dir / "prompt_overlap_removed.jsonl", removed_records)

    template_manifest = {
        "train_templates": sorted(train_templates),
        "dev_templates": sorted(dev_templates),
        "unseen_test_templates": sorted(test_templates),
    }
    write_json(output_dir / "template_manifest.json", template_manifest)

    report = {
        "inputs": args.inputs,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "requested": {
            "test_template_ratio": args.test_template_ratio,
            "dev_template_ratio": args.dev_template_ratio,
            "seen_eval_ratio": args.seen_eval_ratio,
            "min_test_templates_per_behavior": args.min_test_templates_per_behavior,
            "keep_prompt_overlap": args.keep_prompt_overlap,
        },
        "dedupe": dedupe_report,
        "loaded_records": len(loaded_records),
        "split_summary": {
            "train": split_summary(train_records),
            "seen_eval": split_summary(seen_eval_records),
            "dev": split_summary(dev_records),
            "unseen_test": split_summary(unseen_test_records),
        },
        "overlap": {
            "seen_eval_vs_train": overlap_report(train_records, seen_eval_records),
            "unseen_test_vs_train": overlap_report(train_records, unseen_test_records),
            "dev_vs_train": overlap_report(train_records, dev_records),
        },
        "template_counts": {
            "train": len(train_templates),
            "dev": len(dev_templates),
            "unseen_test": len(test_templates),
        },
        "removed_prompt_overlaps": {
            split_name: split_summary(split_records)
            for split_name, split_records in removed_prompt_overlaps.items()
        },
    }
    write_json(args.report, report)

    print(f"Loaded records: {len(loaded_records)}")
    print(f"Split records: train={len(train_records)}, seen_eval={len(seen_eval_records)}, "
          f"unseen_test={len(unseen_test_records)}, dev={len(dev_records)}")
    print("Train behavior:", behavior_counts(train_records))
    print("Seen-eval behavior:", behavior_counts(seen_eval_records))
    print("Unseen-test behavior:", behavior_counts(unseen_test_records))
    print(f"Template overlap train/unseen_test: {report['overlap']['unseen_test_vs_train']['template_overlap']}")
    print(f"Saved split: {output_dir}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
