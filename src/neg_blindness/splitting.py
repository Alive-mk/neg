from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable

from neg_blindness.schema import ExperimentRecord, normalize_text


def _group_records(
    records: list[ExperimentRecord],
    key_fn: Callable[[ExperimentRecord], str],
    allowed_ids: set[str],
) -> list[tuple[str, list[ExperimentRecord]]]:
    groups: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        if record.id not in allowed_ids:
            continue
        groups[key_fn(record)].append(record)
    return list(groups.items())


def _select_groups(
    grouped: list[tuple[str, list[ExperimentRecord]]],
    target_count: int,
    rng: random.Random,
) -> set[str]:
    if target_count <= 0:
        return set()
    grouped = list(grouped)
    rng.shuffle(grouped)
    grouped.sort(key=lambda pair: len(pair[1]), reverse=True)

    selected: set[str] = set()
    current = 0
    remaining = grouped[:]
    while remaining and current < target_count:
        remaining.sort(
            key=lambda pair: (
                abs((current + len(pair[1])) - target_count),
                len(pair[1]),
            )
        )
        key, items = remaining.pop(0)
        selected.add(key)
        current += len(items)
    return selected


def _assign_groups(
    records: list[ExperimentRecord],
    key_fn: Callable[[ExperimentRecord], str],
    split_name: str,
    target_count: int,
    assigned_ids: set[str],
    split_members: dict[str, set[str]],
    rng: random.Random,
) -> int:
    available_ids = {record.id for record in records if record.id not in assigned_ids}
    grouped = _group_records(records, key_fn, available_ids)
    selected_keys = _select_groups(grouped, target_count, rng)
    selected_count = 0
    for key, items in grouped:
        if key not in selected_keys:
            continue
        for record in items:
            split_members[split_name].add(record.id)
            assigned_ids.add(record.id)
            selected_count += 1
    return selected_count


def _prompt_pair(record: ExperimentRecord) -> tuple[str, str]:
    return (
        normalize_text(record.prompt_pos),
        normalize_text(record.prompt_neg),
    )


def _enforce_prompt_disjointness(
    records: list[ExperimentRecord],
    split_members: dict[str, set[str]],
) -> None:
    pair_to_records: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    id_to_split: dict[str, str] = {}
    split_priority = {"test": 0, "dev": 1, "train": 2}

    for split_name, ids in split_members.items():
        for record_id in ids:
            id_to_split[record_id] = split_name

    for record in records:
        pair_to_records[_prompt_pair(record)].append(record)

    for grouped_records in pair_to_records.values():
        split_counts = Counter(id_to_split[record.id] for record in grouped_records)
        if len(split_counts) <= 1:
            continue

        destination = sorted(
            split_counts.items(),
            key=lambda item: (-item[1], split_priority[item[0]]),
        )[0][0]
        grouped_ids = {record.id for record in grouped_records}

        for split_name in split_members:
            split_members[split_name].difference_update(grouped_ids)
        split_members[destination].update(grouped_ids)


def split_records(
    records: list[ExperimentRecord],
    dev_ratio: float = 0.1,
    test_ratio: float = 0.2,
    family_share: float = 0.5,
    entity_share: float = 0.3,
    template_share: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[str, list[ExperimentRecord]], dict]:
    rng = random.Random(random_seed)
    total = len(records)
    dev_target = int(total * dev_ratio)
    test_target = int(total * test_ratio)
    if total >= 3 and dev_ratio > 0:
        dev_target = max(1, dev_target)
    if total >= 4 and test_ratio > 0:
        test_target = max(1, test_target)
    if dev_target + test_target >= total and total > 0:
        overflow = (dev_target + test_target) - (total - 1)
        if overflow > 0:
            if test_target >= dev_target:
                test_target = max(0, test_target - overflow)
            else:
                dev_target = max(0, dev_target - overflow)

    split_members: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    assigned_ids: set[str] = set()

    phases = [
        ("family", lambda r: r.family_id, family_share),
        ("entity", lambda r: r.entity_id, entity_share),
        ("template", lambda r: r.template_id, template_share),
    ]

    phase_report: dict[str, dict[str, int]] = {"dev": {}, "test": {}}

    for split_name, target in (("test", test_target), ("dev", dev_target)):
        for phase_name, key_fn, share in phases:
            phase_target = int(target * share)
            assigned = _assign_groups(
                records=records,
                key_fn=key_fn,
                split_name=split_name,
                target_count=phase_target,
                assigned_ids=assigned_ids,
                split_members=split_members,
                rng=rng,
            )
            phase_report[split_name][phase_name] = assigned

        residual_target = target - len(split_members[split_name])
        if residual_target > 0:
            assigned = _assign_groups(
                records=records,
                key_fn=lambda r: r.template_id,
                split_name=split_name,
                target_count=residual_target,
                assigned_ids=assigned_ids,
                split_members=split_members,
                rng=rng,
            )
            phase_report[split_name]["template_residual"] = assigned

    for record in records:
        if record.id not in assigned_ids:
            split_members["train"].add(record.id)

    _enforce_prompt_disjointness(records, split_members)

    split_records_map: dict[str, list[ExperimentRecord]] = {}
    for split_name, ids in split_members.items():
        split_records_map[split_name] = [record for record in records if record.id in ids]

    report = build_split_report(split_records_map)
    report["phase_assignment"] = phase_report
    report["requested_ratios"] = {"dev_ratio": dev_ratio, "test_ratio": test_ratio}
    return split_records_map, report


def _distribution(records: list[ExperimentRecord], attr: str) -> dict[str, int]:
    return dict(Counter(getattr(record, attr) for record in records))


def _holdout_coverage(
    train_records: list[ExperimentRecord],
    eval_records: list[ExperimentRecord],
    attr: str,
) -> dict[str, float | int]:
    train_values = {getattr(record, attr) for record in train_records}
    unseen = [record for record in eval_records if getattr(record, attr) not in train_values]
    ratio = len(unseen) / len(eval_records) if eval_records else 0.0
    return {
        "unseen_records": len(unseen),
        "total_records": len(eval_records),
        "unseen_ratio": ratio,
    }


def _exact_prompt_overlap(
    left: list[ExperimentRecord],
    right: list[ExperimentRecord],
) -> int:
    left_pairs = {_prompt_pair(record) for record in left}
    right_pairs = {_prompt_pair(record) for record in right}
    return len(left_pairs & right_pairs)


def build_split_report(split_map: dict[str, list[ExperimentRecord]]) -> dict:
    train = split_map["train"]
    dev = split_map["dev"]
    test = split_map["test"]
    return {
        "split_sizes": {name: len(records) for name, records in split_map.items()},
        "distributions": {
            split_name: {
                "semantic_mode": _distribution(records, "semantic_mode"),
                "neg_type": _distribution(records, "neg_type"),
                "scope_type": _distribution(records, "scope_type"),
                "domain": _distribution(records, "domain"),
            }
            for split_name, records in split_map.items()
        },
        "holdout_coverage": {
            "dev": {
                "family_id": _holdout_coverage(train, dev, "family_id"),
                "entity_id": _holdout_coverage(train, dev, "entity_id"),
                "template_id": _holdout_coverage(train, dev, "template_id"),
            },
            "test": {
                "family_id": _holdout_coverage(train, test, "family_id"),
                "entity_id": _holdout_coverage(train, test, "entity_id"),
                "template_id": _holdout_coverage(train, test, "template_id"),
            },
        },
        "prompt_overlap": {
            "train_dev": _exact_prompt_overlap(train, dev),
            "train_test": _exact_prompt_overlap(train, test),
            "dev_test": _exact_prompt_overlap(dev, test),
        },
    }
