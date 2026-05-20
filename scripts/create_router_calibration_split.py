from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_predictions(path: Path, predictions: dict[str, str], ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = {rid: predictions[rid] for rid in predictions if rid in ids}
    path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")


def behavior_counts(records: list[dict]) -> Counter[str]:
    return Counter(record["expected_neg_behavior"] for record in records)


def score_assignment(
    current_counts: Counter[str],
    add_counts: Counter[str],
    target_counts: dict[str, int],
    target_total: int,
    current_total: int,
) -> tuple[float, float]:
    next_total = current_total + sum(add_counts.values())
    size_gap = abs(next_total - target_total)
    class_gap = 0.0
    for label, target in target_counts.items():
        class_gap += abs((current_counts[label] + add_counts[label]) - target)
    return (class_gap, size_gap)


def stratified_group_split(records: list[dict], calibration_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        group_id = str(record.get("family_id") or record["id"])
        groups[group_id].append(record)

    target_total = round(len(records) * calibration_ratio)
    overall_counts = behavior_counts(records)
    target_counts = {
        label: round(count * calibration_ratio)
        for label, count in overall_counts.items()
    }

    rng = random.Random(seed)
    grouped_items = list(groups.items())
    rng.shuffle(grouped_items)
    grouped_items.sort(
        key=lambda item: (
            -len(item[1]),
            -max(Counter(r["expected_neg_behavior"] for r in item[1]).values()),
            item[0],
        )
    )

    calibration_groups: set[str] = set()
    calibration_counts: Counter[str] = Counter()
    calibration_total = 0

    for group_id, group_records in grouped_items:
        group_counts = behavior_counts(group_records)
        if calibration_total >= target_total:
            continue
        take_score = score_assignment(
            calibration_counts,
            group_counts,
            target_counts,
            target_total,
            calibration_total,
        )
        skip_score = score_assignment(
            calibration_counts,
            Counter(),
            target_counts,
            target_total,
            calibration_total,
        )
        if take_score <= skip_score:
            calibration_groups.add(group_id)
            calibration_counts.update(group_counts)
            calibration_total += len(group_records)

    if not calibration_groups:
        first_group_id, first_group_records = grouped_items[0]
        calibration_groups.add(first_group_id)
        calibration_counts.update(behavior_counts(first_group_records))

    calibration_records = [record for group_id, group in groups.items() if group_id in calibration_groups for record in group]
    fit_records = [record for group_id, group in groups.items() if group_id not in calibration_groups for record in group]
    return fit_records, calibration_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/splits/merged_excl_boost/train_clean_strict.jsonl")
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit-output", default="data/processed/splits/router_calibration/train_fit_clean_strict.jsonl")
    parser.add_argument("--calibration-output", default="data/processed/splits/router_calibration/calibration_clean_strict.jsonl")
    parser.add_argument("--llm-predictions")
    parser.add_argument("--fit-llm-output")
    parser.add_argument("--calibration-llm-output")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    fit_records, calibration_records = stratified_group_split(records, args.calibration_ratio, args.seed)

    write_jsonl(Path(args.fit_output), fit_records)
    write_jsonl(Path(args.calibration_output), calibration_records)

    llm_predictions = load_predictions(Path(args.llm_predictions)) if args.llm_predictions else None
    if llm_predictions is not None:
        fit_ids = {record["id"] for record in fit_records}
        calibration_ids = {record["id"] for record in calibration_records}
        if args.fit_llm_output:
            write_predictions(Path(args.fit_llm_output), llm_predictions, fit_ids)
        if args.calibration_llm_output:
            write_predictions(Path(args.calibration_llm_output), llm_predictions, calibration_ids)

    print(
        f"fit={len(fit_records)} calibration={len(calibration_records)} "
        f"fit_counts={dict(behavior_counts(fit_records))} "
        f"calibration_counts={dict(behavior_counts(calibration_records))}"
    )
    fit_families = {str(record.get('family_id') or record['id']) for record in fit_records}
    calibration_families = {str(record.get('family_id') or record['id']) for record in calibration_records}
    print(
        f"family_overlap={len(fit_families & calibration_families)} "
        f"fit_families={len(fit_families)} calibration_families={len(calibration_families)}"
    )


if __name__ == "__main__":
    main()
