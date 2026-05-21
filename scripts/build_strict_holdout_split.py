from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, load_records, write_json  # noqa: E402
from neg_blindness.schema import ExperimentRecord, normalize_text  # noqa: E402
from neg_blindness.validators import dedupe_records, ensure_unique_record_ids  # noqa: E402


LEAKAGE_FIELDS = ("prompt_pair", "template_id", "family_id", "entity_id", "source_record_id")
BEHAVIORS = ("suppress_target", "select_gold_neg", "preserve_positive")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def read_inputs(paths: list[str]) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for path in paths:
        records.extend(load_records(path))
    return records


def source_record_id(record: ExperimentRecord) -> str:
    value = record.metadata.get("source_id") or record.id
    return str(value).strip()


def leakage_keys(record: ExperimentRecord, fields: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for field in fields:
        if field == "prompt_pair":
            value = (
                f"{normalize_text(record.prompt_pos)} || "
                f"{normalize_text(record.prompt_neg)}"
            )
        elif field == "template_id":
            value = record.template_id
        elif field == "family_id":
            value = record.family_id
        elif field == "entity_id":
            value = record.entity_id
        elif field == "source_record_id":
            value = source_record_id(record)
        else:
            raise ValueError(f"Unsupported leakage field: {field}")
        text = normalize_text(str(value))
        if text:
            keys.append(f"{field}:{text}")
    return keys


def build_components(
    records: list[ExperimentRecord],
    fields: Iterable[str],
) -> tuple[list[list[ExperimentRecord]], dict[str, int]]:
    union_find = UnionFind(len(records))
    key_to_index: dict[str, int] = {}
    key_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        for key in leakage_keys(record, fields):
            key_counts[key.split(":", 1)[0]] += 1
            if key in key_to_index:
                union_find.union(index, key_to_index[key])
            else:
                key_to_index[key] = index

    grouped: dict[int, list[ExperimentRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union_find.find(index)].append(record)

    components = sorted(
        grouped.values(),
        key=lambda items: (-len(items), sorted(record.id for record in items)[0]),
    )
    return components, dict(key_counts)


def behavior_counter(records: Iterable[ExperimentRecord]) -> Counter[str]:
    return Counter(record.expected_neg_behavior for record in records)


def component_summary(component: list[ExperimentRecord]) -> dict[str, object]:
    return {
        "records": len(component),
        "behaviors": dict(behavior_counter(component)),
        "templates": len({record.template_id for record in component}),
        "families": len({record.family_id for record in component}),
        "entities": len({record.entity_id for record in component}),
    }


def split_target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    names = list(ratios)
    raw = {name: total * ratios[name] for name in names}
    targets = {name: int(raw[name]) for name in names}
    missing = total - sum(targets.values())
    for name in sorted(names, key=lambda item: raw[item] - targets[item], reverse=True):
        if missing <= 0:
            break
        targets[name] += 1
        missing -= 1
    return targets


def score_assignment(
    current_size: int,
    target_size: int,
    current_behaviors: Counter[str],
    target_behaviors: dict[str, int],
    component: list[ExperimentRecord],
) -> tuple[float, int, str]:
    projected_size = current_size + len(component)
    size_overflow = max(0, projected_size - target_size)
    size_gap = abs(target_size - projected_size)
    comp_behaviors = behavior_counter(component)
    behavior_gap = 0
    for behavior in BEHAVIORS:
        projected = current_behaviors[behavior] + comp_behaviors[behavior]
        behavior_gap += abs(target_behaviors.get(behavior, 0) - projected)
    return (size_overflow * 1000 + size_gap + behavior_gap * 0.25, size_overflow, "")


def assign_components(
    components: list[list[ExperimentRecord]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[ExperimentRecord]]:
    rng = random.Random(seed)
    shuffled = list(components)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    total_records = sum(len(component) for component in components)
    target_sizes = split_target_counts(total_records, ratios)
    total_behaviors = behavior_counter(
        record for component in components for record in component
    )
    target_behaviors = {
        split: {
            behavior: int(round(total_behaviors[behavior] * ratios[split]))
            for behavior in BEHAVIORS
        }
        for split in ratios
    }

    output: dict[str, list[ExperimentRecord]] = {split: [] for split in ratios}
    output_behaviors: dict[str, Counter[str]] = {
        split: Counter() for split in ratios
    }

    for component in shuffled:
        best_split = min(
            ratios,
            key=lambda split: (
                score_assignment(
                    current_size=len(output[split]),
                    target_size=target_sizes[split],
                    current_behaviors=output_behaviors[split],
                    target_behaviors=target_behaviors[split],
                    component=component,
                ),
                split,
            ),
        )
        output[best_split].extend(component)
        output_behaviors[best_split].update(behavior_counter(component))

    return output


def parse_ratios(value: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("--ratios must use split=float items")
        name, raw_ratio = item.split("=", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError("split name must be non-empty")
        ratio = float(raw_ratio)
        if ratio < 0:
            raise argparse.ArgumentTypeError("split ratio must be non-negative")
        ratios[name] = ratio
    total = sum(ratios.values())
    if not ratios or total <= 0:
        raise argparse.ArgumentTypeError("at least one positive ratio is required")
    return {name: ratio / total for name, ratio in ratios.items()}


def parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(fields) - set(LEAKAGE_FIELDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported leakage field(s): {unknown}")
    return fields


def write_split_files(output_dir: Path, splits: dict[str, list[ExperimentRecord]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        dump_records(output_dir / f"{split}.jsonl", records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=parse_ratios("train=0.75,dev=0.10,test=0.15"),
        help="Comma-separated split ratios, for example train=0.75,dev=0.10,test=0.15.",
    )
    parser.add_argument(
        "--leakage-fields",
        type=parse_fields,
        default=LEAKAGE_FIELDS,
        help="Comma-separated connected-component keys. Default checks prompt/template/family/entity/source.",
    )
    parser.add_argument(
        "--dedupe",
        choices=["canonical", "id", "none"],
        default="canonical",
        help="canonical removes duplicate prompt/label records before component splitting.",
    )
    args = parser.parse_args()

    loaded_records = read_inputs(args.inputs)
    if args.dedupe == "canonical":
        records, dedupe_report = dedupe_records(loaded_records)
        records, unique_id_report = ensure_unique_record_ids(records)
    elif args.dedupe == "id":
        records, unique_id_report = ensure_unique_record_ids(loaded_records)
        dedupe_report = {"mode": "id", "input_records": len(loaded_records), "output_records": len(records)}
    else:
        records = loaded_records
        dedupe_report = {"mode": "none", "input_records": len(loaded_records), "output_records": len(records)}
        unique_id_report = {"mode": "none"}

    components, key_counts = build_components(records, args.leakage_fields)
    splits = assign_components(components, args.ratios, args.seed)
    output_dir = Path(args.output_dir)
    write_split_files(output_dir, splits)

    report = {
        "inputs": args.inputs,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": args.ratios,
        "leakage_fields": list(args.leakage_fields),
        "loaded_records": len(loaded_records),
        "records_after_dedupe": len(records),
        "dedupe_report": dedupe_report,
        "unique_id_report": unique_id_report,
        "component_count": len(components),
        "component_size_counts": dict(Counter(len(component) for component in components)),
        "largest_components": [component_summary(component) for component in components[:20]],
        "key_counts": key_counts,
        "split_summary": {
            split: {
                "records": len(split_records),
                "behaviors": dict(behavior_counter(split_records)),
                "templates": len({record.template_id for record in split_records}),
                "families": len({record.family_id for record in split_records}),
                "entities": len({record.entity_id for record in split_records}),
            }
            for split, split_records in splits.items()
        },
    }
    write_json(args.report, report)

    print(f"Loaded records: {len(loaded_records)}")
    print(f"Records after dedupe: {len(records)}")
    print(f"Components: {len(components)}")
    print("Split summary:", json.dumps(report["split_summary"], ensure_ascii=False))
    print(f"Saved split: {output_dir}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
