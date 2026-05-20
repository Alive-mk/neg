from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_records(records: list[dict[str, Any]], cap: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if cap is None or cap <= 0 or len(records) <= cap:
        return list(records)
    chosen = list(records)
    rng.shuffle(chosen)
    return chosen[:cap]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="data/processed/splits/merged_excl_boost/train_clean_strict.jsonl")
    parser.add_argument("--extra-train", default="data/processed/splits/router_targeted_retraining_pack/priority_pass_train.jsonl")
    parser.add_argument("--output", default="data/processed/splits/router_targeted_retraining_pack/phase2_focus_mix.jsonl")
    parser.add_argument("--report", default="outputs/routerfix_phase2_focus_mix_report.json")
    parser.add_argument("--extra-repeat", type=int, default=4)
    parser.add_argument("--select-cap", type=int, default=64)
    parser.add_argument("--suppress-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_records = load_jsonl(Path(args.base_train))
    extra_records = load_jsonl(Path(args.extra_train))

    target_signatures = {
        (
            record["expected_neg_behavior"],
            record["scope_type"],
            record["semantic_mode"],
        )
        for record in extra_records
    }

    base_target = [
        record
        for record in base_records
        if (
            record["expected_neg_behavior"],
            record["scope_type"],
            record["semantic_mode"],
        )
        in target_signatures
    ]
    base_preserve = [r for r in base_target if r["expected_neg_behavior"] == "preserve_positive"]
    base_select = [r for r in base_target if r["expected_neg_behavior"] == "select_gold_neg"]
    base_suppress = [r for r in base_records if r["expected_neg_behavior"] == "suppress_target"]

    chosen_preserve = list(base_preserve)
    chosen_select = sample_records(base_select, args.select_cap, rng)
    chosen_suppress = sample_records(base_suppress, args.suppress_cap, rng)

    repeated_extra: list[dict[str, Any]] = []
    for repeat_idx in range(max(1, args.extra_repeat)):
        for record in extra_records:
            copied = dict(record)
            copied["id"] = f"{record['id']}__mixr{repeat_idx + 1}"
            metadata = dict(copied.get("metadata") or {})
            metadata["phase2_focus_repeat"] = repeat_idx + 1
            copied["metadata"] = metadata
            repeated_extra.append(copied)

    mixed_records = chosen_preserve + chosen_select + chosen_suppress + repeated_extra
    write_jsonl(Path(args.output), mixed_records)

    report = {
        "base_train": args.base_train,
        "extra_train": args.extra_train,
        "target_signatures": sorted(list(target_signatures)),
        "counts": {
            "base_preserve_all": len(base_preserve),
            "base_select_all": len(base_select),
            "base_suppress_all": len(base_suppress),
            "chosen_preserve": len(chosen_preserve),
            "chosen_select": len(chosen_select),
            "chosen_suppress": len(chosen_suppress),
            "extra_records": len(extra_records),
            "extra_repeat": args.extra_repeat,
            "extra_effective": len(repeated_extra),
            "mixed_total": len(mixed_records),
        },
        "mixed_behavior": dict(Counter(r["expected_neg_behavior"] for r in mixed_records)),
        "mixed_scope": dict(Counter(r["scope_type"] for r in mixed_records)),
        "mixed_semantic": dict(Counter(r["semantic_mode"] for r in mixed_records)),
        "output": args.output,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"mixed_total={len(mixed_records)} "
        f"behavior={report['mixed_behavior']} "
        f"scope={report['mixed_scope']}"
    )
    print(f"Saved: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
