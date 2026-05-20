from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


FIELDS = [
    "setting",
    "id",
    "prompt",
    "reference",
    "model_output",
    "keyword_result",
    "keyword_reason",
    "human_label",
    "error_type",
    "notes",
]


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def as_text(items: Any) -> str:
    if isinstance(items, list):
        return " || ".join(clean_cell(str(item)) for item in items)
    return clean_cell(str(items or ""))


def clean_cell(value: str) -> str:
    return value.replace("\r", "\\n").replace("\n", "\\n").replace("\t", " ")


def keyword_label(value: Any) -> str:
    return "pass" if bool(value) else "fail"


def row_from_result(setting: str, result: dict[str, Any], reference_field: str) -> dict[str, str]:
    return {
        "setting": setting,
        "id": clean_cell(str(result.get("id", ""))),
        "prompt": clean_cell(str(result.get("prompt_neg", ""))),
        "reference": as_text(result.get(reference_field, [])),
        "model_output": clean_cell(str(result.get("response", ""))),
        "keyword_result": keyword_label(result.get("passed")),
        "keyword_reason": clean_cell(str(result.get("failure_reason") or result.get("reason") or "")),
        "human_label": "",
        "error_type": "",
        "notes": "",
    }


def choose_rows(
    rows: list[dict[str, str]],
    n: int,
    selection: str,
    rng: random.Random,
) -> list[dict[str, str]]:
    if selection == "first":
        return rows[:n]
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:n]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-input", required=True)
    parser.add_argument("--preserve-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-setting", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=["first", "random"], default="first")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    suppress_rows = [
        row_from_result("suppress_oracle", result, "forbidden_neg")
        for result in load_results(Path(args.suppress_input))
    ]
    preserve_rows = [
        row_from_result("preserve_notoken", result, "gold_pos")
        for result in load_results(Path(args.preserve_input))
    ]

    rows = (
        choose_rows(suppress_rows, args.per_setting, args.selection, rng)
        + choose_rows(preserve_rows, args.per_setting, args.selection, rng)
    )
    write_tsv(Path(args.output), rows)

    print(f"Suppress rows: {min(args.per_setting, len(suppress_rows))}/{len(suppress_rows)}")
    print(f"Preserve rows: {min(args.per_setting, len(preserve_rows))}/{len(preserve_rows)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
