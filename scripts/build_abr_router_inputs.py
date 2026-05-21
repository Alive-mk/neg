"""Build Agentic Behavior Router input files for E4 and WikiFact.

The gold behavior is kept only for offline evaluation. Router prompts generated
by run_abr_agent_router.py must not include it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import load_records, write_json, write_jsonl  # noqa: E402
from neg_blindness.evaluation import build_candidate_sets  # noqa: E402


BEHAVIOR_MAP = {
    "suppress_target": "SUPPRESS",
    "preserve_positive": "PRESERVE",
    "select_gold_neg": "SELECT",
}


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def candidate_pool(record: Any) -> list[str]:
    _, neg_candidates = build_candidate_sets(record)
    return unique(neg_candidates)


def e4_rows(input_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in load_records(input_path):
        if record.expected_neg_behavior not in BEHAVIOR_MAP:
            raise ValueError(
                f"Unsupported expected_neg_behavior for {record.id}: "
                f"{record.expected_neg_behavior}"
            )
        rows.append(
            {
                "id": record.id,
                "prompt": record.prompt_neg,
                "candidate_pool": candidate_pool(record),
                "gold_behavior": BEHAVIOR_MAP[record.expected_neg_behavior],
                "source_behavior": record.expected_neg_behavior,
            }
        )
    return rows


def wikifact_rows(input_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in load_records(input_path):
        rows.append(
            {
                "id": record.id,
                "prompt": record.prompt_neg,
                "candidate_pool": candidate_pool(record),
                "gold_behavior": "SUPPRESS",
                "source_behavior": record.expected_neg_behavior or "suppress_target",
            }
        )
    return rows


def count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "gold_behavior_counts": dict(Counter(row["gold_behavior"] for row in rows)),
        "source_behavior_counts": dict(Counter(row.get("source_behavior", "") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e4-input",
        default="data/processed/validated_largetest_v2_clean_strict.jsonl",
    )
    parser.add_argument(
        "--wikifact-input",
        default="data/external/wikifact_neg_patched.jsonl",
    )
    parser.add_argument("--output-dir", default="outputs/abr_router_preexp")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e4 = e4_rows(args.e4_input)
    wikifact = wikifact_rows(args.wikifact_input)

    write_jsonl(out_dir / "e4_abr_prompt_only.jsonl", e4)
    write_jsonl(out_dir / "e4_abr_prompt_candidates.jsonl", e4)
    write_jsonl(out_dir / "wikifact_abr_inputs.jsonl", wikifact)
    write_json(
        out_dir / "input_summary.json",
        {
            "e4_input": args.e4_input,
            "wikifact_input": args.wikifact_input,
            "e4": count_summary(e4),
            "wikifact": count_summary(wikifact),
            "notes": {
                "gold_behavior": "offline evaluation only; excluded from ABR prompts",
                "select_prefix": "empty prefix in downstream evaluation",
            },
        },
    )

    print(f"E4 rows: {len(e4)} {dict(Counter(row['gold_behavior'] for row in e4))}")
    print(f"WikiFact rows: {len(wikifact)}")
    print(f"Saved inputs under {out_dir}")


if __name__ == "__main__":
    main()
