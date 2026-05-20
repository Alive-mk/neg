from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import load_model_configs
from neg_blindness.io_utils import load_records, write_json
from neg_blindness.mechanistic import (
    LocalMechanisticRunner,
    aggregate_mechanistic_results,
    analyze_record_mechanistically,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--patch-blend-weight", type=float, default=1.0)
    parser.add_argument(
        "--record-ids",
        nargs="*",
        help="Optional explicit record id filter.",
    )
    args = parser.parse_args()

    configs = load_model_configs(args.models)
    if args.model_name not in configs:
        raise ValueError(f"model {args.model_name} not found in {args.models}")
    config = configs[args.model_name]
    if config.mode != "hf_local":
        raise ValueError("mechanistic analysis currently supports hf_local models only")

    records = load_records(args.input)
    if args.record_ids:
        selected = set(args.record_ids)
        records = [record for record in records if record.id in selected]
    if args.max_records is not None:
        records = records[: args.max_records]

    runner = LocalMechanisticRunner(config)
    analyzed: list[dict] = []
    failures: list[dict] = []
    for record in records:
        try:
            analyzed.append(
                analyze_record_mechanistically(
                    record=record,
                    runner=runner,
                    patch_blend_weight=args.patch_blend_weight,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": record.id, "error": str(exc)})

    payload = {
        "model_name": args.model_name,
        "input_path": args.input,
        "num_input_records": len(records),
        "num_analyzed_records": len(analyzed),
        "num_failures": len(failures),
        "failures": failures,
        "aggregate": aggregate_mechanistic_results(analyzed),
        "per_record": analyzed,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
