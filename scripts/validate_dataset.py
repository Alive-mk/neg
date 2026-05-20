from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, chat_json_request, load_model_configs
from neg_blindness.io_utils import dump_records, read_jsonl, write_json
from neg_blindness.prompts import VERIFIER_SYSTEM_PROMPT, verifier_user_prompt
from neg_blindness.schema import ExperimentRecord
from neg_blindness.validators import (
    dedupe_records,
    ensure_unique_record_ids,
    validate_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--models")
    parser.add_argument("--verifier-name")
    parser.add_argument("--verifier-threshold", type=float, default=0.7)
    args = parser.parse_args()

    raw_items = read_jsonl(args.input)
    valid_records: list[ExperimentRecord] = []
    invalid_items: list[dict] = []
    error_counter: Counter[str] = Counter()

    for item in raw_items:
        try:
            record = ExperimentRecord.from_dict(item)
        except Exception as exc:  # noqa: BLE001
            invalid_items.append({"item": item, "errors": [f"parse_error: {exc}"]})
            error_counter["parse_error"] += 1
            continue

        result = validate_record(record)
        if result.ok:
            valid_records.append(record)
            continue
        invalid_items.append({"id": record.id, "errors": result.errors})
        for error in result.errors:
            error_counter[error] += 1

    deduped_records, dedupe_report = dedupe_records(valid_records)
    unique_id_records, unique_id_report = ensure_unique_record_ids(deduped_records)

    verifier_report = {"enabled": False, "rejected": 0}
    if args.models and args.verifier_name:
        configs = load_model_configs(args.models)
        verifier = configs[args.verifier_name]
        cache = ScoreCache("outputs/verifier_cache", args.verifier_name)
        kept_records: list[ExperimentRecord] = []
        verifier_rejections: list[dict] = []
        api_errors = 0
        for record in unique_id_records:
            try:
                result = chat_json_request(
                    config=verifier,
                    system_prompt=VERIFIER_SYSTEM_PROMPT,
                    user_prompt=verifier_user_prompt(record.as_dict()),
                    cache=cache,
                )
            except Exception as exc:
                print(f"[verifier] API error for {record.id}, keeping record: {exc}")
                api_errors += 1
                kept_records.append(record)
                continue
            passed = bool(result.get("pass"))
            score = float(result.get("score", 0.0))
            if passed and score >= args.verifier_threshold:
                kept_records.append(record)
                continue
            verifier_rejections.append(
                {
                    "id": record.id,
                    "score": score,
                    "reasons": result.get("reasons", []),
                }
            )
        if api_errors:
            print(f"[verifier] {api_errors} items kept due to API errors (not rejected)")
        verifier_report = {
            "enabled": True,
            "kept": len(kept_records),
            "rejected": len(verifier_rejections),
            "sample_rejections": verifier_rejections[:100],
        }
        unique_id_records = kept_records

    dump_records(args.output, unique_id_records)
    write_json(
        args.report,
        {
            "input_rows": len(raw_items),
            "rule_valid_rows": len(valid_records),
            "final_rows": len(unique_id_records),
            "rule_error_counts": dict(error_counter),
            "invalid_samples": invalid_items[:100],
            "dedupe_report": dedupe_report,
            "unique_id_report": unique_id_report,
            "verifier_report": verifier_report,
        },
    )


if __name__ == "__main__":
    main()
