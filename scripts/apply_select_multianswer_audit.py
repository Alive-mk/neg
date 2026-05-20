"""Apply a reviewed select multi-answer audit to model eval outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def load_reviewed_audit(path: Path) -> dict[str, set[str]]:
    valid: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("audit_decision") == "valid_multi_answer":
                valid.setdefault(row["id"], set()).add(row["neg_best"])
    return valid


def iter_model_outputs(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        (model_name, model_payload.get("per_record", []))
        for model_name, model_payload in payload.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", required=True)
    parser.add_argument("--audit-tsv", required=True)
    parser.add_argument("--eval-output", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.clean_input))
    audited_valid = load_reviewed_audit(Path(args.audit_tsv))
    report: dict[str, dict[str, Any]] = {}

    for eval_output in args.eval_output:
        for model_name, per_record in iter_model_outputs(Path(eval_output)):
            total = hard = audited = audit_rescues = 0
            for item in per_record:
                record = records.get(item["id"])
                if not record or record.get("expected_neg_behavior") != "select_gold_neg":
                    continue
                total += 1
                is_hard = bool(item.get("neg_rank_correct"))
                is_audited_rescue = item.get("neg_best") in audited_valid.get(item["id"], set())
                hard += int(is_hard)
                audit_rescues += int((not is_hard) and is_audited_rescue)
                audited += int(is_hard or is_audited_rescue)
            report[model_name] = {
                "source": eval_output,
                "select_n": total,
                "hard_correct": hard,
                "hard_neg_rank": hard / total if total else 0.0,
                "audited_correct": audited,
                "audit_rescues": audit_rescues,
                "audited_soft_neg_rank": audited / total if total else 0.0,
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for model_name, item in report.items():
        print(
            f"{model_name}: hard={100 * item['hard_neg_rank']:.1f} "
            f"soft={100 * item['audited_soft_neg_rank']:.1f} "
            f"rescues={item['audit_rescues']}/{item['select_n']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
