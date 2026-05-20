"""
Recompute aggregate metrics on the record ids covered by a reference output.

This is useful when older ablation files cover only a subset of the current
clean audit set and need a fair comparison against mainline outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.evaluation import aggregate_results  # noqa: E402


def load_clean_records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        record["id"]: record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def output_record_ids(path: Path, model_name: str | None) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_name is None:
        if len(payload) != 1:
            raise ValueError(f"{path} has multiple models; pass --reference-model")
        model_name = next(iter(payload))
    return {record["id"] for record in payload[model_name].get("per_record", [])}


def normalize_item(item: dict[str, Any], source_record: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out.setdefault("semantic_mode", source_record.get("semantic_mode", ""))
    out.setdefault("scope_type", source_record.get("scope_type", ""))
    if source_record.get("expected_neg_behavior") == "select_gold_neg":
        valid_set = set(source_record.get("gold_neg", []) + source_record.get("valid_negatives", []))
        out["multi_neg_rank_correct"] = out.get("neg_best") in valid_set
        out["select_ambiguous"] = len(valid_set) > len(set(source_record.get("gold_neg", [])))
    if "distractor_reject" not in out:
        out["distractor_reject"] = bool(out.get("pos_correct")) and bool(out.get("neg_correct"))
    return out


def metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, dict):
        return 100.0 * float(value.get("mean", 0.0))
    return 100.0 * float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", required=True)
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_records = load_clean_records(Path(args.clean_input))
    keep_ids = output_record_ids(Path(args.reference_output), args.reference_model)
    reagg: dict[str, Any] = {}

    for output_path in args.outputs:
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
        for model_name, result in payload.items():
            per_record = [
                normalize_item(r, clean_records[r["id"]])
                for r in result.get("per_record", [])
                if r.get("id") in keep_ids and r.get("id") in clean_records
            ]
            summary = aggregate_results(per_record)
            key = f"{Path(output_path).name}:{model_name}"
            reagg[key] = {
                "source": output_path,
                "model_name": model_name,
                "n_records": len(per_record),
                "reference_output": args.reference_output,
                "summary": summary,
            }
            print(
                f"{key} "
                f"n={len(per_record)} "
                f"FlipAcc={metric(summary, 'FlipAcc'):.1f} "
                f"ScopeCtrl={metric(summary, 'ScopeControlAcc'):.1f} "
                f"NegRank={metric(summary, 'NegRankAcc'):.1f} "
                f"MultiNegRank={metric(summary, 'MultiAnswerNegRankAcc'):.1f} "
                f"NegSupp={metric(summary, 'NegSuppRate'):.1f} "
                f"OverNeg={metric(summary, 'OverNegationRate'):.1f}"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reagg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
