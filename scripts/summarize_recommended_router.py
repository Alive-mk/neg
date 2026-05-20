from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def behavior_label(token: str) -> str:
    return "[SELECT]" if token == "" else token


def router_summary(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    correct = 0
    per_behavior: dict[str, dict[str, Any]] = {}
    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(behavior_label(predictions[r["id"]]) == token for r in subset)
        per_behavior[behavior] = {
            "correct": n_correct,
            "total": len(subset),
            "accuracy": n_correct / len(subset) if subset else 0.0,
        }
        correct += n_correct
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "per_behavior": per_behavior,
    }


def extract_downstream_summary(eval_data: dict[str, Any]) -> dict[str, float]:
    model_name, payload = next(iter(eval_data.items()))
    summary = payload["summary"]
    return {
        "model_name": model_name,
        "FlipAcc": summary["FlipAcc"]["mean"],
        "ScopeCtrl": summary["ScopeControlAcc"]["mean"],
        "NegRank": summary["NegRankAcc"]["mean"],
        "OverNeg": summary["OverNegationRate"]["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-records", required=True)
    parser.add_argument("--strict-predictions", required=True)
    parser.add_argument("--strict-eval", required=True)
    parser.add_argument("--calibration-records", required=True)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--calibration-eval", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    strict_records = load_jsonl(Path(args.strict_records))
    strict_predictions = load_json(Path(args.strict_predictions))
    strict_eval = load_json(Path(args.strict_eval))

    calibration_records = load_jsonl(Path(args.calibration_records))
    calibration_predictions = load_json(Path(args.calibration_predictions))
    calibration_eval = load_json(Path(args.calibration_eval))

    report = {
        "policy": "recommended",
        "strict": {
            "router": router_summary(strict_records, strict_predictions),
            "downstream": extract_downstream_summary(strict_eval),
        },
        "calibration": {
            "router": router_summary(calibration_records, calibration_predictions),
            "downstream": extract_downstream_summary(calibration_eval),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"strict router={report['strict']['router']['accuracy']*100:.1f}% "
        f"downstream={report['strict']['downstream']['FlipAcc']*100:.1f}/"
        f"{report['strict']['downstream']['ScopeCtrl']*100:.1f}/"
        f"{report['strict']['downstream']['NegRank']*100:.1f}"
    )
    print(
        f"calibration router={report['calibration']['router']['accuracy']*100:.1f}% "
        f"downstream={report['calibration']['downstream']['FlipAcc']*100:.1f}/"
        f"{report['calibration']['downstream']['ScopeCtrl']*100:.1f}/"
        f"{report['calibration']['downstream']['NegRank']*100:.1f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
