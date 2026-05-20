"""
Audit the current high-risk results called out in progress/issues.

The script is intentionally read-only: it summarizes router prediction errors and
preserve free-generation failure modes so the paper can discuss the actual cause
instead of relying on qualitative guesses.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_MAP = {
    "suppress_target": "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg": "[SELECT]",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def summarize_router(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_type: dict[str, dict[str, Any]] = {}
    pred_dist = Counter()
    correct = 0

    for record in records:
        true_behavior = record["expected_neg_behavior"]
        true_token = TOKEN_MAP[true_behavior]
        pred_token = predictions.get(record["id"], "")
        pred_dist[pred_token] += 1
        confusion[true_behavior][pred_token] += 1
        correct += int(pred_token == true_token)

    for behavior, token in TOKEN_MAP.items():
        subset = [r for r in records if r["expected_neg_behavior"] == behavior]
        n_correct = sum(predictions.get(r["id"]) == token for r in subset)
        per_type[behavior] = {
            "correct": n_correct,
            "total": len(subset),
            "accuracy": pct(n_correct, len(subset)),
        }

    return {
        "overall": {
            "correct": correct,
            "total": len(records),
            "accuracy": pct(correct, len(records)),
        },
        "per_type": per_type,
        "prediction_distribution": dict(pred_dist),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def freegen_failure_bucket(response: str) -> str:
    text = response.strip()
    low = text.lower()
    if text.startswith("[") or "[PRESERVE]" in text:
        return "control_token_echo"
    if low.startswith("you are a helpful assistant"):
        return "chat_template_echo"
    if len(re.findall(r"\b\w+\b", text)) < 6:
        return "too_short_or_empty"
    words = re.findall(r"[a-zA-Z]{4,}", low)
    if len(words) >= 8:
        # Repetition catches degenerative greedy loops such as "consistent bedtime" repeated.
        top_count = Counter(words).most_common(1)[0][1]
        if top_count / len(words) >= 0.25:
            return "repetition_loop"
    return "wrong_or_missing_gold"


def summarize_preserve_freegen(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    failures = [item for item in results if not item.get("passed")]
    buckets = Counter(freegen_failure_bucket(item.get("response", "")) for item in failures)
    return {
        "n_records": payload.get("n_records", len(results)),
        "n_pass": payload.get("n_pass", len(results) - len(failures)),
        "accuracy": 100.0 * payload.get("preserve_flip_acc", 0.0),
        "n_fail": len(failures),
        "failure_buckets": dict(buckets),
        "failure_examples": [
            {
                "id": item.get("id"),
                "bucket": freegen_failure_bucket(item.get("response", "")),
                "gold_pos": item.get("gold_pos", [])[:1],
                "response": item.get("response", "")[:180],
            }
            for item in failures[:8]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/processed/validated_largetest_v2.jsonl")
    parser.add_argument("--router-predictions", default="outputs/llm_router_v3_1_predictions.json")
    parser.add_argument("--preserve-freegen", default="outputs/eval_freegen_preserve_oracle.json")
    parser.add_argument("--output", default="outputs/current_risk_audit.json")
    args = parser.parse_args()

    records = load_jsonl(Path(args.records))
    predictions = json.loads(Path(args.router_predictions).read_text(encoding="utf-8"))
    audit = {
        "router": summarize_router(records, predictions),
        "preserve_freegen": summarize_preserve_freegen(Path(args.preserve_freegen)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    router = audit["router"]
    print(
        "Router accuracy: "
        f"{router['overall']['correct']}/{router['overall']['total']} "
        f"= {router['overall']['accuracy']:.1f}%"
    )
    for behavior, stats in router["per_type"].items():
        print(f"  {behavior}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1f}%")
    print("Router confusion:", router["confusion"])

    freegen = audit["preserve_freegen"]
    print(
        "Preserve free-gen: "
        f"{freegen['n_pass']}/{freegen['n_records']} = {freegen['accuracy']:.1f}%"
    )
    print("Preserve failure buckets:", freegen["failure_buckets"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
