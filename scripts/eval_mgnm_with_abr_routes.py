"""Evaluate MGNM candidate ranking using ABR route predictions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs  # noqa: E402
from neg_blindness.evaluation import aggregate_results, evaluate_record  # noqa: E402
from neg_blindness.io_utils import load_records, write_json  # noqa: E402


PREFIX_MAP = {
    "SUPPRESS": "[SUPPRESS]",
    "PRESERVE": "[PRESERVE]",
    "SELECT": "",
    "PARSE_ERROR": "",
}

GOLD_MAP = {
    "suppress_target": "SUPPRESS",
    "preserve_positive": "PRESERVE",
    "select_gold_neg": "SELECT",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean_metric(summary: dict[str, Any], key: str) -> float | None:
    row = summary.get(key)
    if isinstance(row, dict):
        return float(row.get("mean", 0.0))
    if row is None:
        return None
    return float(row)


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}"


def route_prefixes(routes_path: str) -> tuple[dict[str, str], dict[str, str]]:
    routes = read_jsonl(Path(routes_path))
    prefixes: dict[str, str] = {}
    labels: dict[str, str] = {}
    for row in routes:
        pred = str(row.get("pred_behavior", "PARSE_ERROR")).upper()
        prefixes[str(row["id"])] = PREFIX_MAP.get(pred, "")
        labels[str(row["id"])] = pred if pred in PREFIX_MAP else "PARSE_ERROR"
    return prefixes, labels


def grouped_accuracy(records: list[Any], per_record: list[dict[str, Any]], route_labels: dict[str, str]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in per_record}
    grouped: dict[str, list[bool]] = defaultdict(list)
    route_correct: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        gold_behavior = GOLD_MAP.get(record.expected_neg_behavior, record.expected_neg_behavior)
        result = by_id[record.id]
        grouped[gold_behavior].append(bool(result["neg_correct"]))
        route_correct[gold_behavior].append(route_labels.get(record.id) == gold_behavior)
    return {
        behavior: {
            "n": len(values),
            "neg_correct_accuracy": sum(values) / len(values) if values else 0.0,
            "route_accuracy": sum(route_correct[behavior]) / len(route_correct[behavior])
            if route_correct[behavior] else 0.0,
        }
        for behavior, values in grouped.items()
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    configs = load_model_configs(args.models)
    records = load_records(args.input)
    prefixes, route_labels = route_prefixes(args.routes)
    missing = sorted({record.id for record in records} - set(prefixes))
    if missing and not args.allow_missing_as_empty:
        raise SystemExit(f"ABR routes missing {len(missing)} ids, examples: {missing[:10]}")

    selected = args.model_names or [
        name for name, config in configs.items()
        if config.mode in {"logprob", "choice", "chat_json", "hf_local"}
        and name not in {"generator", "verifier"}
    ]
    output: dict[str, Any] = {}
    for model_name in selected:
        config = configs[model_name]
        cache = ScoreCache(args.cache_dir, model_name)
        per_record = [
            evaluate_record(
                record,
                config,
                cache=cache,
                neg_prefix=prefixes.get(record.id, ""),
                use_multi_answer_negatives=args.use_multi_answer_negatives,
            )
            for record in records
        ]
        summary = aggregate_results(per_record)
        output[model_name] = {
            "summary": summary,
            "grouped_accuracy": grouped_accuracy(records, per_record, route_labels),
            "route_counts": dict(Counter(route_labels.get(record.id, "MISSING") for record in records)),
            "per_record": per_record,
        }
    return output


def write_downstream_summary(output_dir: Path, args: argparse.Namespace) -> None:
    rows = [
        ("Oracle token", Path(args.oracle_summary) if args.oracle_summary else None),
        ("No token", Path(args.no_token_summary) if args.no_token_summary else None),
        ("Two-stage + guard", Path(args.two_stage_summary) if args.two_stage_summary else None),
        ("ABR prompt-only", output_dir / "e4_downstream_prompt_only.json"),
        ("ABR prompt+candidates", output_dir / "e4_downstream_prompt_candidates.json"),
    ]

    expected_count: int | None = None

    def first_summary(path: Path | None) -> dict[str, Any] | None:
        if not path or not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return None
        first = next(iter(data.values()))
        summary = first.get("summary")
        if (
            expected_count is not None
            and isinstance(summary, dict)
            and int(summary.get("count", -1)) != expected_count
        ):
            return None
        return summary

    for abr_path in [
        output_dir / "e4_downstream_prompt_only.json",
        output_dir / "e4_downstream_prompt_candidates.json",
    ]:
        abr_summary = first_summary(abr_path)
        if abr_summary:
            expected_count = int(abr_summary.get("count", 0))
            break

    lines = [
        "# E4 Downstream Summary",
        "",
        "| Setting | NegFlipAcc | ScopeCtrl | OverNeg | NegRank |",
        "| ------- | ---------: | --------: | ------: | ------: |",
    ]
    for name, path in rows:
        summary = first_summary(path)
        if not summary:
            lines.append(f"| {name} | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {name} | {pct(mean_metric(summary, 'FlipAcc'))} | "
            f"{pct(mean_metric(summary, 'ScopeControlAcc'))} | "
            f"{pct(mean_metric(summary, 'OverNegationRate'))} | "
            f"{pct(mean_metric(summary, 'NegRankAcc'))} |"
        )
    lines.append("")
    lines.append(
        "Note: baseline rows are populated only from explicit existing summary paths; "
        "ABR rows are generated by this pre-experiment."
    )
    lines.append("")
    (output_dir / "e4_downstream_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_abr_router_preexp")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument("--allow-missing-as-empty", action="store_true")
    parser.add_argument("--use-multi-answer-negatives", action="store_true")
    parser.add_argument("--oracle-summary", default="")
    parser.add_argument("--no-token-summary", default="")
    parser.add_argument("--two-stage-summary", default="")
    args = parser.parse_args()

    output = evaluate(args)
    write_json(args.output, output)
    write_downstream_summary(Path(args.output).parent, args)

    for model_name, result in output.items():
        summary = result["summary"]
        print(
            f"{model_name}: FlipAcc={pct(mean_metric(summary, 'FlipAcc'))} "
            f"ScopeCtrl={pct(mean_metric(summary, 'ScopeControlAcc'))} "
            f"OverNeg={pct(mean_metric(summary, 'OverNegationRate'))} "
            f"NegRank={pct(mean_metric(summary, 'NegRankAcc'))}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
