"""Aggregate existing evaluation JSON files into reproducible main tables.

The script is read-only with respect to experiment results: it does not rerun
models or recompute metrics. It extracts already-computed summaries from E4
clean-strict, WikiFact, BoolQ, and MMLU outputs and writes JSON/CSV/Markdown
tables from a small row-spec config.

Example config:
{
  "rows": [
    {
      "label": "MGNM (Qwen)",
      "group": "Qwen2.5-7B",
      "clean_strict": {"path": "outputs/eval_qwen_mgnm_clean.json", "key": "qwen_mgnm"},
      "wikifact": {"path": "outputs/eval_qwen_mgnm_wikifact.json", "key": "qwen_mgnm"},
      "boolq_pmi": {"path": "outputs/eval_boolq_full_pmi_qwen.json", "key": "qwen_mgnm"},
      "mmlu": {"path": "outputs/mmlu_qwen_mgnm.json"}
    }
  ]
}
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAIN_COLUMNS = [
    "group",
    "label",
    "clean_n",
    "hard_negrank",
    "multi_answer_negrank",
    "neg_flip_acc",
    "scope_ctrl",
    "over_neg",
    "neg_supp_rate",
    "wikifact_flip_acc",
    "wikifact_neg_supp_rate",
    "boolq_raw_acc",
    "boolq_pmi_acc",
    "boolq_alpha",
    "boolq_null_prior_delta",
    "mmlu_delta",
    "mmlu_n",
    "notes",
]

CLEAN_COLUMNS = [
    "group",
    "label",
    "clean_n",
    "hard_negrank",
    "multi_answer_negrank",
    "neg_flip_acc",
    "scope_ctrl",
    "over_neg",
    "neg_supp_rate",
    "notes",
]

WIKIFACT_COLUMNS = [
    "group",
    "label",
    "wikifact_flip_acc",
    "wikifact_neg_supp_rate",
    "notes",
]

BOOLQ_COLUMNS = [
    "group",
    "label",
    "boolq_raw_acc",
    "boolq_pmi_acc",
    "boolq_alpha",
    "boolq_null_prior_delta",
    "notes",
]

MMLU_COLUMNS = [
    "group",
    "label",
    "mmlu_delta",
    "mmlu_n",
    "notes",
]


def load_json(path: Path, allow_missing: bool) -> Any | None:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def select_payload(data: Any, key: str | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError("expected top-level JSON object")
    if key:
        if key not in data:
            raise KeyError(f"missing model key: {key}")
        value = data[key]
        if not isinstance(value, dict):
            raise TypeError(f"model key {key!r} is not an object")
        return value
    if any(name in data for name in ("summary", "aggregate", "summary_raw", "summary_pmi", "delta")):
        return data
    if len(data) == 1:
        value = next(iter(data.values()))
        if not isinstance(value, dict):
            raise TypeError("single top-level value is not an object")
        return value
    raise KeyError("multiple top-level model keys found; set the source key")


def source_payload(row: dict[str, Any], name: str, allow_missing: bool) -> dict[str, Any] | None:
    spec = row.get(name)
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError(f"{name} source must be a path string or object with path/key")
    data = load_json(Path(spec["path"]), allow_missing=allow_missing)
    if data is None:
        return None
    return select_payload(data, spec.get("key"))


def summary_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload.get("summary"), dict):
        return payload["summary"]
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("summary"), dict):
        return aggregate["summary"]
    if any(isinstance(payload.get(name), dict) for name in ("FlipAcc", "NegRankAcc", "accuracy")):
        return payload
    return {}


def metric_percent(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    value = float(value)
    if -1.0 <= value <= 1.0:
        value *= 100.0
    return round(value, 3)


def count_from_payload(payload: dict[str, Any] | None, summary: dict[str, Any]) -> int | None:
    if not payload:
        return None
    for value in (summary.get("count"), payload.get("n_records")):
        if value is not None:
            return int(value)
    per_record = payload.get("per_record")
    if isinstance(per_record, list):
        return len(per_record)
    return None


def boolq_accuracy(payload: dict[str, Any] | None, summary_name: str) -> float | None:
    if not payload:
        return None
    summary = payload.get(summary_name)
    if isinstance(summary, dict):
        return metric_percent(summary, "accuracy")
    if summary_name == "summary_raw":
        return metric_percent(summary_from_payload(payload), "accuracy")
    return None


def prior_delta(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    prior = payload.get("prior")
    if not isinstance(prior, dict) or "No" not in prior or "Yes" not in prior:
        return None
    return round(float(prior["No"]) - float(prior["Yes"]), 6)


def mmlu_delta(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    delta = payload.get("delta")
    if isinstance(delta, dict):
        value = delta.get("overall_accuracy")
        if value is None:
            return None
        value = float(value)
        if -1.0 <= value <= 1.0:
            value *= 100.0
        return round(value, 3)
    if isinstance(delta, (int, float)):
        value = float(delta)
        if -1.0 <= value <= 1.0:
            value *= 100.0
        return round(value, 3)
    return None


def mmlu_n(payload: dict[str, Any] | None) -> int | None:
    if not payload:
        return None
    for section_name in ("finetuned", "base"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("n_questions") is not None:
            return int(section["n_questions"])
    return None


def aggregate_row(row: dict[str, Any], allow_missing: bool) -> dict[str, Any]:
    clean_payload = source_payload(row, "clean_strict", allow_missing)
    wikifact_payload = source_payload(row, "wikifact", allow_missing)
    boolq_raw_payload = source_payload(row, "boolq_raw", allow_missing)
    boolq_pmi_payload = source_payload(row, "boolq_pmi", allow_missing)
    mmlu_payload = source_payload(row, "mmlu", allow_missing)

    clean_summary = summary_from_payload(clean_payload)
    wikifact_summary = summary_from_payload(wikifact_payload)

    if boolq_pmi_payload and not boolq_raw_payload:
        boolq_raw_payload = boolq_pmi_payload

    return {
        "group": row.get("group", ""),
        "label": row["label"],
        "clean_n": count_from_payload(clean_payload, clean_summary),
        "hard_negrank": metric_percent(clean_summary, "NegRankAcc"),
        "multi_answer_negrank": metric_percent(clean_summary, "MultiAnswerNegRankAcc"),
        "neg_flip_acc": metric_percent(clean_summary, "FlipAcc"),
        "scope_ctrl": metric_percent(clean_summary, "ScopeControlAcc"),
        "over_neg": metric_percent(clean_summary, "OverNegationRate"),
        "neg_supp_rate": metric_percent(clean_summary, "NegSuppRate"),
        "wikifact_flip_acc": metric_percent(wikifact_summary, "FlipAcc"),
        "wikifact_neg_supp_rate": metric_percent(wikifact_summary, "NegSuppRate"),
        "boolq_raw_acc": boolq_accuracy(boolq_raw_payload, "summary_raw"),
        "boolq_pmi_acc": boolq_accuracy(boolq_pmi_payload, "summary_pmi"),
        "boolq_alpha": boolq_pmi_payload.get("alpha") if boolq_pmi_payload else None,
        "boolq_null_prior_delta": prior_delta(boolq_pmi_payload),
        "mmlu_delta": mmlu_delta(mmlu_payload),
        "mmlu_n": mmlu_n(mmlu_payload),
        "notes": row.get("notes", ""),
    }


def write_json(path: Path, rows: list[dict[str, Any]], config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_config": str(config_path),
        "notes": {
            "hard_negrank": "NegRankAcc; strict single-gold SELECT metric.",
            "multi_answer_negrank": "MultiAnswerNegRankAcc; accepts audited valid_negatives when present.",
            "mmlu_delta": "Percentage-point delta from the MMLU diagnostic output.",
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("group", "Group"),
        ("label", "Method"),
        ("hard_negrank", "Hard NegRank"),
        ("multi_answer_negrank", "Multi-answer NegRank"),
        ("neg_flip_acc", "NegFlipAcc"),
        ("scope_ctrl", "ScopeCtrl"),
        ("over_neg", "OverNeg"),
        ("wikifact_flip_acc", "WikiFact"),
        ("boolq_raw_acc", "BoolQ Raw"),
        ("boolq_pmi_acc", "BoolQ PMI"),
        ("mmlu_delta", "MMLU Delta"),
    ]
    lines = [
        "# Main Table Aggregation",
        "",
        "Percent values are copied from existing evaluation summaries. Hard NegRank is `NegRankAcc`; Multi-answer NegRank is `MultiAnswerNegRankAcc`.",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_cell(row.get(key)) for key, _ in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing E4/WikiFact/BoolQ/MMLU summaries into JSON, CSV, and Markdown tables."
    )
    parser.add_argument("--config", required=True, help="JSON file with a top-level rows list.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix. The script writes *_combined.{json,csv,md} and per-benchmark CSVs.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Leave metrics blank when a configured source file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path, allow_missing=False)
    rows_spec = config.get("rows")
    if not isinstance(rows_spec, list) or not rows_spec:
        raise ValueError("config must contain a non-empty rows list")

    rows = [aggregate_row(row, allow_missing=args.allow_missing) for row in rows_spec]
    prefix = Path(args.output_prefix)
    write_json(prefix.with_name(prefix.name + "_combined.json"), rows, config_path)
    write_csv(prefix.with_name(prefix.name + "_combined.csv"), rows, MAIN_COLUMNS)
    write_markdown(prefix.with_name(prefix.name + "_combined.md"), rows)
    write_csv(prefix.with_name(prefix.name + "_clean_strict.csv"), rows, CLEAN_COLUMNS)
    write_csv(prefix.with_name(prefix.name + "_wikifact.csv"), rows, WIKIFACT_COLUMNS)
    write_csv(prefix.with_name(prefix.name + "_boolq.csv"), rows, BOOLQ_COLUMNS)
    write_csv(prefix.with_name(prefix.name + "_mmlu.csv"), rows, MMLU_COLUMNS)

    print(f"Aggregated {len(rows)} rows -> {prefix.parent}/{prefix.name}_*")


if __name__ == "__main__":
    main()
