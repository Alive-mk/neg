from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORRECT_LABELS = {"correct", "pass", "passed", "ok", "true", "1", "yes"}
WRONG_LABELS = {"wrong", "fail", "failed", "false", "0", "no"}
KEYWORD_PASS = {"pass", "passed", "correct", "ok", "true", "1", "yes"}
KEYWORD_FAIL = {"fail", "failed", "wrong", "false", "0", "no"}


def normalize_binary(value: str, *, field_name: str) -> bool | None:
    text = value.strip().lower()
    if not text:
        return None
    if text in CORRECT_LABELS or text in KEYWORD_PASS:
        return True
    if text in WRONG_LABELS or text in KEYWORD_FAIL:
        return False
    raise ValueError(f"Unsupported {field_name} value: {value!r}")


def pct(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled = []
    pending = []
    keyword_true_total = 0

    for row in rows:
        keyword_value = normalize_binary(row.get("keyword_result", ""), field_name="keyword_result")
        if keyword_value is True:
            keyword_true_total += 1

        human_value = normalize_binary(row.get("human_label", ""), field_name="human_label")
        if human_value is None:
            pending.append(row)
            continue
        if keyword_value is None:
            raise ValueError(f"Missing keyword_result for labeled row id={row.get('id')}")
        labeled.append((row, keyword_value, human_value))

    human_correct = sum(1 for _, _, human_value in labeled if human_value)
    keyword_correct_labeled = sum(1 for _, keyword_value, _ in labeled if keyword_value)
    agreement = sum(
        1 for _, keyword_value, human_value in labeled if keyword_value == human_value
    )
    false_positive = sum(
        1 for _, keyword_value, human_value in labeled
        if keyword_value and not human_value
    )
    false_negative = sum(
        1 for _, keyword_value, human_value in labeled
        if (not keyword_value) and human_value
    )
    true_positive = sum(
        1 for _, keyword_value, human_value in labeled
        if keyword_value and human_value
    )
    true_negative = sum(
        1 for _, keyword_value, human_value in labeled
        if (not keyword_value) and (not human_value)
    )

    return {
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "pending_rows": len(pending),
        "keyword_metric_all_rows": pct(keyword_true_total, len(rows)),
        "keyword_metric_labeled_rows": pct(keyword_correct_labeled, len(labeled)),
        "human_audit": pct(human_correct, len(labeled)),
        "agreement": pct(agreement, len(labeled)),
        "confusion": {
            "keyword_true_human_true": true_positive,
            "keyword_true_human_false": false_positive,
            "keyword_false_human_true": false_negative,
            "keyword_false_human_false": true_negative,
        },
        "error_type_counts": dict(Counter(
            row.get("error_type", "").strip() or "unfilled"
            for row, _, _ in labeled
        )),
        "pending_ids": [row.get("id", "") for row in pending[:20]],
    }


def summarize_by_setting(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_setting: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_setting[row.get("setting", "")].append(row)
    output = {
        setting: summarize_rows(setting_rows)
        for setting, setting_rows in sorted(by_setting.items())
    }
    output["overall"] = summarize_rows(rows)
    return output


def format_percent(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| Setting | Rows | Labeled | Pending | Keyword metric | Human audit | Agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting, stats in summary.items():
        label = "Overall" if setting == "overall" else setting
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(stats["rows"]),
                    str(stats["labeled_rows"]),
                    str(stats["pending_rows"]),
                    format_percent(stats["keyword_metric_all_rows"]),
                    format_percent(stats["human_audit"]),
                    format_percent(stats["agreement"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Exit successfully even if human_label is still blank for some rows.",
    )
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    summary = summarize_by_setting(rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(
            {
                "input": args.input,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.output_md:
        write_markdown(Path(args.output_md), summary)

    overall = summary["overall"]
    print(
        "Overall: "
        f"rows={overall['rows']} labeled={overall['labeled_rows']} "
        f"pending={overall['pending_rows']} "
        f"keyword={format_percent(overall['keyword_metric_all_rows'])} "
        f"human={format_percent(overall['human_audit'])} "
        f"agreement={format_percent(overall['agreement'])}"
    )
    for setting, stats in summary.items():
        if setting == "overall":
            continue
        print(
            f"{setting}: rows={stats['rows']} labeled={stats['labeled_rows']} "
            f"pending={stats['pending_rows']} "
            f"keyword={format_percent(stats['keyword_metric_all_rows'])} "
            f"human={format_percent(stats['human_audit'])} "
            f"agreement={format_percent(stats['agreement'])}"
        )
    print(f"Saved: {args.output_json}")
    if args.output_md:
        print(f"Saved: {args.output_md}")

    if overall["pending_rows"] and not args.allow_pending:
        raise SystemExit(
            f"{overall['pending_rows']} rows still need human_label; "
            "rerun with --allow-pending to write a pending report."
        )


if __name__ == "__main__":
    main()
