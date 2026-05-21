"""Evaluate ABR route predictions and generate compact summaries."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["SUPPRESS", "PRESERVE", "SELECT"]
PRED_LABELS = ["SUPPRESS", "PRESERVE", "SELECT", "PARSE_ERROR"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = str(row.get("gold_behavior", ""))
        pred = str(row.get("pred_behavior", "PARSE_ERROR"))
        if gold in LABELS:
            confusion[gold][pred if pred in PRED_LABELS else "PARSE_ERROR"] += 1

    total = sum(sum(confusion[gold].values()) for gold in LABELS)
    correct = sum(confusion[label][label] for label in LABELS)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(count for pred, count in confusion[label].items() if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            gold: {pred: confusion[gold][pred] for pred in PRED_LABELS}
            for gold in LABELS
        },
    }


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(name: str, summary: dict[str, Any] | None) -> str:
    if not summary:
        return f"| {name} | NA | NA | NA | NA | NA |"
    per = summary.get("per_class", {})
    return (
        f"| {name} | {pct(summary.get('accuracy'))} | {pct(summary.get('macro_f1'))} | "
        f"{pct(per.get('SUPPRESS', {}).get('recall'))} | "
        f"{pct(per.get('PRESERVE', {}).get('recall'))} | "
        f"{pct(per.get('SELECT', {}).get('recall'))} |"
    )


def existing_router_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test_summary")
    if not test:
        return None
    per_behavior = test.get("per_behavior", {})

    def rec(source: str) -> float:
        row = per_behavior.get(source, {})
        return float(row.get("accuracy", 0.0))

    return {
        "accuracy": float(test.get("accuracy", 0.0)),
        "macro_f1": float(test.get("balanced_behavior_accuracy", 0.0)),
        "per_class": {
            "SUPPRESS": {"recall": rec("suppress_target")},
            "PRESERVE": {"recall": rec("preserve_positive")},
            "SELECT": {"recall": rec("select_gold_neg")},
        },
    }


def write_router_markdown(output_dir: Path) -> None:
    prompt_only = load_summary(output_dir / "e4_abr_prompt_only_summary.json")
    prompt_candidates = load_summary(output_dir / "e4_abr_prompt_candidates_summary.json")
    twostage = existing_router_summary(Path("outputs/supervised_router_twostage_rule_report.json"))
    textguard = existing_router_summary(Path("outputs/supervised_router_twostage_rule_textguard_report.json"))

    lines = [
        "# E4 ABR Router Summary",
        "",
        "| Router | Accuracy | Macro-F1 | Suppress Recall | Preserve Recall | Select Recall |",
        "| ------ | -------: | -------: | --------------: | --------------: | ------------: |",
        row_from_summary("ABR prompt-only", prompt_only),
        row_from_summary("ABR prompt+candidates", prompt_candidates),
        row_from_summary("Two-stage supervised", twostage),
        row_from_summary("Two-stage + text guard", textguard),
        "",
    ]
    (output_dir / "e4_abr_router_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_wikifact_markdown(output_dir: Path) -> None:
    prompt_candidates = load_summary(output_dir / "wikifact_abr_summary.json")
    prompt_only = load_summary(output_dir / "wikifact_abr_prompt_only_summary.json")
    two_stage = existing_router_summary(Path("outputs/supervised_router_twostage_rule_wikifact_report.json"))
    textguard = existing_router_summary(Path("outputs/supervised_router_twostage_rule_wikifact_textguard_report.json"))

    def suppress_recall(summary: dict[str, Any] | None) -> str:
        if not summary:
            return "NA"
        return pct(summary.get("per_class", {}).get("SUPPRESS", {}).get("recall"))

    lines = [
        "# WikiFact ABR Router Summary",
        "",
        "| Router | WikiFact Suppress Recall | Downstream NegFlipAcc |",
        "| ------ | -----------------------: | --------------------: |",
        f"| two-stage no guard | {suppress_recall(two_stage)} | NA |",
        f"| metadata/text guard | {suppress_recall(textguard)} | NA |",
        f"| ABR prompt-only | {suppress_recall(prompt_only)} | NA |",
        f"| ABR prompt+candidates | {suppress_recall(prompt_candidates)} | NA |",
        "",
    ]
    (output_dir / "wikifact_abr_router_summary.md").write_text("\n".join(lines), encoding="utf-8")


def error_examples(rows: list[dict[str, Any]]) -> str:
    groups = [
        ("SUPPRESS", "SELECT"),
        ("SUPPRESS", "PRESERVE"),
        ("PRESERVE", "SUPPRESS"),
        ("PRESERVE", "SELECT"),
        ("SELECT", "SUPPRESS"),
        ("SELECT", "PRESERVE"),
    ]
    reason = {
        ("SUPPRESS", "SELECT"): "可能原因：prompt 中的 not 被解释成“选择错误选项”，而不是压制被否定目标。",
        ("SUPPRESS", "PRESERVE"): "可能原因：否定词被误判为作用在辅助条件上。",
        ("PRESERVE", "SUPPRESS"): "可能原因：router 看到否定后压制了应保留的核心答案。",
        ("PRESERVE", "SELECT"): "可能原因：out-of-scope 或 double negation 被误判成 invalid-option 任务。",
        ("SELECT", "SUPPRESS"): "可能原因：候选选择任务被当作简单目标压制。",
        ("SELECT", "PRESERVE"): "可能原因：显式否定或候选文本让 router 误以为答案应保持不变。",
    }
    lines = ["# E4 ABR Error Analysis", ""]
    summary = summarize(rows)
    lines.extend(["## Confusion Matrix", ""])
    lines.append("| Gold | SUPPRESS | PRESERVE | SELECT | PARSE_ERROR |")
    lines.append("| ---- | -------: | -------: | -----: | ----------: |")
    for gold in LABELS:
        row = summary["confusion_matrix"][gold]
        lines.append(
            f"| {gold} | {row['SUPPRESS']} | {row['PRESERVE']} | {row['SELECT']} | {row['PARSE_ERROR']} |"
        )
    lines.append("")

    for gold, pred in groups:
        examples = [
            row for row in rows
            if row.get("gold_behavior") == gold and row.get("pred_behavior") == pred
        ][:10]
        lines.extend([f"## {gold} -> {pred}", "", reason[(gold, pred)], ""])
        if not examples:
            lines.append("No examples.")
            lines.append("")
            continue
        for idx, row in enumerate(examples, start=1):
            candidates = row.get("candidate_pool", [])
            short_candidates = "; ".join(str(item) for item in candidates[:5])
            lines.append(f"{idx}. `{row.get('id')}`")
            lines.append(f"   Prompt: {row.get('prompt')}")
            lines.append(f"   Candidates: {short_candidates}")
            lines.append(f"   Rationale: {row.get('rationale', '')}")
        lines.append("")
    lines.extend(
        [
            "## Checks",
            "",
            "- suppress 是否仍然被误分为 SELECT：见 `SUPPRESS -> SELECT`。",
            "- preserve 是否被误分为 SUPPRESS：见 `PRESERVE -> SUPPRESS`。",
            "- prompt+candidates 是否优于 prompt-only：见 `e4_abr_router_summary.md`。",
            "- rationale 是否解释 scope：需要对上面错误样本人工复核。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--markdown-output-dir",
        default="outputs/abr_router_preexp",
        help="Directory for aggregate markdown summaries.",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.routes))
    summary = summarize(rows)
    write_json(Path(args.output), summary)

    output_dir = Path(args.markdown_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.output).name.startswith("e4_abr_"):
        write_router_markdown(output_dir)
        if "prompt_candidates" in Path(args.output).name:
            (output_dir / "e4_abr_error_analysis.md").write_text(
                error_examples(rows),
                encoding="utf-8",
            )
    if Path(args.output).name.startswith("wikifact_abr"):
        write_wikifact_markdown(output_dir)

    print(
        f"n={summary['n']} accuracy={summary['accuracy'] * 100:.1f}% "
        f"macro_f1={summary['macro_f1'] * 100:.1f}%"
    )
    for label in LABELS:
        row = summary["per_class"][label]
        print(
            f"  {label}: P={row['precision'] * 100:.1f}% "
            f"R={row['recall'] * 100:.1f}% F1={row['f1'] * 100:.1f}% "
            f"n={row['support']}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
