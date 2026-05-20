"""
Prefix-gap policy for conservative PRESERVE rescue.

For v3 SELECT + v4 PRESERVE + narrow out-of-scope instructional patterns, do not
blindly add [PRESERVE]. First score candidates with the empty prefix. If the
empty-prefix top candidate has a large top1-top2 gap, keep empty; otherwise add
[PRESERVE] because the no-prefix choice is unstable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import build_candidate_sets, score_candidates
from neg_blindness.io_utils import load_records


OPEN_IF_RE = re.compile(
    r"^\s*(explain|describe|list|outline|provide)\b.*\bif\b.*\b(?:not|do not|does not|don't|is not|are not)\b",
    re.I,
)
WITHOUT_METHOD_RE = re.compile(
    r"\bwithout using\b.*\b(standard|wizard|method|approach|technique)\b",
    re.I,
)


def load_json(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def should_consider(record: Any, base_token: str, v4_token: str) -> bool:
    if base_token != "[SELECT]" or v4_token != "[PRESERVE]":
        return False
    text = str(record.prompt_neg)
    return bool(OPEN_IF_RE.search(text) or WITHOUT_METHOD_RE.search(text))


def top_gap(scores: dict[str, float]) -> float:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return float("inf")
    return ranked[0][1] - ranked[1][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", default="qwen2_5_7b_e4")
    parser.add_argument("--records", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--v4", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_prefix_gap_policy")
    parser.add_argument("--gap-threshold", type=float, default=0.20)
    args = parser.parse_args()

    configs = load_model_configs(args.models)
    config = configs[args.model_name]
    records = load_records(args.records)
    base = load_json(Path(args.base))
    v4 = load_json(Path(args.v4))
    cache = ScoreCache(args.cache_dir, args.model_name)

    predictions: dict[str, str] = {}
    details: dict[str, dict[str, Any]] = {}
    for record in records:
        base_token = base[record.id]
        v4_token = v4[record.id]
        if not should_consider(record, base_token, v4_token):
            predictions[record.id] = "" if base_token == "[SELECT]" else base_token
            details[record.id] = {"route": "base", "base": base_token, "v4": v4_token}
            continue

        _, neg_candidates = build_candidate_sets(record)
        empty_scores = score_candidates(config, record.prompt_neg, neg_candidates, cache)
        gap = top_gap(empty_scores)
        ranked = sorted(empty_scores.items(), key=lambda item: item[1], reverse=True)
        if gap <= args.gap_threshold:
            predictions[record.id] = "[PRESERVE]"
            route = "low_gap_preserve"
        else:
            predictions[record.id] = ""
            route = "high_gap_empty"
        details[record.id] = {
            "route": route,
            "base": base_token,
            "v4": v4_token,
            "empty_top_gap": gap,
            "empty_top": ranked[0][0] if ranked else "",
            "empty_second": ranked[1][0] if len(ranked) > 1 else "",
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    considered = [row for row in details.values() if row["route"] in {"low_gap_preserve", "high_gap_empty"}]
    report = {
        "records": args.records,
        "base": args.base,
        "v4": args.v4,
        "gap_threshold": args.gap_threshold,
        "considered": len(considered),
        "routes": {
            "low_gap_preserve": sum(1 for row in considered if row["route"] == "low_gap_preserve"),
            "high_gap_empty": sum(1 for row in considered if row["route"] == "high_gap_empty"),
        },
        "details": details,
        "output": str(out_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["gap_threshold", "considered", "routes", "output"]}, indent=2))


if __name__ == "__main__":
    main()
