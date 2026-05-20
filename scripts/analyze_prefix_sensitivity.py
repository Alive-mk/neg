from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import build_candidate_sets, score_candidates
from neg_blindness.io_utils import load_records


PREFIXES = {
    "empty": "",
    "preserve": "[PRESERVE]",
    "suppress": "[SUPPRESS]",
    "select": "[SELECT]",
}


def best(scores: dict[str, float]) -> tuple[str, float]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[0]


def margin_for_gold(scores: dict[str, float], gold_items: set[str]) -> float | None:
    gold_scores = [score for item, score in scores.items() if item in gold_items]
    other_scores = [score for item, score in scores.items() if item not in gold_items]
    if not gold_scores or not other_scores:
        return None
    return max(gold_scores) - max(other_scores)


def analyze_record(record: Any, config: Any, cache: ScoreCache) -> dict[str, Any]:
    _, neg_candidates = build_candidate_sets(record)
    rows: dict[str, Any] = {}
    gold_pos = set(record.gold_pos)
    gold_neg = set(record.gold_neg)
    for name, prefix in PREFIXES.items():
        prompt = f"{prefix} {record.prompt_neg}" if prefix else record.prompt_neg
        scores = score_candidates(config, prompt, neg_candidates, cache)
        neg_best, neg_best_score = best(scores)
        rows[name] = {
            "neg_best": neg_best,
            "neg_best_score": neg_best_score,
            "preserve_positive": neg_best in gold_pos,
            "neg_rank_correct": neg_best in gold_neg,
            "gold_pos_margin": margin_for_gold(scores, gold_pos),
            "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        }
    return {
        "id": record.id,
        "expected_neg_behavior": record.expected_neg_behavior,
        "semantic_mode": record.semantic_mode,
        "scope_type": record.scope_type,
        "prompt_neg": record.prompt_neg,
        "gold_pos": record.gold_pos,
        "gold_neg": record.gold_neg,
        "candidate_pool_neg": record.candidate_pool_neg,
        "prefixes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", default="qwen2_5_7b_e4")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ids-json", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_prefix_sensitivity")
    args = parser.parse_args()

    configs = load_model_configs(args.models)
    config = configs[args.model_name]
    records = load_records(args.input)
    if args.ids_json:
        keep_ids = set(json.loads(Path(args.ids_json).read_text(encoding="utf-8")))
        records = [record for record in records if record.id in keep_ids]

    cache = ScoreCache(args.cache_dir, args.model_name)
    rows = [analyze_record(record, config, cache) for record in records]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        compact = {
            name: {
                "best": payload["neg_best"],
                "preserve": payload["preserve_positive"],
                "margin": payload["gold_pos_margin"],
            }
            for name, payload in row["prefixes"].items()
        }
        print(row["id"], json.dumps(compact, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
