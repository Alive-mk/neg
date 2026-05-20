from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rule_router_clean import predict_behavior


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_pattern(text: str) -> str:
    t = f" {text.lower()} "
    if " not without " in t:
        return "not_without"
    if " without " in t:
        return "without"
    if " not fail " in t or " never fail " in t:
        return "not_fail"
    if " not ignore " in t or " not neglect " in t or " not forget " in t:
        return "not_ignore_family"
    if " not incorrect " in t or " not uncommon " in t or " not untrue " in t:
        return "not_incorrect_family"
    if " should not " in t:
        return "should_not"
    if " do not " in t or " does not " in t or " is not " in t or " are not " in t:
        return "simple_not"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="data/processed/validated_largetest_v2_clean_strict.jsonl")
    parser.add_argument("--llm", default="outputs/llm_router_candidate_predictions.json")
    parser.add_argument("--supervised", default="outputs/supervised_router_tfidf_rule_predictions.json")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    records = load_jsonl(Path(args.records))
    llm_predictions = load_json(Path(args.llm))
    supervised_predictions = load_json(Path(args.supervised))

    preserve_errors: list[dict] = []
    suppress_errors: list[dict] = []
    select_errors: list[dict] = []
    pattern_counts: Counter[str] = Counter()
    triple_counts: Counter[str] = Counter()

    for record in records:
        rid = record["id"]
        gold = record["expected_neg_behavior"]
        llm = llm_predictions.get(rid, "")
        sup = supervised_predictions.get(rid, "")
        rule = predict_behavior(record)
        pattern = detect_pattern(record.get("prompt_neg", ""))
        triple = f"llm={llm}|sup={sup}|rule={rule}"
        if gold == "preserve_positive" and llm == "[SELECT]":
            pattern_counts[pattern] += 1
            triple_counts[triple] += 1
            preserve_errors.append(
                {
                    "id": rid,
                    "pattern": pattern,
                    "llm": llm,
                    "supervised": sup,
                    "rule": rule,
                    "prompt_neg": record.get("prompt_neg", ""),
                    "gold_pos": record.get("gold_pos", []),
                    "gold_neg": record.get("gold_neg", []),
                    "candidates": record.get("candidate_pool_neg", [])[:8],
                }
            )
        elif gold == "suppress_target" and llm == "[SELECT]":
            suppress_errors.append(
                {
                    "id": rid,
                    "pattern": pattern,
                    "supervised": sup,
                    "rule": rule,
                    "prompt_neg": record.get("prompt_neg", ""),
                }
            )
        elif gold == "select_gold_neg" and llm != "[SELECT]":
            select_errors.append(
                {
                    "id": rid,
                    "pattern": pattern,
                    "llm": llm,
                    "supervised": sup,
                    "rule": rule,
                    "prompt_neg": record.get("prompt_neg", ""),
                }
            )

    print("Preserve misrouted as SELECT:", len(preserve_errors))
    print("Pattern counts:", dict(pattern_counts.most_common()))
    print("LLM/SUP/RULE combinations:", dict(triple_counts.most_common()))
    print()

    print("Sample preserve->SELECT errors:")
    for row in preserve_errors[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))

    print()
    print("Suppress misrouted as SELECT:", len(suppress_errors))
    for row in suppress_errors[: min(args.limit, 20)]:
        print(json.dumps(row, ensure_ascii=False))

    print()
    print("Select not predicted as SELECT:", len(select_errors))
    for row in select_errors[: min(args.limit, 20)]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
