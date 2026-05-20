"""Evaluate models on BoolQ negation subset via log-probability ranking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ModelConfig, ScoreCache, get_local_runner, _LOCAL_MODEL_RUNNERS
from neg_blindness.metrics import bootstrap_ci


def load_boolq(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_record(runner, prompt: str, candidates: list[str], cache: ScoreCache) -> dict[str, float]:
    scores = {}
    for cand in candidates:
        cache_key = f"{hash(prompt)}_{cand}"
        cached = cache.get(cache_key)
        if cached is not None:
            scores[cand] = float(cached)
        else:
            s = runner.score_continuation(prompt, " " + cand)
            cache.set(cache_key, s)
            scores[cand] = s
    return scores


def evaluate_model(
    model_name: str,
    config: ModelConfig,
    records: list[dict],
    cache_dir: str,
) -> dict:
    print(f"\n  Evaluating {model_name} ({len(records)} records)...", flush=True)
    runner = get_local_runner(config)
    cache = ScoreCache(cache_dir, f"boolq_{model_name}")

    results = []
    for i, rec in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(records)}", flush=True)

        scores = score_record(runner, rec["prompt"], rec["candidates"], cache)
        predicted = max(scores, key=scores.__getitem__)
        correct = predicted == rec["gold_answer"]
        results.append({
            "id": rec["id"],
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "predicted": predicted,
            "correct": correct,
            "scores": scores,
        })

    acc_values = [r["correct"] for r in results]
    summary = {
        "count": len(results),
        "accuracy": bootstrap_ci(acc_values),
    }
    print(f"    Accuracy: {summary['accuracy']['mean']*100:.1f}%", flush=True)
    return {"summary": summary, "per_record": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True, help="JSON config with model list")
    parser.add_argument("--input", default="data/boolq_negation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_boolq")
    parser.add_argument("--model-names", nargs="*")
    args = parser.parse_args()

    with open(args.models_config, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    all_configs = {m["name"]: ModelConfig.from_dict(m) for m in payload["models"]}

    records = load_boolq(args.input)
    print(f"Loaded {len(records)} BoolQ negation records")

    names = args.model_names or [
        n for n, c in all_configs.items()
        if c.mode == "hf_local" and n not in {"generator", "verifier"}
    ]

    output: dict = {}
    for name in names:
        if name not in all_configs:
            print(f"  WARNING: {name} not in config, skipping")
            continue
        output[name] = evaluate_model(name, all_configs[name], records, args.cache_dir)

        # Free GPU memory before loading the next model
        if name in _LOCAL_MODEL_RUNNERS:
            del _LOCAL_MODEL_RUNNERS[name]
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
