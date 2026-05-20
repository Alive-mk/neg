"""Evaluate BoolQ with MGNM behavior token prefix.

For BoolQ yes/no questions, [PRESERVE] signals to the model:
"this is a preserve_positive context — don't suppress the factual answer."
Tests whether the behavior token restores calibration for MGNM models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ModelConfig, ScoreCache, get_local_runner, _LOCAL_MODEL_RUNNERS
from neg_blindness.metrics import bootstrap_ci

import torch

CANDIDATES = ["Yes", "No"]


def evaluate_with_prefix(
    model_name: str,
    config: ModelConfig,
    records: list[dict],
    cache_dir: str,
    prefix: str,
) -> dict:
    runner = get_local_runner(config)
    cache_name = f"boolq_{model_name}_token{prefix.strip().replace('[','').replace(']','')}"
    cache = ScoreCache(cache_dir, cache_name)

    results = []
    for rec in records:
        prompt = f"{prefix} {rec['prompt']}" if prefix else rec["prompt"]
        scores = {}
        for cand in CANDIDATES:
            key = f"{hash(prompt)}_{cand}"
            cached = cache.get(key)
            if cached is not None:
                scores[cand] = float(cached)
            else:
                s = runner.score_continuation(prompt, " " + cand)
                cache.set(key, s)
                scores[cand] = s
        predicted = max(scores, key=scores.__getitem__)
        results.append({
            "id": rec["id"],
            "gold_answer": rec["gold_answer"],
            "predicted": predicted,
            "correct": predicted == rec["gold_answer"],
        })

    acc_values = [r["correct"] for r in results]
    yes_pred = sum(1 for r in results if r["predicted"] == "Yes")
    mean_acc = sum(acc_values) / len(acc_values) * 100
    print(f"  {model_name:<22} prefix={prefix!r:<14} acc={mean_acc:.1f}%  pred_Yes={yes_pred}")

    if model_name in _LOCAL_MODEL_RUNNERS:
        del _LOCAL_MODEL_RUNNERS[model_name]
    torch.cuda.empty_cache()

    return {
        "summary": {"accuracy": bootstrap_ci(acc_values)},
        "prefix": prefix,
        "per_record": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_boolq")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument("--prefix", default="[PRESERVE]",
                        help="Behavior token prefix to prepend to prompts")
    args = parser.parse_args()

    with open(args.models_config) as fh:
        payload = json.load(fh)
    all_configs = {m["name"]: ModelConfig.from_dict(m) for m in payload["models"]}

    records = []
    with open(args.input) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records | prefix={args.prefix!r}")

    names = args.model_names or [
        n for n, c in all_configs.items()
        if c.mode == "hf_local" and n not in {"generator", "verifier"}
    ]

    output = {}
    for name in names:
        if name not in all_configs:
            continue
        output[name] = evaluate_with_prefix(
            name, all_configs[name], records, args.cache_dir, args.prefix
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
