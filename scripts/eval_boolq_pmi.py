"""Re-score BoolQ results with scaled PMI calibration.

Scaled PMI score: log p(answer | prompt) - alpha * log p(answer | null_prompt)
Corrects for systematic Yes/No prior bias introduced by fine-tuning.

Computes the null-prompt prior by scoring " Yes" and " No" against a
minimal context ("Answer:"), then recomputes accuracy for each model.
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


NULL_PROMPT = "Answer:"
CANDIDATES = ["Yes", "No"]


def compute_prior(runner, cache: ScoreCache) -> dict[str, float]:
    prior = {}
    for cand in CANDIDATES:
        key = f"prior_{cand}"
        cached = cache.get(key)
        if cached is not None:
            prior[cand] = float(cached)
        else:
            s = runner.score_continuation(NULL_PROMPT, " " + cand)
            cache.set(key, s)
            prior[cand] = s
    return prior


def recompute_with_pmi(
    model_name: str,
    config: ModelConfig,
    records: list[dict],
    cache_dir: str,
    alpha: float,
) -> dict:
    runner = get_local_runner(config)
    cache = ScoreCache(cache_dir, f"boolq_{model_name}")

    prior = compute_prior(runner, cache)
    print(
        f"  {model_name}: alpha={alpha:.3f}  "
        f"prior log p(Yes)={prior['Yes']:.3f}  log p(No)={prior['No']:.3f}"
    )

    results = []
    for rec in records:
        raw_scores = {}
        for cand in CANDIDATES:
            key = f"{hash(rec['prompt'])}_{cand}"
            cached = cache.get(key)
            if cached is not None:
                raw_scores[cand] = float(cached)
            else:
                s = runner.score_continuation(rec["prompt"], " " + cand)
                cache.set(key, s)
                raw_scores[cand] = s

        pmi_scores = {c: raw_scores[c] - alpha * prior[c] for c in CANDIDATES}
        predicted_raw = max(raw_scores, key=raw_scores.__getitem__)
        predicted_pmi = max(pmi_scores, key=pmi_scores.__getitem__)
        results.append({
            "id": rec["id"],
            "gold_answer": rec["gold_answer"],
            "predicted_raw": predicted_raw,
            "predicted_pmi": predicted_pmi,
            "correct_raw": predicted_raw == rec["gold_answer"],
            "correct_pmi": predicted_pmi == rec["gold_answer"],
        })

    acc_raw = [r["correct_raw"] for r in results]
    acc_pmi = [r["correct_pmi"] for r in results]
    yes_pmi = sum(1 for r in results if r["predicted_pmi"] == "Yes")
    no_pmi  = sum(1 for r in results if r["predicted_pmi"] == "No")

    raw_mean = sum(acc_raw) / len(acc_raw) * 100
    pmi_mean = sum(acc_pmi) / len(acc_pmi) * 100
    print(f"    Raw acc: {raw_mean:.1f}%  →  PMI acc: {pmi_mean:.1f}%"
          f"  (PMI pred_Yes={yes_pmi}, pred_No={no_pmi})")

    if model_name in _LOCAL_MODEL_RUNNERS:
        del _LOCAL_MODEL_RUNNERS[model_name]
    torch.cuda.empty_cache()

    return {
        "summary_raw": {"accuracy": bootstrap_ci(acc_raw)},
        "summary_pmi": {"accuracy": bootstrap_ci(acc_pmi)},
        "prior": prior,
        "alpha": alpha,
        "per_record": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_boolq")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument("--alpha", type=float, default=1.0)
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
    print(f"Loaded {len(records)} records")

    names = args.model_names or [
        n for n, c in all_configs.items()
        if c.mode == "hf_local" and n not in {"generator", "verifier"}
    ]

    output = {}
    for name in names:
        if name not in all_configs:
            print(f"  WARNING: {name} not in config, skipping")
            continue
        output[name] = recompute_with_pmi(
            name,
            all_configs[name],
            records,
            args.cache_dir,
            args.alpha,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
