"""
Prompt Engineering baseline evaluation.

Prepends a negation-awareness instruction to each record's prompts before
scoring, without any fine-tuning. Tests three prefix styles:

  - none     : original prompts (= base model, for sanity check)
  - warning  : "Pay careful attention to negation words..."
  - persona  : "You are a precise reasoner who correctly inverts meaning on negation..."

These mirror the Warning-based and Persona-based prompt families from
Barreto & Jana (EMNLP 2025 Findings), adapted for continuation-scoring mode.

Usage:
  python scripts/evaluate_prompt_baseline.py \
      --models  configs/model_config.json \
      --input   data/processed/splits/large_test/test.jsonl \
      --output  outputs/eval_prompt_baseline.json \
      --model-name qwen2_5_7b \
      --styles none warning persona
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import aggregate_results, evaluate_record
from neg_blindness.io_utils import load_records, write_json
from neg_blindness.schema import ExperimentRecord


PREFIXES: dict[str, str] = {
    "none": "",
    "warning": (
        "[Negation Notice] Pay careful attention to negation words such as "
        "'not', 'no', 'never', 'without', 'cannot'. "
        "Negation completely inverts the expected meaning. "
        "Choose the answer that correctly reflects the negated statement.\n\n"
    ),
    "persona": (
        "[Instruction] You are a precise language model that correctly handles "
        "negation. When a statement contains negation words like 'not' or 'never', "
        "you invert the expected answer accordingly.\n\n"
    ),
    "cot": (
        "[Reasoning Guide] To answer correctly, follow these steps: "
        "Step 1 — locate any negation words in the question (e.g. 'not', 'no', 'never', 'without'). "
        "Step 2 — determine how the negation changes the expected answer relative to the non-negated form. "
        "Step 3 — select the answer that reflects this negated meaning. "
        "Apply this reasoning before choosing your answer.\n\n"
    ),
}


def apply_prefix(record: ExperimentRecord, prefix: str) -> ExperimentRecord:
    if not prefix:
        return record
    rec = copy.deepcopy(record)
    rec.prompt_pos = prefix + rec.prompt_pos
    rec.prompt_neg = prefix + rec.prompt_neg
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_prompt_baseline")
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["none", "warning", "persona"],
        choices=list(PREFIXES.keys()),
    )
    args = parser.parse_args()

    configs = load_model_configs(args.models)
    config = configs[args.model_name]
    records = load_records(args.input)
    print(f"[prompt_baseline] loaded {len(records)} records, model={args.model_name}")

    output: dict[str, dict] = {}
    for style in args.styles:
        prefix = PREFIXES[style]
        cache = ScoreCache(f"{args.cache_dir}/{style}", args.model_name)
        modified = [apply_prefix(r, prefix) for r in records]
        per_record = [evaluate_record(r, config, cache=cache) for r in modified]
        summary = aggregate_results(per_record)
        output[style] = {"summary": summary, "per_record": per_record}
        neg = summary.get("NegSuppRate", {})
        flip = summary.get("FlipAcc", {})
        print(
            f"[prompt_baseline] style={style:8s}  "
            f"NegSuppRate={neg.get('mean', 0):.3f}  FlipAcc={flip.get('mean', 0):.3f}"
        )

    write_json(args.output, output)
    print(f"[prompt_baseline] written -> {args.output}")


if __name__ == "__main__":
    main()
