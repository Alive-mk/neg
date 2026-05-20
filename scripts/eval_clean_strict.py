from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate_models


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Thin clean-strict E4 wrapper around scripts/evaluate_models.py. "
            "By default this reports hard single-gold NegRank; pass "
            "--use-multi-answer-negatives only for the audited multi-answer view."
        )
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="auto",
        help="Default is oracle behavior tokens derived from expected_neg_behavior.",
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help="Opt in to Multi-answer NegRank; omitted by default for hard single-gold.",
    )
    args = parser.parse_args()

    forwarded = [
        str(Path(evaluate_models.__file__).name),
        "--models",
        args.models,
        "--input",
        args.input,
        "--output",
        args.output,
        "--cache-dir",
        args.cache_dir,
        "--neg-prefix",
        args.neg_prefix,
    ]
    if args.model_names:
        forwarded.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        forwarded.append("--use-multi-answer-negatives")

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        evaluate_models.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
