"""Thin clean-strict E4 evaluation entry point.

This wrapper intentionally delegates to scripts/evaluate_models.py.  Its only
job is to standardize the clean-strict input/output/cache defaults while
preserving the existing metric implementation and hard single-gold default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"
DEFAULT_MODELS = "configs/model_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate models on the clean-strict E4 subset by reusing "
            "scripts/evaluate_models.py."
        ),
        epilog=(
            "Metric policy: hard single-gold is the default. Add "
            "--use-multi-answer-negatives only for an explicit multi-answer "
            "diagnostic run."
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Forwarded to evaluate_models.py. Use 'auto' for oracle behavior "
            "tokens; the default is no prefix."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Forwarded to evaluate_models.py. Off by default so clean-strict "
            "runs use hard single-gold select correctness."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    forwarded = [
        "evaluate_models.py",
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

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from evaluate_models import main as evaluate_main

    sys.argv = forwarded
    evaluate_main()


if __name__ == "__main__":
    main()
