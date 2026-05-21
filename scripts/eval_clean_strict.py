"""Standard clean-strict E4 evaluation entrypoint.

This is a thin wrapper around scripts/evaluate_models.py. It only supplies the
canonical clean-strict defaults; all scoring and aggregation logic stays in the
shared evaluator. By default it uses hard single-gold NegRank because
--use-multi-answer-negatives is opt-in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_models import main as evaluate_models_main  # noqa: E402


DEFAULT_MODELS = "configs/model_config.json"
DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate models on the E4 clean-strict candidate-ranking split. "
            "Defaults preserve evaluate_models.py hard single-gold NegRank."
        )
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
            "Prefix prepended to every neg prompt. Use 'auto' for oracle "
            "behavior tokens; default is empty/no-token."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Opt in to soft SELECT scoring with record.valid_negatives. "
            "Omit this flag for default hard single-gold NegRank."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    evaluator_argv = [
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
        evaluator_argv.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        evaluator_argv.append("--use-multi-answer-negatives")

    old_argv = sys.argv
    try:
        sys.argv = ["evaluate_models.py", *evaluator_argv]
        evaluate_models_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
