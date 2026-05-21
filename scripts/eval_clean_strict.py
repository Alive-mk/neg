"""Clean-strict E4 candidate-ranking evaluation wrapper.

This script is intentionally a thin CLI wrapper around ``evaluate_models.py``.
It keeps the default SELECT metric at hard single-gold by not forwarding
``--use-multi-answer-negatives``. ``MultiAnswerNegRankAcc`` is still emitted by
the underlying evaluator as a diagnostic metric.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_delegate_argv(args: argparse.Namespace) -> list[str]:
    script_path = Path(__file__).with_name("evaluate_models.py")
    delegate = [
        str(script_path),
        "--models",
        args.models,
        "--input",
        args.input,
        "--output",
        args.output,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.model_names:
        delegate.extend(["--model-names", *args.model_names])
    if args.neg_prefix:
        delegate.extend(["--neg-prefix", args.neg_prefix])
    return delegate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate models on the clean-strict E4 set. This wrapper delegates "
            "to scripts/evaluate_models.py and preserves the hard single-gold "
            "SELECT metric by default."
        )
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Prefix prepended to every neg prompt. Use 'auto' for oracle "
            "behavior tokens, or leave empty for no token."
        ),
    )
    args = parser.parse_args()

    sys.argv = build_delegate_argv(args)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
