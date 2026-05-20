"""Thin clean-strict E4 evaluation entry point.

This wrapper intentionally delegates scoring and aggregation to
scripts/evaluate_models.py. By default it keeps evaluate_models.py's hard
single-gold NegRankAcc behavior; pass --use-multi-answer-negatives only for the
explicit multi-answer diagnostic setting.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_command(args: argparse.Namespace) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "evaluate_models.py"),
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
        command.extend(["--model-names", *args.model_names])
    if args.neg_prefix:
        command.extend(["--neg-prefix", args.neg_prefix])
    if args.use_multi_answer_negatives:
        command.append("--use-multi-answer-negatives")

    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate E4 clean strict records through scripts/evaluate_models.py. "
            "Default NegRankAcc is hard single-gold."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help="Forwarded to evaluate_models.py; pass 'auto' for oracle behavior tokens.",
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help="Forwarded diagnostic option; default is hard single-gold NegRankAcc.",
    )
    args = parser.parse_args()

    subprocess.run(build_command(args), check=True)


if __name__ == "__main__":
    main()
