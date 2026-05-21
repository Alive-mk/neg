"""Thin clean-strict evaluation entry point.

This wrapper reuses scripts/evaluate_models.py and only supplies the canonical
clean strict input path by default. Hard single-gold NegRank remains the default
because --use-multi-answer-negatives is opt-in, matching evaluate_models.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluate_models import main as evaluate_models_main


DEFAULT_CLEAN_STRICT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
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
    if args.model_names is not None:
        forwarded.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        forwarded.append("--use-multi-answer-negatives")
    return forwarded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the E4 clean strict set via scripts/evaluate_models.py. "
            "Defaults keep hard single-gold NegRank; pass "
            "--use-multi-answer-negatives only for the explicit multi-answer audit."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default=DEFAULT_CLEAN_STRICT_INPUT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Prefix prepended to every neg prompt. Use 'auto' for oracle "
            "behavior tokens, or leave empty for no-token evaluation."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Opt in to valid_negatives for select_gold_neg records. Omit this "
            "flag for the hard single-gold clean strict main metric."
        ),
    )
    args = parser.parse_args()

    sys.argv = [
        str(Path(__file__).with_name("evaluate_models.py")),
        *build_forwarded_args(args),
    ]
    evaluate_models_main()


if __name__ == "__main__":
    main()
