"""Thin clean-strict E4 entry point.

This wrapper delegates all scoring and aggregation to ``evaluate_models.py``.
By default it uses the clean-strict largetest split and keeps the existing
hard single-gold NegRank behavior. Multi-answer SELECT evaluation still
requires passing ``--use-multi-answer-negatives`` explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_models import main as evaluate_models_main


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the clean-strict E4 split via evaluate_models.py. "
            "Defaults to hard single-gold NegRank unless "
            "--use-multi-answer-negatives is passed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Clean-strict E4 JSONL input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Prefix prepended to every neg prompt. Use 'auto' to pick "
            "[SUPPRESS]/[SELECT]/[PRESERVE] from expected_neg_behavior."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help="Count record.valid_negatives as correct for select_gold_neg records.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    argv = [
        "--models", args.models,
        "--input", args.input,
        "--output", args.output,
        "--cache-dir", args.cache_dir,
        "--neg-prefix", args.neg_prefix,
    ]
    if args.model_names:
        argv.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        argv.append("--use-multi-answer-negatives")

    sys.argv = [sys.argv[0], *argv]
    evaluate_models_main()


if __name__ == "__main__":
    main()
