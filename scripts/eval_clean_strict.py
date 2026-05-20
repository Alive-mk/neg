"""Thin entry point for clean-strict E4 candidate-ranking evaluation.

This wrapper delegates metric computation to ``evaluate_models.py``. Its only
job is to provide the standard clean-strict input/output defaults while keeping
``--use-multi-answer-negatives`` opt-in, so the default remains hard
single-gold NegRank.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_models import main as evaluate_models_main


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate models on the clean-strict E4 split using the shared "
            "evaluate_models.py metric implementation."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
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
        help=(
            "Count record.valid_negatives as correct for select_gold_neg "
            "records. Leave unset for hard single-gold NegRank."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    forwarded = [
        "--models",
        args.models,
        "--input",
        args.input,
        "--output",
        args.output,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.model_names is not None:
        forwarded.extend(["--model-names", *args.model_names])
    if args.neg_prefix:
        forwarded.extend(["--neg-prefix", args.neg_prefix])
    if args.use_multi_answer_negatives:
        forwarded.append("--use-multi-answer-negatives")

    sys.argv = [sys.argv[0], *forwarded]
    evaluate_models_main()


if __name__ == "__main__":
    main()
