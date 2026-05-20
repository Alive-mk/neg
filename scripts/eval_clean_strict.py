"""Clean-strict E4 evaluation entrypoint.

This is a thin wrapper around ``scripts/evaluate_models.py``.  It only fixes
the standard clean-strict input/output/cache defaults; metric computation stays
in ``neg_blindness.evaluation``.
"""
from __future__ import annotations

import argparse
import sys

from evaluate_models import main as evaluate_models_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean-strict E4 candidate-ranking evaluation via "
            "scripts/evaluate_models.py. By default this keeps the hard "
            "single-gold select_gold_neg definition used by evaluate_models.py."
        )
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument(
        "--input",
        default="data/processed/validated_largetest_v2_clean_strict.jsonl",
        help="Clean-strict E4 JSONL input.",
    )
    parser.add_argument("--output", default="outputs/eval_clean_strict.json")
    parser.add_argument("--cache-dir", default="outputs/score_cache_clean_strict")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Forwarded to evaluate_models.py. Use 'auto' for oracle behavior "
            "tokens; default is no extra prefix."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Forwarded to evaluate_models.py. Default is off, so select "
            "correctness and FlipAcc use hard single-gold labels."
        ),
    )
    args = parser.parse_args()

    delegated_argv = [
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
        delegated_argv.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        delegated_argv.append("--use-multi-answer-negatives")

    sys.argv = delegated_argv
    evaluate_models_main()


if __name__ == "__main__":
    main()
