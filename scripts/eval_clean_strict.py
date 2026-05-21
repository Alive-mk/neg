#!/usr/bin/env python3
"""Thin clean-strict E4 wrapper around ``scripts/evaluate_models.py``.

By default this keeps the hard single-gold SELECT setting: it does not pass
``--use-multi-answer-negatives`` through to ``evaluate_models.py`` unless the
caller explicitly requests it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from shlex import quote

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.append(str(SCRIPTS))

from evaluate_models import main as evaluate_models_main  # noqa: E402

DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_MODELS = "configs/model_config.json"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-ranking models on the clean strict E4 split. "
            "This delegates to scripts/evaluate_models.py and preserves its "
            "metric definitions."
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
            "behavior tokens, or a fixed string such as '[SUPPRESS]'."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Opt into multi-answer SELECT scoring for neg_correct/FlipAcc. "
            "Default is off, so hard single-gold NegRank remains the wrapper "
            "default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the equivalent evaluate_models.py command without running model scoring.",
    )
    return parser.parse_args()


def build_forwarded_argv(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "scripts/evaluate_models.py",
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
        forwarded.append("--model-names")
        forwarded.extend(args.model_names)
    if args.use_multi_answer_negatives:
        forwarded.append("--use-multi-answer-negatives")
    return forwarded


def main() -> None:
    args = parse_args()
    forwarded = build_forwarded_argv(args)
    if args.dry_run:
        print("python " + " ".join(quote(part) for part in forwarded))
        return

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        evaluate_models_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
