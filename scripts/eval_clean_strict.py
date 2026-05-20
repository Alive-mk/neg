from __future__ import annotations

import argparse
import runpy
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_MODELS = ROOT / "scripts" / "evaluate_models.py"

DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate models on the clean-strict E4 split by delegating to "
            "scripts/evaluate_models.py. The default metric policy is hard "
            "single-gold: --use-multi-answer-negatives is intentionally not "
            "exposed here, so NegRankAcc and FlipAcc keep the strict main-table "
            "definition. MultiAnswerNegRankAcc is still reported by the shared "
            "aggregator as a separate diagnostic."
        )
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Prefix prepended to every neg prompt. Use 'auto' for oracle "
            "behavior tokens, or leave empty to preserve evaluate_models.py's "
            "default no-prefix behavior."
        ),
    )
    parser.add_argument(
        "--print-delegated-command",
        action="store_true",
        help="Print the equivalent evaluate_models.py command without running it.",
    )
    return parser


def delegated_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        str(EVALUATE_MODELS),
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
        argv.extend(["--model-names", *args.model_names])
    if args.neg_prefix:
        argv.extend(["--neg-prefix", args.neg_prefix])
    return argv


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    forwarded = delegated_argv(args)

    if args.print_delegated_command:
        print(" ".join(shlex.quote(part) for part in [sys.executable, *forwarded]))
        return

    previous_argv = sys.argv[:]
    try:
        sys.argv = forwarded
        runpy.run_path(str(EVALUATE_MODELS), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
