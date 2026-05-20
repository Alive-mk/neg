from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_MODELS = ROOT / "scripts" / "evaluate_models.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-strict E4 evaluator wrapper. Delegates scoring and aggregation "
            "to scripts/evaluate_models.py while keeping select_gold_neg correctness "
            "on the default hard single-gold definition."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="",
        help=(
            "Prefix prepended to every neg prompt. Use 'auto' to pick "
            "[SUPPRESS]/[SELECT]/[PRESERVE] from expected_neg_behavior, or a "
            "fixed string for out-of-domain sets."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    forwarded_argv = [
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
        forwarded_argv.extend(["--model-names", *args.model_names])
    if args.neg_prefix:
        forwarded_argv.extend(["--neg-prefix", args.neg_prefix])

    sys.argv = forwarded_argv
    runpy.run_path(str(EVALUATE_MODELS), run_name="__main__")


if __name__ == "__main__":
    main()
