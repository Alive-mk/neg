from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_models  # noqa: E402


DEFAULT_MODELS = "configs/model_config.json"
DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Thin clean-strict wrapper around scripts/evaluate_models.py. "
            "By default this keeps hard single-gold NegRank because it does not "
            "pass --use-multi-answer-negatives."
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
            "Forwarded to evaluate_models.py. Use 'auto' only for oracle "
            "behavior-token evaluation."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Opt in to Multi-answer NegRank for select_gold_neg records. "
            "Leave unset for the default hard single-gold clean-strict metric."
        ),
    )
    return parser


def to_evaluate_models_argv(args: argparse.Namespace) -> list[str]:
    argv = [
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
        argv.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        argv.append("--use-multi-answer-negatives")
    return argv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.argv = to_evaluate_models_argv(args)
    evaluate_models.main()


if __name__ == "__main__":
    main()
