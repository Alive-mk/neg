"""Clean-strict E4 candidate-ranking evaluation entry point.

The token control mode is explicit on purpose. MGNM oracle-token, no-token, and
fixed-prefix evaluations are different protocols and should not share an
ambiguous default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import evaluate_models  # noqa: E402


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clean-strict E4 with an explicit token protocol. "
            "Use --token-mode oracle only for oracle behavior-token MGNM runs; "
            "use --token-mode no-token for baselines and no-token ablations."
        )
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--token-mode",
        choices=["no-token", "oracle", "fixed"],
        required=True,
        help=(
            "no-token leaves negative prompts unchanged; oracle forwards "
            "--neg-prefix auto and uses expected_neg_behavior labels; fixed "
            "uses --fixed-prefix for every negative prompt."
        ),
    )
    parser.add_argument(
        "--fixed-prefix",
        default="",
        help="Prefix used only with --token-mode fixed, e.g. '[SUPPRESS]'.",
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Opt in to multi-answer SELECT correctness. This also changes "
            "select_gold_neg neg_correct/FlipAcc, so do not mix it with hard "
            "single-gold clean-strict results."
        ),
    )
    return parser


def default_output(token_mode: str, use_multi_answer: bool) -> str:
    suffix = token_mode.replace("-", "_")
    if use_multi_answer:
        suffix = f"{suffix}_multianswer"
    return f"outputs/eval_clean_strict_{suffix}.json"


def to_evaluate_models_argv(args: argparse.Namespace) -> list[str]:
    output = args.output or default_output(
        args.token_mode,
        args.use_multi_answer_negatives,
    )
    argv = [
        str(SCRIPT_DIR / "evaluate_models.py"),
        "--models",
        args.models,
        "--input",
        args.input,
        "--output",
        output,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.model_names:
        argv.extend(["--model-names", *args.model_names])
    if args.token_mode == "oracle":
        argv.extend(["--neg-prefix", "auto"])
    elif args.token_mode == "fixed":
        if not args.fixed_prefix:
            raise SystemExit("--fixed-prefix is required when --token-mode fixed")
        argv.extend(["--neg-prefix", args.fixed_prefix])
    elif args.fixed_prefix:
        raise SystemExit("--fixed-prefix is only valid with --token-mode fixed")
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
