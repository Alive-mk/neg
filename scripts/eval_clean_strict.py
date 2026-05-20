from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from evaluate_models import run_evaluation  # noqa: E402


DEFAULT_CLEAN_STRICT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_clean_strict"
DEFAULT_OUTPUT = "outputs/eval_clean_strict.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-ranking models on the clean strict E4 split. "
            "This is a thin wrapper around scripts/evaluate_models.py; by default "
            "it keeps hard single-gold NegRank for the main table."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default=DEFAULT_CLEAN_STRICT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="auto",
        help=(
            "Prefix prepended to neg prompts. The clean strict oracle-token "
            "setting uses 'auto' to derive [SUPPRESS]/[SELECT]/[PRESERVE] from "
            "expected_neg_behavior."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Diagnostic only: count record.valid_negatives as correct for "
            "select_gold_neg. Leave unset for the hard single-gold main metric."
        ),
    )
    args = parser.parse_args()

    run_evaluation(
        models=args.models,
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_names=args.model_names,
        neg_prefix=args.neg_prefix,
        use_multi_answer_negatives=args.use_multi_answer_negatives,
    )


if __name__ == "__main__":
    main()
