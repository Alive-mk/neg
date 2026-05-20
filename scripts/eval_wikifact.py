from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_models import main as evaluate_models_main  # noqa: E402


DEFAULT_WIKIFACT_INPUT = "data/external/wikifact_neg_patched.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_wikifact"
DEFAULT_OUTPUT = "outputs/eval_wikifact.json"
DEFAULT_NEG_PREFIX = "[SUPPRESS]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-ranking models on the WikiFact suppress-only "
            "OOD set. This is a thin wrapper around scripts/evaluate_models.py "
            "and keeps the existing WikiFact convention of using a fixed "
            "[SUPPRESS] prefix."
        )
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", default=DEFAULT_WIKIFACT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default=DEFAULT_NEG_PREFIX,
        help=(
            "Prefix prepended to every WikiFact neg prompt. The standard "
            "suppress-only setting uses '[SUPPRESS]'."
        ),
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Forwarded diagnostic flag from evaluate_models.py. It should have "
            "no effect for the suppress-only WikiFact set."
        ),
    )
    args = parser.parse_args()

    forwarded_args = [
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
        forwarded_args.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        forwarded_args.append("--use-multi-answer-negatives")

    sys.argv = forwarded_args
    evaluate_models_main()


if __name__ == "__main__":
    main()
