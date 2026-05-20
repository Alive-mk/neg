from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_with_router import main as eval_with_router_main  # noqa: E402


DEFAULT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
DEFAULT_OUTPUT = "outputs/eval_router.json"
DEFAULT_CACHE_DIR = "outputs/score_cache_router"
DEFAULT_MODELS = "configs/model_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standard router downstream evaluation entrypoint. This is a thin "
            "wrapper around scripts/eval_with_router.py and requires an explicit "
            "record_id -> token predictions JSON to avoid mixing router policies."
        )
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "JSON mapping record_id to behavior token. Use an empty string for "
            "no-token / empty-SELECT records."
        ),
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help=(
            "Forwarded diagnostic flag. Leave unset for the default hard "
            "single-gold NegRank main metric."
        ),
    )
    args = parser.parse_args()

    forwarded_args = [
        "eval_with_router.py",
        "--models",
        args.models,
        "--input",
        args.input,
        "--output",
        args.output,
        "--predictions",
        args.predictions,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.model_names:
        forwarded_args.extend(["--model-names", *args.model_names])
    if args.use_multi_answer_negatives:
        forwarded_args.append("--use-multi-answer-negatives")

    sys.argv = forwarded_args
    eval_with_router_main()


if __name__ == "__main__":
    main()
