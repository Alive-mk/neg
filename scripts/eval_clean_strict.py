from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_MODELS = ROOT / "scripts" / "evaluate_models.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Thin clean-strict E4 wrapper around scripts/evaluate_models.py. "
            "Hard single-gold NegRank is the default because this wrapper does "
            "not pass --use-multi-answer-negatives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="configs/model_config.json")
    parser.add_argument(
        "--input",
        default="data/processed/validated_largetest_v2_clean_strict.jsonl",
        help="Clean strict E4 evaluation JSONL.",
    )
    parser.add_argument("--output", default="outputs/eval_clean_strict.json")
    parser.add_argument("--cache-dir", default="outputs/score_cache_clean_strict")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix",
        default="auto",
        help=(
            "Passed through to evaluate_models.py. Use auto for oracle behavior "
            "tokens, an empty string for no-token, or a fixed token for diagnostics."
        ),
    )
    args = parser.parse_args()

    command = [
        sys.executable,
        str(EVALUATE_MODELS),
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
        command.extend(["--model-names", *args.model_names])

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
