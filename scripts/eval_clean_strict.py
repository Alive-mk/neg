from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument("--use-multi-answer-negatives", action="store_true")
    args, _ = guard.parse_known_args()
    if args.use_multi_answer_negatives:
        raise SystemExit(
            "eval_clean_strict.py is the hard single-gold clean-strict entry. "
            "Use scripts/evaluate_models.py directly for multi-answer diagnostics."
        )

    target = Path(__file__).resolve().with_name("evaluate_models.py")
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
