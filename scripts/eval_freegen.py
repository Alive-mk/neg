"""Standard entrypoint for supplementary free-generation evaluation.

This wrapper only dispatches to the existing suppress/preserve evaluators. It
does not change generation settings, keyword-proxy scoring, or output schemas.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def target_for_mode(mode: str) -> tuple[str, Callable[[], None]]:
    if mode == "suppress":
        from evaluate_free_generation import main

        return "evaluate_free_generation.py", main
    if mode == "preserve":
        from eval_freegen_preserve import main

        return "eval_freegen_preserve.py", main
    raise ValueError(f"Unsupported free-generation mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch free-generation evaluation to the existing suppress or "
            "preserve evaluator. Free-generation is a supplementary keyword "
            "proxy, not the main candidate-ranking benchmark."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["suppress", "preserve"],
        default="suppress",
        help="Evaluation type to run. Default: suppress.",
    )
    parser.add_argument(
        "--target-help",
        action="store_true",
        help="Show help for the selected underlying evaluator.",
    )
    args, forward_args = parser.parse_known_args()
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    if args.target_help:
        forward_args = ["--help"]
    if not forward_args:
        parser.error(
            "pass evaluator arguments such as --model-path, --input and --output; "
            "use --target-help for the selected evaluator's full options"
        )

    target_name, target_main = target_for_mode(args.mode)
    sys.argv = [target_name, *forward_args]
    target_main()


if __name__ == "__main__":
    main()
