from __future__ import annotations

import runpy
import sys
from pathlib import Path

DEFAULT_CLEAN_STRICT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"
EVALUATE_MODELS = Path(__file__).with_name("evaluate_models.py")

HELP_NOTE = f"""\
Clean strict wrapper around scripts/evaluate_models.py.

Default input: {DEFAULT_CLEAN_STRICT_INPUT}
Metric default: hard single-gold NegRank, because this wrapper does not add
--use-multi-answer-negatives. Pass that flag explicitly only for the
multi-answer diagnostic.

Prefix behavior is unchanged from evaluate_models.py: pass --neg-prefix auto
for oracle behavior-token clean strict runs.
"""


def _has_input(argv: list[str]) -> bool:
    return any(arg == "--input" or arg.startswith("--input=") for arg in argv)


def _wants_help(argv: list[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in argv)


def main() -> None:
    forwarded = sys.argv[1:]
    if _wants_help(forwarded):
        print(HELP_NOTE)
    elif not _has_input(forwarded):
        forwarded = ["--input", DEFAULT_CLEAN_STRICT_INPUT, *forwarded]

    sys.argv = [str(EVALUATE_MODELS), *forwarded]
    runpy.run_path(str(EVALUATE_MODELS), run_name="__main__")


if __name__ == "__main__":
    main()
