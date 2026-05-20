from __future__ import annotations

import sys

from evaluate_models import main as evaluate_models_main


DEFAULT_CLEAN_STRICT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"


def main() -> None:
    argv = sys.argv[1:]
    if not any(arg == "--input" or arg.startswith("--input=") for arg in argv):
        sys.argv[1:1] = ["--input", DEFAULT_CLEAN_STRICT_INPUT]
    evaluate_models_main()


if __name__ == "__main__":
    main()
