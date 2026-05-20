"""
Thin clean-strict evaluation entrypoint.

This script intentionally delegates scoring and aggregation to
scripts/evaluate_models.py. By default it keeps the hard single-gold
NegRank behavior because --use-multi-answer-negatives remains opt-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_models import main as evaluate_models_main  # noqa: E402


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def main() -> None:
    argv = sys.argv[1:]
    if not _has_option(argv, "--cache-dir"):
        sys.argv.extend(["--cache-dir", "outputs/score_cache_clean_strict"])
    evaluate_models_main()


if __name__ == "__main__":
    main()
