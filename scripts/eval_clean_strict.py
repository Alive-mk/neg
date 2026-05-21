"""Thin clean-strict E4 wrapper around ``scripts/evaluate_models.py``.

This entry point standardizes the P0 clean-strict command without changing the
underlying metric implementation. By default it keeps evaluate_models.py's hard
single-gold SELECT scoring; pass --use-multi-answer-negatives explicitly when
running the audited multi-answer variant.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATE_MODELS = ROOT / "scripts" / "evaluate_models.py"

DEFAULT_ARGS = {
    "--models": "configs/model_config.json",
    "--input": "data/processed/validated_largetest_v2_clean_strict.jsonl",
    "--output": "outputs/eval_clean_strict.json",
    "--cache-dir": "outputs/score_cache_clean_strict",
}


def has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def main() -> None:
    forwarded = sys.argv[1:]
    defaults: list[str] = []
    for option, value in DEFAULT_ARGS.items():
        if not has_option(forwarded, option):
            defaults.extend([option, value])

    sys.argv = [str(EVALUATE_MODELS), *defaults, *forwarded]
    runpy.run_path(str(EVALUATE_MODELS), run_name="__main__")


if __name__ == "__main__":
    main()
