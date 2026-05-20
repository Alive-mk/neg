"""Thin clean-strict entrypoint for E4 candidate-ranking evaluation.

This wrapper keeps the metric implementation in ``evaluate_models.py`` and
only supplies the canonical clean-strict input path when callers omit
``--input``. By default it preserves the hard single-gold NegRank behavior;
multi-answer SELECT scoring still requires the existing explicit
``--use-multi-answer-negatives`` flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from evaluate_models import main as evaluate_models_main  # noqa: E402


DEFAULT_CLEAN_STRICT_INPUT = "data/processed/validated_largetest_v2_clean_strict.jsonl"


def main() -> None:
    evaluate_models_main(input_required=False, default_input=DEFAULT_CLEAN_STRICT_INPUT)


if __name__ == "__main__":
    main()
