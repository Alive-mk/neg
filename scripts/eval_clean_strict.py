"""Clean-strict E4 evaluation entry point.

This is a thin wrapper around scripts/evaluate_models.py. By default it leaves
--use-multi-answer-negatives disabled, so NegRankAcc is the hard single-gold
metric used for clean-strict main evaluation.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
