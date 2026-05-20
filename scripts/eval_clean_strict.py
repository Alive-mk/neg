"""Thin clean-strict E4 evaluation entry point.

This wrapper intentionally reuses scripts/evaluate_models.py without changing
its defaults. In particular, hard single-gold NegRank remains the default
because --use-multi-answer-negatives is opt-in.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
