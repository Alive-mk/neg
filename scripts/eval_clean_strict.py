"""Clean strict E4 evaluation entry point.

This wrapper intentionally delegates to scripts/evaluate_models.py so metric
definitions and defaults stay identical. Hard single-gold NegRank remains the
default because --use-multi-answer-negatives is still opt-in.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
