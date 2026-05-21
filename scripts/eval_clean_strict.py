"""Clean-strict E4 evaluation entry point.

This wrapper intentionally delegates to evaluate_models.py so the clean-strict
entry point uses the same candidate construction, scoring, and aggregation code.
By default, evaluate_models.py reports hard single-gold NegRank; pass
--use-multi-answer-negatives only for the explicit multi-answer diagnostic.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
