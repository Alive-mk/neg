"""Clean-strict E4 evaluation entry point.

This is intentionally a thin wrapper around ``evaluate_models.py``.  The
default scoring remains hard single-gold NegRank because
``--use-multi-answer-negatives`` is only enabled when the caller passes it
explicitly.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
