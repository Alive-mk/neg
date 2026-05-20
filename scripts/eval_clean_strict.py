"""Clean strict E4 evaluation entrypoint.

This is a thin wrapper around ``evaluate_models.py``. By default it keeps the
existing hard single-gold NegRank behavior; pass
``--use-multi-answer-negatives`` only for explicit multi-answer diagnostics.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
