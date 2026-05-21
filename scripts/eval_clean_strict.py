#!/usr/bin/env python3
"""Clean-strict E4 evaluation entry point.

This thin wrapper reuses ``scripts/evaluate_models.py`` and therefore keeps the
default SELECT scoring as hard single-gold. Pass
``--use-multi-answer-negatives`` explicitly to opt into the multi-answer
diagnostic metric.
"""

from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
