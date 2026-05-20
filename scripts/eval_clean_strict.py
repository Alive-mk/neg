"""Clean-strict E4 evaluation entry point.

This is a thin wrapper around ``scripts/evaluate_models.py`` so the standard
P0 clean-strict command has a stable script name. Metric definitions and
candidate scoring stay in ``neg_blindness.evaluation``. By default SELECT is
scored with the strict single-gold target; pass
``--use-multi-answer-negatives`` only for the audited multi-answer variant.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
