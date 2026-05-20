"""Thin entry point for clean-strict E4 candidate-ranking evaluation.

The metric implementation lives in scripts/evaluate_models.py.  This wrapper
exists so the reproducibility checklist has a stable clean-strict command while
preserving evaluate_models.py defaults, including hard single-gold NegRank.
"""
from __future__ import annotations

from evaluate_models import main


if __name__ == "__main__":
    main()
