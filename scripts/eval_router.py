"""Router downstream evaluation entry point.

This is a thin wrapper around eval_with_router.py. It exists so router runs have
a stable command name while sharing the same prediction coverage validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import eval_with_router  # noqa: E402


if __name__ == "__main__":
    eval_with_router.main()
