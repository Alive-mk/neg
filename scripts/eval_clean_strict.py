from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_models import build_parser, run_evaluation  # noqa: E402


def main() -> None:
    parser = build_parser(
        description=(
            "Evaluate the clean-strict E4 candidate-ranking split using the "
            "standard evaluate_models.py implementation."
        ),
        epilog=(
            "This wrapper keeps the hard single-gold NegRank default. For the "
            "audited multi-answer SELECT metric, use scripts/evaluate_models.py "
            "with --use-multi-answer-negatives explicitly."
        ),
        include_multi_answer_flag=False,
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
