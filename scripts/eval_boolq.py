from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


DEFAULT_MODELS_CONFIG = "configs/model_config_boolq.json"
DEFAULT_INPUT = "data/boolq_full_val.jsonl"
DEFAULT_CACHE_DIR = "outputs/score_cache_boolq"
DEFAULT_RAW_OUTPUT = "outputs/eval_boolq_raw.json"
DEFAULT_PMI_OUTPUT = "outputs/eval_boolq_pmi.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standard BoolQ evaluation entrypoint. By default it runs the "
            "existing raw yes/no log-probability ranking evaluator on the full "
            "BoolQ validation file; --pmi dispatches to the existing scaled "
            "PMI evaluator without changing its metric definition."
        )
    )
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        help=(
            "Output JSON path. Defaults to outputs/eval_boolq_raw.json, or "
            "outputs/eval_boolq_pmi.json when --pmi is set."
        ),
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--pmi",
        action="store_true",
        help="Use standard/scaled PMI scoring via scripts/eval_boolq_pmi.py.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="PMI prior scale. The standard main-table setting is alpha=1.0.",
    )
    return parser.parse_args()


def forwarded_args(args: argparse.Namespace) -> list[str]:
    output = args.output or (DEFAULT_PMI_OUTPUT if args.pmi else DEFAULT_RAW_OUTPUT)
    argv = [
        "eval_boolq_pmi.py" if args.pmi else "eval_boolq_negation.py",
        "--models-config",
        args.models_config,
        "--input",
        args.input,
        "--output",
        output,
        "--cache-dir",
        args.cache_dir,
    ]
    if args.model_names:
        argv.extend(["--model-names", *args.model_names])
    if args.pmi:
        argv.extend(["--alpha", str(args.alpha)])
    return argv


def main() -> None:
    args = parse_args()
    sys.argv = forwarded_args(args)
    if args.pmi:
        from eval_boolq_pmi import main as eval_main  # noqa: E402
    else:
        from eval_boolq_negation import main as eval_main  # noqa: E402

    eval_main()


if __name__ == "__main__":
    main()
