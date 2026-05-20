from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import aggregate_results, evaluate_record
from neg_blindness.io_utils import load_records, write_json


def build_parser(
    description: str | None = None,
    epilog: str | None = None,
    include_multi_answer_flag: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--models", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--neg-prefix", default="",
        help="Prefix prepended to every neg prompt. "
             "Use 'auto' to pick [SUPPRESS]/[SELECT]/[PRESERVE] from expected_neg_behavior, "
             "or a fixed string like '[SUPPRESS]' for out-of-domain sets (e.g. WikiFact).",
    )
    if include_multi_answer_flag:
        parser.add_argument(
            "--use-multi-answer-negatives",
            action="store_true",
            help="Count record.valid_negatives as correct for select_gold_neg records.",
        )
    return parser


def run_evaluation(args: argparse.Namespace) -> None:
    _AUTO_TOKENS = {
        "suppress_target":   "[SUPPRESS]",
        "select_gold_neg":   "[SELECT]",
        "preserve_positive": "[PRESERVE]",
    }

    def resolve_prefix(record) -> str:
        if args.neg_prefix == "auto":
            return _AUTO_TOKENS.get(record.expected_neg_behavior, "")
        return args.neg_prefix

    configs = load_model_configs(args.models)
    records = load_records(args.input)

    selected_names = args.model_names or list(configs.keys())
    selected_names = [
        name
        for name in selected_names
        if configs[name].mode in {"logprob", "choice", "chat_json", "hf_local"}
        and name not in {"generator", "verifier"}
    ]

    output: dict[str, dict] = {}
    for model_name in selected_names:
        config = configs[model_name]
        cache = ScoreCache(args.cache_dir, model_name)
        per_record = [
            evaluate_record(
                record,
                config,
                cache=cache,
                neg_prefix=resolve_prefix(record),
                use_multi_answer_negatives=getattr(args, "use_multi_answer_negatives", False),
            )
            for record in records
        ]
        output[model_name] = {
            "summary": aggregate_results(per_record),
            "per_record": per_record,
        }

    write_json(args.output, output)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
