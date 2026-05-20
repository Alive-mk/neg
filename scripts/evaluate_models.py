from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import aggregate_results, evaluate_record
from neg_blindness.io_utils import load_records, write_json


AUTO_BEHAVIOR_TOKENS = {
    "suppress_target": "[SUPPRESS]",
    "select_gold_neg": "[SELECT]",
    "preserve_positive": "[PRESERVE]",
}


def run_evaluation(
    models: str,
    input_path: str,
    output_path: str,
    cache_dir: str = "outputs/score_cache",
    model_names: list[str] | None = None,
    neg_prefix: str = "",
    use_multi_answer_negatives: bool = False,
) -> dict[str, dict]:
    configs = load_model_configs(models)
    records = load_records(input_path)

    def resolve_prefix(record) -> str:
        if neg_prefix == "auto":
            return AUTO_BEHAVIOR_TOKENS.get(record.expected_neg_behavior, "")
        return neg_prefix

    selected_names = model_names or list(configs.keys())
    selected_names = [
        name
        for name in selected_names
        if configs[name].mode in {"logprob", "choice", "chat_json", "hf_local"}
        and name not in {"generator", "verifier"}
    ]

    output: dict[str, dict] = {}
    for model_name in selected_names:
        config = configs[model_name]
        cache = ScoreCache(cache_dir, model_name)
        per_record = [
            evaluate_record(
                record,
                config,
                cache=cache,
                neg_prefix=resolve_prefix(record),
                use_multi_answer_negatives=use_multi_answer_negatives,
            )
            for record in records
        ]
        output[model_name] = {
            "summary": aggregate_results(per_record),
            "per_record": per_record,
        }

    write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help="Count record.valid_negatives as correct for select_gold_neg records.",
    )
    args = parser.parse_args()

    run_evaluation(
        models=args.models,
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_names=args.model_names,
        neg_prefix=args.neg_prefix,
        use_multi_answer_negatives=args.use_multi_answer_negatives,
    )


if __name__ == "__main__":
    main()
