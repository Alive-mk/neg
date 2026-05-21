"""Evaluate candidate-ranking models using per-record router predictions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import aggregate_results, evaluate_record
from neg_blindness.io_utils import load_records, write_json

ALLOWED_TOKENS = {"", "[SUPPRESS]", "[SELECT]", "[PRESERVE]"}


def validate_predictions(predictions, records, allow_missing_as_empty=False) -> None:
    if not isinstance(predictions, dict):
        raise SystemExit("router predictions must be a JSON object mapping record_id -> token")

    invalid = [
        (str(record_id), str(token))
        for record_id, token in predictions.items()
        if str(token) not in ALLOWED_TOKENS
    ]
    if invalid:
        examples = ", ".join(f"{record_id}={token!r}" for record_id, token in invalid[:5])
        raise SystemExit(
            "router predictions contain unsupported tokens; allowed tokens are "
            f"{sorted(ALLOWED_TOKENS)}. Examples: {examples}"
        )

    record_ids = {record.id for record in records}
    prediction_ids = {str(record_id) for record_id in predictions}
    missing = sorted(record_ids - prediction_ids)
    if missing and not allow_missing_as_empty:
        examples = ", ".join(missing[:10])
        raise SystemExit(
            f"router predictions missing {len(missing)} input record ids. "
            "Pass --allow-missing-as-empty only for an explicit no-token/empty-prefix policy. "
            f"Examples: {examples}"
        )

    extra = sorted(prediction_ids - record_ids)
    if extra:
        examples = ", ".join(extra[:10])
        print(
            f"Warning: router predictions contain {len(extra)} ids not present in input; "
            f"ignoring examples: {examples}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",      required=True)
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--predictions", required=True,
                        help="JSON file mapping record_id -> predicted token")
    parser.add_argument("--cache-dir",   default="outputs/score_cache_router")
    parser.add_argument("--model-names", nargs="*")
    parser.add_argument(
        "--allow-missing-as-empty",
        action="store_true",
        help="Treat missing prediction ids as empty prefixes. Use only for an explicit no-token/empty-SELECT policy.",
    )
    parser.add_argument(
        "--use-multi-answer-negatives",
        action="store_true",
        help="Count record.valid_negatives as correct for select_gold_neg records.",
    )
    args = parser.parse_args()

    configs  = load_model_configs(args.models)
    records  = load_records(args.input)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    validate_predictions(predictions, records, args.allow_missing_as_empty)

    selected = args.model_names or [
        n for n, c in configs.items()
        if c.mode in {"logprob", "choice", "chat_json", "hf_local"}
        and n not in {"generator", "verifier"}
    ]

    output = {}
    for model_name in selected:
        config = configs[model_name]
        cache  = ScoreCache(args.cache_dir, model_name)
        per_record = [
            evaluate_record(
                record, config, cache=cache,
                neg_prefix=str(predictions.get(record.id, "")),
                use_multi_answer_negatives=args.use_multi_answer_negatives,
            )
            for record in records
        ]
        output[model_name] = {
            "summary":    aggregate_results(per_record),
            "per_record": [r.__dict__ if hasattr(r, "__dict__") else r
                           for r in per_record],
        }

    write_json(args.output, output)
    # Print summary
    for mn, res in output.items():
        s = res["summary"]
        def mv(k): v = s.get(k,0); return (v["mean"] if isinstance(v,dict) else v)*100
        print(f"{mn}: FlipAcc={mv('FlipAcc'):.1f}%  ScopeCtrl={mv('ScopeControlAcc'):.1f}%  "
              f"NegRank={mv('NegRankAcc'):.1f}%")

if __name__ == "__main__":
    main()
