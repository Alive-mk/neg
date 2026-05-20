"""
Evaluate MGNM oracle model using per-record router predictions.
Loads router_predictions.json and applies per-record tokens.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, load_model_configs
from neg_blindness.evaluation import aggregate_results, evaluate_record
from neg_blindness.io_utils import load_records, write_json

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
        "--use-multi-answer-negatives",
        action="store_true",
        help="Count record.valid_negatives as correct for select_gold_neg records.",
    )
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text())
    configs  = load_model_configs(args.models)
    records  = load_records(args.input)

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
                neg_prefix=predictions.get(record.id, ""),
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
