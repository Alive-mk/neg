from __future__ import annotations

from evaluate_models import build_parser, run_evaluation


def main() -> None:
    parser = build_parser(
        description=(
            "Run clean strict candidate-ranking evaluation with hard "
            "single-gold NegRank. Multi-answer SELECT credit is intentionally "
            "disabled in this wrapper; use evaluate_models.py or "
            "reaggregate_on_clean_e4.py for MultiAnswerNegRank diagnostics."
        ),
        include_multi_answer_flag=False,
    )
    args = parser.parse_args()

    run_evaluation(
        models=args.models,
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_names=args.model_names,
        neg_prefix=args.neg_prefix,
        use_multi_answer_negatives=False,
    )


if __name__ == "__main__":
    main()
