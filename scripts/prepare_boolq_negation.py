"""Filter BoolQ (train + validation) for negation questions; also save full val set.

Outputs:
  data/boolq_negation.jsonl   — negation subset from both splits (train+val), ~209 items
  data/boolq_full_val.jsonl   — full validation set, 3270 items (general QA capability)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

NEG_OUTPUT = ROOT / "data" / "boolq_negation.jsonl"
FULL_OUTPUT = ROOT / "data" / "boolq_full_val.jsonl"

NEG_PATTERN = re.compile(
    r"\b(not|n't|never|no\b|neither|nor|without|cannot|can't|won't|don't|"
    r"doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"couldn't|wouldn't|shouldn't|mustn't)\b",
    re.IGNORECASE,
)


def make_record(row: dict, split: str, idx: int) -> dict:
    question: str = row["question"]
    passage: str = row["passage"]
    answer: bool = row["answer"]
    gold = "Yes" if answer else "No"
    prompt = f"Passage: {passage}\n\nQuestion: {question}\nAnswer:"
    return {
        "id": f"boolq_{split}_{idx:05d}",
        "question": question,
        "passage": passage,
        "answer": answer,
        "prompt": prompt,
        "gold_answer": gold,
        "distractor": "No" if answer else "Yes",
        "candidates": ["Yes", "No"],
        "split": split,
        "has_negation": bool(NEG_PATTERN.search(question)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare BoolQ JSONL files used by raw and PMI BoolQ evaluation. "
            "This command downloads/loads the BoolQ dataset only after CLI "
            "arguments are parsed, so --help is safe in offline environments."
        )
    )
    parser.add_argument(
        "--neg-output",
        default=str(NEG_OUTPUT),
        help="Output JSONL for negation questions from train+validation.",
    )
    parser.add_argument(
        "--full-output",
        default=str(FULL_OUTPUT),
        help="Output JSONL for the full validation set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neg_output = Path(args.neg_output)
    full_output = Path(args.full_output)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"datasets not available: {exc}", file=sys.stderr)
        sys.exit(1)

    neg_output.parent.mkdir(parents=True, exist_ok=True)
    full_output.parent.mkdir(parents=True, exist_ok=True)

    # Negation subset: train + validation
    neg_records = []
    for split in ["train", "validation"]:
        ds = load_dataset("boolq", split=split)
        for idx, row in enumerate(ds):
            rec = make_record(row, split, idx)
            if rec["has_negation"]:
                neg_records.append(rec)

    with neg_output.open("w", encoding="utf-8") as fh:
        for rec in neg_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"BoolQ negation subset (train+val): {len(neg_records)} questions → {neg_output}")

    # Full validation set
    ds_val = load_dataset("boolq", split="validation")
    full_records = [make_record(row, "validation", idx) for idx, row in enumerate(ds_val)]
    with full_output.open("w", encoding="utf-8") as fh:
        for rec in full_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"BoolQ full validation: {len(full_records)} questions → {full_output}")


if __name__ == "__main__":
    main()
