"""Merge two mechanism analysis JSON files (split halves) into one."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import mean, median
from neg_blindness.mechanistic import aggregate_mechanistic_results
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_per_record = []
    model_name = None
    input_path = None
    for path in args.inputs:
        d = json.load(open(path))
        all_per_record.extend(d["per_record"])
        model_name = d["model_name"]
        input_path = d["input_path"]

    merged = {
        "model_name": model_name,
        "input_path": input_path,
        "num_input_records": len(all_per_record),
        "num_analyzed_records": len(all_per_record),
        "num_failures": 0,
        "failures": [],
        "aggregate": aggregate_mechanistic_results(all_per_record),
        "per_record": all_per_record,
    }
    json.dump(merged, open(args.output, "w"), ensure_ascii=False)
    print(f"Merged {len(all_per_record)} records -> {args.output}")


if __name__ == "__main__":
    main()
