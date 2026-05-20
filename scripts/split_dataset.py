from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.io_utils import dump_records, load_records, write_json
from neg_blindness.splitting import split_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.input)
    split_map, report = split_records(
        records,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in split_map.items():
        dump_records(output_dir / f"{split_name}.jsonl", split_items)
    write_json(args.report, report)


if __name__ == "__main__":
    main()

