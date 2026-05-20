from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from neg_blindness.schema import ExperimentRecord


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict | list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[ExperimentRecord]:
    return [ExperimentRecord.from_dict(item) for item in read_jsonl(path)]


def dump_records(path: str | Path, records: Iterable[ExperimentRecord]) -> None:
    write_jsonl(path, (record.as_dict() for record in records))

