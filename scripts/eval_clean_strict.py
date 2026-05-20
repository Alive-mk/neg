from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "--models": "configs/model_config.json",
    "--input": "data/processed/validated_largetest_v2_clean_strict.jsonl",
    "--output": "outputs/eval_clean_strict.json",
    "--cache-dir": "outputs/score_cache",
}


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def main() -> int:
    argv = sys.argv[1:]
    forwarded: list[str] = []
    for option, value in DEFAULTS.items():
        if not _has_option(argv, option):
            forwarded.extend([option, value])

    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_models.py"),
        *forwarded,
        *argv,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
