from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class ReaggregateCleanE4Test(unittest.TestCase):
    def test_hard_and_multi_answer_negrank_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "reaggregated.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "reaggregate_on_clean_e4.py"),
                    "--clean-input",
                    str(FIXTURES / "reaggregate_clean_input.jsonl"),
                    "--outputs",
                    str(FIXTURES / "reaggregate_eval_output.json"),
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        result = payload["reaggregate_eval_output.json:fixture_model"]
        summary = result["summary"]

        self.assertEqual(result["n_records"], 2)
        self.assertEqual(summary["NegRankAcc"]["mean"], 0.5)
        self.assertEqual(summary["MultiAnswerNegRankAcc"]["mean"], 1.0)
        self.assertEqual(summary["SelectAmbiguousRate"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
