from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class ReaggregateOnCleanE4Test(unittest.TestCase):
    def test_hard_and_multi_answer_negrank_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "reaggregate.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/reaggregate_on_clean_e4.py",
                    "--clean-input",
                    str(FIXTURES / "reaggregate_clean_input.jsonl"),
                    "--outputs",
                    str(FIXTURES / "reaggregate_eval_output.json"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            result = payload["reaggregate_eval_output.json:toy_model"]
            summary = result["summary"]

            self.assertEqual(result["n_records"], 2)
            self.assertAlmostEqual(summary["NegRankAcc"]["mean"], 0.5)
            self.assertAlmostEqual(summary["MultiAnswerNegRankAcc"]["mean"], 1.0)
            self.assertAlmostEqual(summary["SelectAmbiguousRate"]["mean"], 0.5)
            self.assertIn("NegRank=50.0 MultiNegRank=100.0", completed.stdout)


if __name__ == "__main__":
    unittest.main()
