from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reaggregate_on_clean_e4  # noqa: E402


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


class ReaggregateOnCleanE4Test(unittest.TestCase):
    def test_select_hard_and_multianswer_negrank_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            clean_input = tmp / "clean.jsonl"
            source_output = tmp / "scores.json"
            reagg_output = tmp / "reaggregated.json"

            write_jsonl(
                clean_input,
                [
                    {
                        "id": "select-valid-only",
                        "expected_neg_behavior": "select_gold_neg",
                        "semantic_mode": "contrastive_resolution",
                        "scope_type": "in_scope",
                        "gold_neg": ["hard_gold"],
                        "valid_negatives": ["also_valid"],
                    },
                    {
                        "id": "select-hard",
                        "expected_neg_behavior": "select_gold_neg",
                        "semantic_mode": "contrastive_resolution",
                        "scope_type": "in_scope",
                        "gold_neg": ["hard_gold_2"],
                        "valid_negatives": [],
                    },
                ],
            )
            source_output.write_text(
                json.dumps(
                    {
                        "model_a": {
                            "per_record": [
                                {
                                    "id": "select-valid-only",
                                    "expected_neg_behavior": "select_gold_neg",
                                    "pos_correct": True,
                                    "neg_best": "also_valid",
                                    "neg_rank_correct": False,
                                    "neg_correct": False,
                                    "flip_required": True,
                                    "flip_correct": False,
                                    "distractor_reject": True,
                                },
                                {
                                    "id": "select-hard",
                                    "expected_neg_behavior": "select_gold_neg",
                                    "pos_correct": True,
                                    "neg_best": "hard_gold_2",
                                    "neg_rank_correct": True,
                                    "neg_correct": True,
                                    "flip_required": True,
                                    "flip_correct": True,
                                    "distractor_reject": True,
                                },
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            argv = [
                "reaggregate_on_clean_e4.py",
                "--clean-input",
                str(clean_input),
                "--outputs",
                str(source_output),
                "--output",
                str(reagg_output),
            ]
            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                reaggregate_on_clean_e4.main()

            payload = json.loads(reagg_output.read_text(encoding="utf-8"))
            summary = payload["scores.json:model_a"]["summary"]

            self.assertEqual(payload["scores.json:model_a"]["n_records"], 2)
            self.assertAlmostEqual(summary["NegRankAcc"]["mean"], 0.5)
            self.assertAlmostEqual(summary["MultiAnswerNegRankAcc"]["mean"], 1.0)
            self.assertAlmostEqual(summary["SelectAmbiguousRate"]["mean"], 0.5)
            self.assertIn("NegRank=50.0", stdout.getvalue())
            self.assertIn("MultiNegRank=100.0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
