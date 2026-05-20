from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ReaggregateOnReferenceIdsTest(unittest.TestCase):
    def test_reference_subset_preserves_hard_and_multianswer_negrank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clean_input = tmp_path / "clean.jsonl"
            reference_output = tmp_path / "reference.json"
            eval_output = tmp_path / "eval.json"
            reagg_output = tmp_path / "reagg.json"

            write_jsonl(
                clean_input,
                [
                    {
                        "id": "select_in_ref",
                        "semantic_mode": "exclusive_choice",
                        "scope_type": "in_scope",
                        "expected_neg_behavior": "select_gold_neg",
                        "gold_neg": ["hard_gold"],
                        "valid_negatives": ["soft_gold"],
                    },
                    {
                        "id": "select_not_in_ref",
                        "semantic_mode": "exclusive_choice",
                        "scope_type": "in_scope",
                        "expected_neg_behavior": "select_gold_neg",
                        "gold_neg": ["hard_gold"],
                        "valid_negatives": [],
                    },
                    {
                        "id": "preserve_in_ref",
                        "semantic_mode": "contrastive_resolution",
                        "scope_type": "out_of_scope",
                        "expected_neg_behavior": "preserve_positive",
                        "gold_neg": [],
                        "valid_negatives": [],
                    },
                ],
            )
            write_json(
                reference_output,
                {
                    "ref_model": {
                        "per_record": [
                            {"id": "select_in_ref"},
                            {"id": "preserve_in_ref"},
                        ]
                    }
                },
            )
            write_json(
                eval_output,
                {
                    "candidate_model": {
                        "per_record": [
                            {
                                "id": "select_in_ref",
                                "expected_neg_behavior": "select_gold_neg",
                                "pos_correct": True,
                                "neg_best": "soft_gold",
                                "neg_correct": False,
                                "neg_rank_correct": False,
                                "flip_required": True,
                                "flip_correct": False,
                            },
                            {
                                "id": "select_not_in_ref",
                                "expected_neg_behavior": "select_gold_neg",
                                "pos_correct": True,
                                "neg_best": "hard_gold",
                                "neg_correct": True,
                                "neg_rank_correct": True,
                                "flip_required": True,
                                "flip_correct": True,
                            },
                            {
                                "id": "preserve_in_ref",
                                "expected_neg_behavior": "preserve_positive",
                                "pos_correct": True,
                                "neg_best": "positive_gold",
                                "neg_correct": True,
                                "neg_rank_correct": False,
                                "preserve_positive": True,
                                "flip_required": False,
                                "flip_correct": False,
                                "over_negation": False,
                            },
                        ]
                    }
                },
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "reaggregate_on_reference_ids.py"),
                    "--clean-input",
                    str(clean_input),
                    "--reference-output",
                    str(reference_output),
                    "--outputs",
                    str(eval_output),
                    "--output",
                    str(reagg_output),
                ],
                check=True,
                cwd=ROOT,
            )

            payload = json.loads(reagg_output.read_text(encoding="utf-8"))
            result = payload["eval.json:candidate_model"]
            summary = result["summary"]

            self.assertEqual(result["n_records"], 2)
            self.assertEqual(summary["NegRankAcc"]["mean"], 0.0)
            self.assertEqual(summary["MultiAnswerNegRankAcc"]["mean"], 1.0)
            self.assertEqual(summary["SelectAmbiguousRate"]["mean"], 1.0)
            self.assertEqual(summary["ScopeControlAcc"]["mean"], 1.0)
            self.assertEqual(summary["OverNegationRate"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
