"""
Analyze scalar PMI calibration for BoolQ.

Standard PMI uses score(answer | prompt) - prior(answer).  This script searches
a scalar alpha in score - alpha * prior.  It is diagnostic: alpha must be chosen
on a held-out calibration split before being reported as a deployment setting.
"""
from __future__ import annotations

import argparse
import json
from numbers import Real
import random
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.metrics import bootstrap_ci


LABELS = ("Yes", "No")


def is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def require_probability_record(record: dict, idx: int) -> None:
    missing = [key for key in ("id", "gold_answer", "scores") if key not in record]
    if missing:
        raise ValueError(f"raw record #{idx} is missing required keys: {missing}")
    if record["gold_answer"] not in LABELS:
        raise ValueError(
            f"raw record {record.get('id', idx)!r} has unsupported gold_answer "
            f"{record['gold_answer']!r}; expected one of {LABELS}"
        )
    scores = record["scores"]
    if not isinstance(scores, dict):
        raise ValueError(f"raw record {record['id']!r} has non-object scores")
    missing_scores = [label for label in LABELS if label not in scores]
    if missing_scores:
        raise ValueError(f"raw record {record['id']!r} is missing scores for {missing_scores}")
    non_numeric = [label for label in LABELS if not is_number(scores[label])]
    if non_numeric:
        raise ValueError(f"raw record {record['id']!r} has non-numeric scores for {non_numeric}")


def load_boolq_inputs(raw_eval: Path, pmi_eval: Path, model: str) -> tuple[list[dict], dict[str, float]]:
    raw_payload = json.loads(raw_eval.read_text(encoding="utf-8"))
    pmi_payload = json.loads(pmi_eval.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("--raw-eval must contain a top-level JSON object keyed by model name")
    if not isinstance(pmi_payload, dict):
        raise ValueError("--pmi-eval must contain a top-level JSON object keyed by model name")

    if model not in raw_payload:
        available = ", ".join(sorted(raw_payload)) or "<none>"
        raise ValueError(f"model {model!r} not found in --raw-eval; available: {available}")
    if model not in pmi_payload:
        available = ", ".join(sorted(pmi_payload)) or "<none>"
        raise ValueError(f"model {model!r} not found in --pmi-eval; available: {available}")

    records = raw_payload[model].get("per_record")
    if not isinstance(records, list) or not records:
        raise ValueError(f"--raw-eval model {model!r} must contain a non-empty per_record list")
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"raw record #{idx} is not a JSON object")
        require_probability_record(record, idx)

    prior = pmi_payload[model].get("prior")
    if not isinstance(prior, dict):
        raise ValueError(f"--pmi-eval model {model!r} must contain a prior object")
    missing_prior = [label for label in LABELS if label not in prior]
    if missing_prior:
        raise ValueError(f"--pmi-eval model {model!r} prior is missing {missing_prior}")
    non_numeric_prior = [label for label in LABELS if not is_number(prior[label])]
    if non_numeric_prior:
        raise ValueError(f"--pmi-eval model {model!r} prior has non-numeric values for {non_numeric_prior}")

    return records, {label: float(prior[label]) for label in LABELS}


def validate_args(records: list[dict], args: argparse.Namespace) -> list[int]:
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    if args.max_alpha < 0:
        raise ValueError("--max-alpha must be >= 0")
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if not 0 < args.calibration_ratio < 1:
        raise ValueError("--calibration-ratio must be between 0 and 1")

    counts = {label: sum(1 for record in records if record["gold_answer"] == label) for label in LABELS}
    too_small = {label: count for label, count in counts.items() if count < 2}
    if too_small:
        raise ValueError(
            "held-out calibration requires at least two examples for each label; "
            f"got {too_small}"
        )
    min_label_count = min(counts.values())
    if args.folds > min_label_count:
        raise ValueError(
            f"--folds ({args.folds}) cannot exceed the smallest label count "
            f"({min_label_count}); counts={counts}"
        )

    heldout_seeds = [int(seed.strip()) for seed in args.heldout_seeds.split(",") if seed.strip()]
    if not heldout_seeds:
        raise ValueError("--heldout-seeds must contain at least one integer seed")
    return heldout_seeds


def predict(record: dict, prior: dict[str, float], alpha: float) -> str:
    scores = record["scores"]
    yes = scores["Yes"] - alpha * prior["Yes"]
    no = scores["No"] - alpha * prior["No"]
    return "Yes" if yes >= no else "No"


def accuracy(records: list[dict], prior: dict[str, float], alpha: float) -> float:
    return sum(predict(r, prior, alpha) == r["gold_answer"] for r in records) / len(records)


def best_alpha(records: list[dict], prior: dict[str, float], max_alpha: float, step: float) -> tuple[float, float]:
    n_steps = int(max_alpha / step)
    best = (-1.0, 0.0)
    for i in range(n_steps + 1):
        alpha = round(i * step, 10)
        acc = accuracy(records, prior, alpha)
        if acc > best[0]:
            best = (acc, alpha)
    return best[1], best[0]


def stratified_folds(records: list[dict], k: int, seed: int) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = {"Yes": [], "No": []}
    for record in records:
        buckets[record["gold_answer"]].append(record)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for bucket in buckets.values():
        rng.shuffle(bucket)
        for idx, record in enumerate(bucket):
            folds[idx % k].append(record)
    return folds


def stratified_split(records: list[dict], calibration_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = {"Yes": [], "No": []}
    for record in records:
        buckets[record["gold_answer"]].append(record)
    rng = random.Random(seed)
    calibration_records: list[dict] = []
    test_records: list[dict] = []
    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n_calibration = max(1, round(len(shuffled) * calibration_ratio))
        n_calibration = min(n_calibration, len(shuffled) - 1)
        calibration_records.extend(shuffled[:n_calibration])
        test_records.extend(shuffled[n_calibration:])
    return calibration_records, test_records


def rescore_records(raw_records: list[dict], prior: dict[str, float], alpha: float) -> dict:
    results = []
    for record in raw_records:
        scores = record["scores"]
        predicted_raw = "Yes" if scores["Yes"] >= scores["No"] else "No"
        yes = scores["Yes"] - alpha * prior["Yes"]
        no = scores["No"] - alpha * prior["No"]
        predicted_pmi = "Yes" if yes >= no else "No"
        results.append(
            {
                "id": record["id"],
                "gold_answer": record["gold_answer"],
                "predicted_raw": predicted_raw,
                "predicted_pmi": predicted_pmi,
                "correct_raw": predicted_raw == record["gold_answer"],
                "correct_pmi": predicted_pmi == record["gold_answer"],
            }
        )
    return {
        "summary_raw": {"accuracy": bootstrap_ci([row["correct_raw"] for row in results])},
        "summary_pmi": {"accuracy": bootstrap_ci([row["correct_pmi"] for row in results])},
        "prior": prior,
        "alpha": alpha,
        "per_record": results,
        "source": "offline_rescore_from_saved_scores",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-eval", default="outputs/eval_boolq_full_llama.json")
    parser.add_argument("--pmi-eval", default="outputs/eval_boolq_full_pmi_llama.json")
    parser.add_argument("--model", default="llama_mgnm")
    parser.add_argument("--output", default="outputs/boolq_alpha_calibration_llama_mgnm.json")
    parser.add_argument("--max-alpha", type=float, default=2.5)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--heldout-seeds", default="7,13,29,43,71")
    parser.add_argument("--emit-rescored-output")
    parser.add_argument("--rescore-alpha", type=float)
    args = parser.parse_args()

    try:
        records, prior = load_boolq_inputs(Path(args.raw_eval), Path(args.pmi_eval), args.model)
        heldout_seeds = validate_args(records, args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    full_alpha, full_acc = best_alpha(records, prior, args.max_alpha, args.step)
    standard = {
        "raw_alpha_0": accuracy(records, prior, 0.0),
        "pmi_alpha_1": accuracy(records, prior, 1.0),
        "best_alpha_full_validation": full_alpha,
        "best_alpha_full_validation_accuracy": full_acc,
    }

    folds = stratified_folds(records, args.folds, args.seed)
    cv_rows = []
    for fold_idx, test_records in enumerate(folds):
        train_records = [r for i, fold in enumerate(folds) if i != fold_idx for r in fold]
        alpha, train_acc = best_alpha(train_records, prior, args.max_alpha, args.step)
        test_acc = accuracy(test_records, prior, alpha)
        cv_rows.append(
            {
                "fold": fold_idx,
                "alpha": alpha,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "n_test": len(test_records),
            }
        )

    heldout_rows = []
    for split_seed in heldout_seeds:
        calibration_records, test_records = stratified_split(records, args.calibration_ratio, split_seed)
        alpha, calibration_acc = best_alpha(calibration_records, prior, args.max_alpha, args.step)
        heldout_rows.append(
            {
                "seed": split_seed,
                "alpha": alpha,
                "calibration_accuracy": calibration_acc,
                "test_raw_alpha_0": accuracy(test_records, prior, 0.0),
                "test_pmi_alpha_1": accuracy(test_records, prior, 1.0),
                "test_tuned_alpha": accuracy(test_records, prior, alpha),
                "n_calibration": len(calibration_records),
                "n_test": len(test_records),
            }
        )

    report = {
        "model": args.model,
        "prior": prior,
        "standard": standard,
        "cross_validation": {
            "folds": cv_rows,
            "mean_test_accuracy": mean(row["test_accuracy"] for row in cv_rows),
            "mean_alpha": mean(row["alpha"] for row in cv_rows),
        },
        "heldout_calibration": {
            "calibration_ratio": args.calibration_ratio,
            "splits": heldout_rows,
            "mean_alpha": mean(row["alpha"] for row in heldout_rows),
            "std_alpha": pstdev(row["alpha"] for row in heldout_rows),
            "mean_test_raw_alpha_0": mean(row["test_raw_alpha_0"] for row in heldout_rows),
            "mean_test_pmi_alpha_1": mean(row["test_pmi_alpha_1"] for row in heldout_rows),
            "mean_test_tuned_alpha": mean(row["test_tuned_alpha"] for row in heldout_rows),
            "std_test_tuned_alpha": pstdev(row["test_tuned_alpha"] for row in heldout_rows),
        },
        "note": "Cross-validation is diagnostic; held-out calibration selects alpha only on calibration splits before testing.",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.emit_rescored_output:
        rescore_alpha = args.rescore_alpha
        if rescore_alpha is None:
            rescore_alpha = round(report["heldout_calibration"]["mean_alpha"], 2)
        rescored = {
            args.model: rescore_records(records, prior, rescore_alpha)
        }
        rescored_path = Path(args.emit_rescored_output)
        rescored_path.parent.mkdir(parents=True, exist_ok=True)
        rescored_path.write_text(json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{args.model}: raw={standard['raw_alpha_0']*100:.2f}% "
        f"PMI(alpha=1)={standard['pmi_alpha_1']*100:.2f}% "
        f"best_alpha={full_alpha:.2f} full={full_acc*100:.2f}% "
        f"cv={report['cross_validation']['mean_test_accuracy']*100:.2f}%"
    )
    print(
        f"heldout: alpha={report['heldout_calibration']['mean_alpha']:.2f}±"
        f"{report['heldout_calibration']['std_alpha']:.2f} "
        f"raw={report['heldout_calibration']['mean_test_raw_alpha_0']*100:.2f}% "
        f"PMI(alpha=1)={report['heldout_calibration']['mean_test_pmi_alpha_1']*100:.2f}% "
        f"tuned={report['heldout_calibration']['mean_test_tuned_alpha']*100:.2f}%"
    )
    print(f"Saved: {out_path}")
    if args.emit_rescored_output:
        print(f"Saved rescored eval: {args.emit_rescored_output}")


if __name__ == "__main__":
    main()
