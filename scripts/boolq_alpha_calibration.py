"""
Analyze scalar PMI calibration for BoolQ.

Standard PMI uses score(answer | prompt) - prior(answer).  This script searches
a scalar alpha in score - alpha * prior.  It is diagnostic: alpha must be chosen
on a held-out calibration split before being reported as a deployment setting.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.metrics import bootstrap_ci


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

    raw_payload = json.loads(Path(args.raw_eval).read_text(encoding="utf-8"))
    pmi_payload = json.loads(Path(args.pmi_eval).read_text(encoding="utf-8"))
    records = raw_payload[args.model]["per_record"]
    prior = pmi_payload[args.model]["prior"]

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
    heldout_seeds = [int(seed.strip()) for seed in args.heldout_seeds.split(",") if seed.strip()]
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
            args.model: rescore_records(raw_payload[args.model]["per_record"], prior, rescore_alpha)
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
