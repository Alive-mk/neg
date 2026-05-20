"""Three-way mechanistic comparison plot for the paper (Baseline / core_5ep / bal_full_3ep).

Usage:
    python scripts/plot_mechanism_compare.py \
        --inputs  outputs/mechanism_qwen_200.json \
                  outputs/mechanism_e4_core_5ep.json \
                  outputs/mechanism_e4_bal_full_3ep.json \
        --labels  "Baseline" "core_5ep" "bal_full_3ep" \
        --output  outputs/mechanism_e4_comparison_full
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def load(path: str) -> dict:
    d = json.load(open(path))
    return d["aggregate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        raise SystemExit(f"matplotlib required: {exc}")

    datasets = [(args.labels[i], load(args.inputs[i])) for i in range(len(args.inputs))]
    n_models = len(datasets)
    # Each model may have different layer counts; use normalized 0-1 x-axis
    max_layers = max(len(agg["e1"]["per_layer"]) for _, agg in datasets)
    # For xtick labels we use the largest model's layer indices
    layers = list(range(max_layers))
    L = max_layers

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # ── E1a: neg_margin across layers ───────────────────────────────────────
    ax1a = fig.add_subplot(gs[0, :])
    ax1a.axhline(0, color="gray", lw=0.8, ls="--")
    for idx, (label, agg) in enumerate(datasets):
        model_layers = [r["layer"] for r in agg["e1"]["per_layer"]]
        neg_m = [r["neg_margin_mean"] for r in agg["e1"]["per_layer"]]
        ax1a.plot(model_layers, neg_m, color=COLORS[idx], marker="o", markersize=3,
                  lw=1.6, label=f"{label} — neg_margin")
    ax1a.set_xlabel("Layer", fontsize=11)
    ax1a.set_ylabel("neg_margin (allowed − forbidden)", fontsize=10)
    ax1a.set_title(
        "E1 — Negation Signal Evolution Across Layers\n"
        "(margin < 0: forbidden target outscores allowed alternatives)",
        fontsize=11,
    )
    ax1a.legend(fontsize=10)
    ax1a.set_xticks(range(0, L, 2))
    ax1a.grid(axis="y", alpha=0.3)

    # ── E1b: neg_best_allowed_rate ───────────────────────────────────────────
    ax1b = fig.add_subplot(gs[1, 0])
    for idx, (label, agg) in enumerate(datasets):
        model_layers = [r["layer"] for r in agg["e1"]["per_layer"]]
        allowed = [r["neg_best_allowed_rate"] for r in agg["e1"]["per_layer"]]
        ax1b.plot(model_layers, allowed, color=COLORS[idx], marker="o", markersize=3,
                  lw=1.4, label=label)
    ax1b.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax1b.set_xlabel("Layer", fontsize=11)
    ax1b.set_ylabel("Rate", fontsize=10)
    ax1b.set_title("E1 — neg_best_allowed_rate (higher = better suppression)", fontsize=11)
    ax1b.legend(fontsize=9)
    ax1b.set_ylim(0, 1.05)
    ax1b.set_xticks(range(0, L, 4))
    ax1b.grid(axis="y", alpha=0.3)

    # ── E2: attention to negation tokens ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    for idx, (label, agg) in enumerate(datasets):
        model_layers = [r["layer"] for r in agg["e2"]["per_layer"]]
        e2_max = [
            r["max_head_attention_mean"] if math.isfinite(r["max_head_attention_mean"]) else 0.0
            for r in agg["e2"]["per_layer"]
        ]
        ax2.plot(model_layers, e2_max, color=COLORS[idx], marker="^", markersize=3,
                 lw=1.4, label=f"{label} — max head")
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Max-head attention weight", fontsize=10)
    ax2.set_title(
        "E2 — Final Position Attention → Negation Token\n"
        "(max head mean per layer)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.set_xticks(range(0, L, 4))
    ax2.grid(axis="y", alpha=0.3)

    # ── E3a: positive_recovery_rate ──────────────────────────────────────────
    ax3a = fig.add_subplot(gs[2, 0])
    for idx, (label, agg) in enumerate(datasets):
        model_layers = [r["layer"] for r in agg["e3"]["per_layer"]]
        pos_rate = [r["positive_recovery_rate"] for r in agg["e3"]["per_layer"]]
        ax3a.plot(model_layers, pos_rate, color=COLORS[idx], marker="o", markersize=3,
                  lw=1.4, label=label)
    ax3a.set_xlabel("Layer (patch source)", fontsize=11)
    ax3a.set_ylabel("Positive recovery rate", fontsize=10)
    ax3a.set_title(
        "E3 — Activation Patching: Positive Recovery Rate\n"
        "(higher last-layer = stronger negation signal at source)",
        fontsize=10,
    )
    ax3a.legend(fontsize=9)
    ax3a.set_ylim(0, 1.05)
    ax3a.set_xticks(range(0, L, 4))
    ax3a.grid(axis="y", alpha=0.3)

    # ── E3b: patched_best_allowed_rate ───────────────────────────────────────
    ax3b = fig.add_subplot(gs[2, 1])
    for idx, (label, agg) in enumerate(datasets):
        model_layers = [r["layer"] for r in agg["e3"]["per_layer"]]
        allowed = [r["patched_best_allowed_rate"] for r in agg["e3"]["per_layer"]]
        ax3b.plot(model_layers, allowed, color=COLORS[idx], marker="s", markersize=3,
                  lw=1.4, label=label)
    ax3b.set_xlabel("Layer (patch source)", fontsize=11)
    ax3b.set_ylabel("Patched best-allowed rate", fontsize=10)
    ax3b.set_title("E3 — Post-Patch Best-Candidate Rate", fontsize=10)
    ax3b.legend(fontsize=9)
    ax3b.set_ylim(0, 1.05)
    ax3b.set_xticks(range(0, L, 4))
    ax3b.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Negation Blindness — Mechanistic Comparison (Qwen2.5-7B, n=200 probes)\n"
        "E4.5: Post-Mitigation Mechanism Check",
        fontsize=13, fontweight="bold", y=0.99,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        save_path = out.with_suffix(suffix)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n=== E4.5 Mechanism Comparison Table ===")
    print(f"{'Model':<25} {'E1_neg_margin_last':>20} {'E2_maxhead_last':>17} {'E3_pos_rec_last':>17}")
    print("-" * 82)
    for label, agg in datasets:
        e1_last = agg["e1"]["per_layer"][-1]["neg_margin_mean"]
        e2_last = agg["e2"]["per_layer"][-1]["max_head_attention_mean"]
        e3_last = agg["e3"]["per_layer"][-1]["positive_recovery_rate"]
        print(f"{label:<25} {e1_last:>20.4f} {e2_last:>17.4f} {e3_last:>17.3f}")


if __name__ == "__main__":
    main()
