"""Plot E1/E2/E3 mechanistic analysis results.

Usage:
    python scripts/plot_mechanism.py \
        --input outputs/mechanism_qwen_200.json \
        --output outputs/mechanism_qwen_200.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        raise SystemExit(f"matplotlib required: {exc}")

    data = json.load(open(args.input))
    agg = data["aggregate"]
    n = data["num_analyzed_records"]

    e1 = agg["e1"]["per_layer"]
    e2 = agg["e2"]["per_layer"]
    e3 = agg["e3"]["per_layer"]

    layers = [r["layer"] for r in e1]
    L = len(layers)

    # ---- E1 data ----
    neg_margin   = [r["neg_margin_mean"] for r in e1]
    pos_margin   = [r["pos_margin_mean"] for r in e1]
    neg_allowed  = [r["neg_best_allowed_rate"] for r in e1]
    pos_gold     = [r["pos_best_gold_rate"] for r in e1]

    # ---- E2 data (drop NaN) ----
    e2_mean  = [r["mean_attention_to_negation"] if math.isfinite(r["mean_attention_to_negation"]) else 0.0 for r in e2]
    e2_max   = [r["max_head_attention_mean"] if math.isfinite(r["max_head_attention_mean"]) else 0.0 for r in e2]

    # ---- E3 data ----
    e3_recovery = [r["recovery_mean"] for r in e3]
    e3_pos_rate = [r["positive_recovery_rate"] for r in e3]
    e3_allowed  = [r["patched_best_allowed_rate"] for r in e3]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # -------------------------------------------------------
    # E1a: neg_margin vs pos_margin across layers
    # -------------------------------------------------------
    ax1a = fig.add_subplot(gs[0, :])
    ax1a.axhline(0, color="gray", lw=0.8, ls="--")
    ax1a.plot(layers, neg_margin, color="#d62728", marker="o", markersize=3, label="neg_margin (negated prompt)")
    ax1a.plot(layers, pos_margin, color="#1f77b4", marker="s", markersize=3, label="pos_margin (positive prompt)")
    ax1a.fill_between(layers, neg_margin, 0, where=[v < 0 for v in neg_margin], alpha=0.12, color="#d62728", label="_nolegend_")
    ax1a.set_xlabel("Layer", fontsize=11)
    ax1a.set_ylabel("Logit margin (allowed − forbidden)", fontsize=10)
    ax1a.set_title(
        f"E1 — Logit-Lens Signal Evolution (n={n})\n"
        "neg_margin < 0 means forbidden target is beating allowed alternatives at that layer",
        fontsize=11,
    )
    ax1a.legend(fontsize=10)
    ax1a.set_xticks(range(0, L, 2))
    ax1a.grid(axis="y", alpha=0.3)

    # -------------------------------------------------------
    # E1b: allowed / gold rate
    # -------------------------------------------------------
    ax1b = fig.add_subplot(gs[1, 0])
    ax1b.plot(layers, neg_allowed, color="#d62728", marker="o", markersize=3, label="neg best is allowed")
    ax1b.plot(layers, pos_gold,   color="#1f77b4", marker="s", markersize=3, label="pos best is gold")
    ax1b.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax1b.set_xlabel("Layer", fontsize=11)
    ax1b.set_ylabel("Rate", fontsize=10)
    ax1b.set_title("E1 — Best-Candidate Correctness Rate", fontsize=11)
    ax1b.legend(fontsize=9)
    ax1b.set_ylim(0, 1.05)
    ax1b.set_xticks(range(0, L, 4))
    ax1b.grid(axis="y", alpha=0.3)

    # -------------------------------------------------------
    # E2: attention to negation tokens
    # -------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    width = 0.5
    ax2.bar(layers, e2_mean, width=width, color="#2ca02c", alpha=0.75, label="mean across heads")
    ax2.plot(layers, e2_max, color="#ff7f0e", marker="^", markersize=4, lw=1.2, label="max head mean")
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Attention weight", fontsize=10)
    ax2.set_title(
        "E2 — Final Position Attention → Negation Tokens\n"
        "(low = negation token ignored at final prediction step)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.set_xticks(range(0, L, 4))
    ax2.grid(axis="y", alpha=0.3)

    # -------------------------------------------------------
    # E3a: recovery mean
    # -------------------------------------------------------
    ax3a = fig.add_subplot(gs[2, 0])
    ax3a.axhline(0, color="gray", lw=0.8, ls="--")
    ax3a.bar(layers, e3_recovery, color="#9467bd", alpha=0.8)
    ax3a.set_xlabel("Layer (patch source)", fontsize=11)
    ax3a.set_ylabel("Δ neg_margin", fontsize=10)
    ax3a.set_title(
        "E3 — Activation Patching Recovery (neg_margin Δ)\n"
        "patching neg-token hidden state → final position",
        fontsize=10,
    )
    ax3a.set_xticks(range(0, L, 4))
    ax3a.grid(axis="y", alpha=0.3)

    # -------------------------------------------------------
    # E3b: pos_recovery_rate and patched_best_allowed
    # -------------------------------------------------------
    ax3b = fig.add_subplot(gs[2, 1])
    ax3b.plot(layers, e3_pos_rate, color="#9467bd", marker="o", markersize=3, label="positive recovery rate")
    ax3b.plot(layers, e3_allowed, color="#8c564b", marker="s", markersize=3, label="patched best allowed rate")
    # baseline
    baseline_allowed = agg["e1"]["per_layer"][-1]["neg_best_allowed_rate"]
    ax3b.axhline(baseline_allowed, color="gray", lw=0.8, ls="--", label=f"baseline allowed rate ({baseline_allowed:.2f})")
    ax3b.set_xlabel("Layer (patch source)", fontsize=11)
    ax3b.set_ylabel("Rate", fontsize=10)
    ax3b.set_title("E3 — Post-Patch Best-Candidate Rate", fontsize=10)
    ax3b.legend(fontsize=9)
    ax3b.set_ylim(0, 1.05)
    ax3b.set_xticks(range(0, L, 4))
    ax3b.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Negation Blindness — Mechanistic Analysis (Qwen2.5-7B, n=200 probes)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=150)

    # also save PNG
    png_out = out.with_suffix(".png")
    fig.savefig(str(png_out), bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    print(f"Saved: {png_out}")

    # ---- print key stats ----
    print("\n=== Key Stats ===")
    print(f"E1 final layer neg_margin: {neg_margin[-1]:.4f}")
    print(f"E1 final layer pos_margin: {pos_margin[-1]:.4f}")
    print(f"E1 peak neg_margin (best layer): {max(neg_margin):.4f} @ L{neg_margin.index(max(neg_margin))}")
    print(f"E2 mean final-5-layer mean_attn: {sum(e2_mean[-6:-1])/5:.5f}")
    print(f"E3 mean recovery (all layers): {sum(e3_recovery)/len(e3_recovery):.4f}")
    print(f"E3 final layer patched_best_allowed: {e3_allowed[-1]:.3f}")


if __name__ == "__main__":
    main()
