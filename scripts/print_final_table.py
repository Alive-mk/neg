"""Print the complete paper results table combining all experiments.

Usage:
    python scripts/print_final_table.py
"""
from __future__ import annotations
import json
import os

RESULTS = [
    # (file, model_key, display_label, section)
    ("outputs/eval_bal_full_3ep_largetest.json", "qwen2_5_7b",      "Base (Qwen2.5-7B)",     "baseline"),
    ("outputs/eval_prompt_baseline_largetest.json", "prompt_warning","Prompt-Warning",         "baseline"),
    ("outputs/eval_prompt_baseline_largetest.json", "prompt_persona","Prompt-Persona",         "baseline"),
    ("outputs/eval_vanilla_3ep_largetest.json",  "qwen2_5_7b_vanilla","Vanilla SFT",           "ablation"),
    ("outputs/eval_ablation_largetest.json",     "mgnm_nosup",       "MGNM w/o L_sup",        "ablation"),
    ("outputs/eval_ablation_largetest.json",     "mgnm_nopre",       "MGNM w/o L_preserve",   "ablation"),
    ("outputs/eval_lsup2_largetest.json",        "qwen2_5_7b_e4",    "MGNM (lsup2, ours)",    "main"),
]

WIKIFACT_RESULTS = [
    ("outputs/eval_wikifact_neg.json", "qwen2_5_7b",   "Base"),
    ("outputs/eval_wikifact_neg.json", "qwen_vanilla", "Vanilla SFT"),
    ("outputs/eval_wikifact_neg.json", "qwen_lsup2",   "MGNM (ours)"),
]


def get_metrics(path, key):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if key not in d:
        return None
    sub = d[key]
    s = sub.get("summary", sub.get("aggregate", {}).get("summary", {}))
    return {k: v.get("mean", 0) * 100 for k, v in s.items() if isinstance(v, dict) and "mean" in v}


def fmt_row(label, m, width=26):
    if m is None:
        return f"{label:<{width}} {'N/A':>8}"
    return (f"{label:<{width}} "
            f"{m.get('PosAcc',0):>7.1f}  "
            f"{m.get('NegSuppRate',0):>8.1f}  "
            f"{m.get('FlipAcc',0):>6.1f}  "
            f"{m.get('ScopeControlAcc',0):>9.1f}  "
            f"{m.get('DoubleNegationAcc',0):>9.1f}  "
            f"{m.get('OverNegationRate',0):>8.1f}")


def header(width=26):
    return (f"{'Method':<{width}} "
            f"{'PosAcc':>7}  {'NegSupp':>8}  {'Flip':>6}  "
            f"{'ScopeCtrl':>9}  {'DoubleNeg':>9}  {'OverNeg':>8}")


print("=" * 95)
print("MAIN TABLE: large_test (n=392)")
print("=" * 95)
print(header())
print("-" * 95)
for path, key, label, section in RESULTS:
    m = get_metrics(path, key)
    print(fmt_row(label, m))

print()
print("=" * 60)
print("EXTERNAL BENCHMARK: WikiFact-Neg (n=100, NegSuppRate only)")
print("=" * 60)
print(f"{'Method':<20} {'NegSuppRate':>12} {'PosAcc':>8}")
print("-" * 44)
for path, key, label in WIKIFACT_RESULTS:
    if not os.path.exists(path):
        print(f"{label:<20} {'N/A':>12}")
        continue
    d = json.load(open(path))
    if key not in d:
        print(f"{label:<20} {'key missing':>12}")
        continue
    s = d[key].get("summary", {})
    neg = s.get("NegSuppRate", {}).get("mean", 0) * 100
    pos = s.get("PosAcc", {}).get("mean", 0) * 100
    print(f"{label:<20} {neg:>12.1f}% {pos:>8.1f}%")
