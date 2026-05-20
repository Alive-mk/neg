"""Print BoolQ negation and full-validation results: raw + PMI calibrated."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict | None:
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


def acc(d: dict | None, key: str, summary_key: str = "summary") -> str:
    if d is None or key not in d:
        return "—"
    s = d[key].get(summary_key, {})
    mean = s.get("accuracy", {}).get("mean", 0)
    return f"{mean*100:.1f}%"


ROWS = [
    # (label, qwen_key, llama_key)
    ("Base",        "qwen_base",   "llama_base"),
    ("DPO",         "qwen_dpo",    "llama_dpo"),
    ("NC-SFT",      "qwen_nc_sft", "llama_nc_sft"),
    ("MGNM (ours)", "qwen_mgnm",   "llama_mgnm"),
]

# Load all result files
qneg_raw  = load("outputs/eval_boolq_neg_qwen.json")
lneg_raw  = load("outputs/eval_boolq_neg_llama.json")
qfull_raw = load("outputs/eval_boolq_full_qwen.json")
lfull_raw = load("outputs/eval_boolq_full_llama.json")

qneg_pmi  = load("outputs/eval_boolq_pmi_qwen.json")
lneg_pmi  = load("outputs/eval_boolq_pmi_llama.json")
qfull_pmi = load("outputs/eval_boolq_full_pmi_qwen.json")
lfull_pmi = load("outputs/eval_boolq_full_pmi_llama.json")

W = 11

print(f"\n{'='*95}")
print("  BoolQ Evaluation (Qwen2.5-7B)")
print(f"{'='*95}")
print(f"  {'Method':<14} {'Neg-Raw':>{W}} {'Neg-PMI':>{W}} {'Full-Raw':>{W}} {'Full-PMI':>{W}}")
print("  " + "-" * 56)
for label, qkey, _ in ROWS:
    nr = acc(qneg_raw,  qkey, "summary")
    np = acc(qneg_pmi,  qkey, "summary_pmi")
    fr = acc(qfull_raw, qkey, "summary")
    fp = acc(qfull_pmi, qkey, "summary_pmi")
    print(f"  {label:<14} {nr:>{W}} {np:>{W}} {fr:>{W}} {fp:>{W}}")

print(f"\n{'='*95}")
print("  BoolQ Evaluation (Llama-3.1-8B)")
print(f"{'='*95}")
print(f"  {'Method':<14} {'Neg-Raw':>{W}} {'Neg-PMI':>{W}} {'Full-Raw':>{W}} {'Full-PMI':>{W}}")
print("  " + "-" * 56)
for label, _, lkey in ROWS:
    nr = acc(lneg_raw,  lkey, "summary")
    np = acc(lneg_pmi,  lkey, "summary_pmi")
    fr = acc(lfull_raw, lkey, "summary")
    fp = acc(lfull_pmi, lkey, "summary_pmi")
    print(f"  {label:<14} {nr:>{W}} {np:>{W}} {fr:>{W}} {fp:>{W}}")

# Prior shift summary
print(f"\n{'='*95}")
print("  Prior Shift Analysis (null-prompt log p bias toward No = prior_No - prior_Yes)")
print(f"{'='*95}")
print(f"  {'Model':<22} {'prior_Yes':>10} {'prior_No':>10} {'No bias':>10}  {'Interpretation'}")
print("  " + "-" * 80)
for d_pmi, name in [(qneg_pmi, 'Qwen'), (lneg_pmi, 'Llama')]:
    if d_pmi is None:
        continue
    for key, data in d_pmi.items():
        p = data.get("prior", {})
        yes_p = p.get("Yes", 0)
        no_p  = p.get("No", 0)
        bias  = no_p - yes_p
        interp = "strong No bias (suppression)" if bias > 0.5 else \
                 "moderate No bias" if bias > 0.1 else \
                 "strong Yes bias" if bias < -0.3 else "balanced"
        print(f"  {key:<22} {yes_p:>10.3f} {no_p:>10.3f} {bias:>+10.3f}  {interp}")

# CD on E4 v2 (separate task)
cd = load("outputs/eval_cd_qwen.json")
if cd and "qwen_cd" in cd:
    s = cd["qwen_cd"]["summary"]
    fa = s.get("FlipAcc", {}).get("mean", 0) * 100
    sc = s.get("ScopeControlAcc", {}).get("mean", 0) * 100
    nr = s.get("NegRankAcc", {}).get("mean", 0) * 100
    print(f"\n{'='*95}")
    print("  Contrastive Decoding on E4 v2 (525 records, Qwen-7B expert / Qwen-0.5B amateur)")
    print(f"{'='*95}")
    print(f"  CD (α=0.5):  FlipAcc={fa:.1f}%   ScopeCtrl={sc:.1f}%   NegRank={nr:.1f}%")
    print(f"  (Qwen MGNM:  FlipAcc=67.7%  ScopeCtrl=91.2%  NegRank=63.9%  for comparison)")
