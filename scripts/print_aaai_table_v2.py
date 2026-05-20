"""Print the full AAAI comparison table on v2 test set (525 records).

Columns: NegRank | FlipAcc | ScopeCtrl | OverNeg | WikiFact | BoolQ† | MMLU Δ
† BoolQ: raw accuracy on full val (3270).  PMI-calibrated in parentheses where
  the prior shift is large (|No_bias| > 0.5): MGNM, DPO.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def g(d, m):
    return round(d.get(m, {}).get("mean", 0) * 100, 1) if d else "—"


def load_summary(path, key=None):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if key and key in d:
        v = d[key]
    else:
        v = next(iter(d.values()))
    return v.get("summary") or v.get("aggregate") or {}


def load_mmlu(path):
    if not os.path.exists(path):
        return "—"
    d = json.load(open(path))
    delta = round(d.get("delta", {}).get("overall_accuracy", 0) * 100, 2)
    return f"{delta:+.2f}"


def load_wikifact(path):
    if not os.path.exists(path):
        return "—"
    s = load_summary(path)
    return g(s, "FlipAcc") if s else "—"


def load_boolq(boolq_pmi_file, boolq_key):
    """Return raw acc, with PMI in parens when the model has strong No-bias (>0.5).

    boolq_pmi_file: outputs/eval_boolq_full_pmi_{group}.json
    raw file:       outputs/eval_boolq_full_{group}.json  (same path, no _pmi)
    """
    if not boolq_key:
        return "—"
    raw_file = boolq_pmi_file.replace("_full_pmi_", "_full_")
    pmi_file = boolq_pmi_file

    raw_acc = "—"
    if os.path.exists(raw_file):
        d = json.load(open(raw_file))
        if boolq_key in d:
            s = d[boolq_key].get("summary", {})
            raw_acc = f"{s.get('accuracy',{}).get('mean',0)*100:.1f}"

    pmi_acc = None
    if os.path.exists(pmi_file):
        d = json.load(open(pmi_file))
        if boolq_key in d:
            rec = d[boolq_key]
            prior = rec.get("prior", {})
            no_bias = prior.get("No", 0) - prior.get("Yes", 0)
            # Only show PMI when model is biased toward No (training-induced suppression)
            if no_bias > 0.5:
                s = rec.get("summary_pmi", {})
                pmi_val = s.get("accuracy", {}).get("mean", 0) * 100
                pmi_acc = f"{pmi_val:.1f}"

    if pmi_acc:
        return f"{raw_acc}({pmi_acc})"
    return raw_acc


def load_sig(path, metric="FlipAcc"):
    if not os.path.exists(path):
        return "—"
    d = json.load(open(path))
    if metric not in d:
        return "—"
    r = d[metric]
    delta = round(r.get("obs_diff", 0) * 100, 1)
    p = r.get("mcnemar_p", 1.0)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return f"{delta:+.1f}pp {sig}"


V2 = "outputs/eval_{}_v2.json"
PROMPT_V2 = "outputs/eval_prompt_baseline_v2.json"
BOOLQ_Q = "outputs/eval_boolq_full_pmi_qwen.json"
BOOLQ_L = "outputs/eval_boolq_full_pmi_llama.json"

# Row tuple: (label, lt_path, lt_key, mmlu_path, wf_path, boolq_key)
# ── Qwen models (v2) ──────────────────────────────────────────────────────
QWEN_ROWS = [
    ("Base (no fine-tuning)", PROMPT_V2,                          "none",
     None,                                 None,                              "qwen_base"),
    ("Prompt+Warning",        PROMPT_V2,                          "warning",
     None,                                 None,                              None),
    ("Prompt+Persona",        PROMPT_V2,                          "persona",
     None,                                 None,                              None),
    ("Prompt+CoT",            PROMPT_V2,                          "cot",
     None,                                 None,                              None),
    ("Contrastive Dec.",      "outputs/eval_cd_qwen.json",        "qwen_cd",
     None,                                 None,                              None),
    ("Vanilla SFT",           V2.format("qwen_vanilla"),          None,
     "outputs/mmlu_vanilla_3ep.json",      None,                              None),
    ("DPO",                   V2.format("qwen_dpo"),              "qwen_dpo",
     "outputs/mmlu_qwen_dpo.json",         "outputs/eval_qwen_dpo_wikifact.json",    "qwen_dpo"),
    ("NC-SFT",                V2.format("qwen_nc_sft"),           "qwen_nc_sft",
     "outputs/mmlu_qwen_nc_sft.json",      "outputs/eval_qwen_nc_sft_wikifact.json", "qwen_nc_sft"),
    ("MGNM (ours)",           V2.format("qwen_mgnm"),             "qwen_mgnm",
     "outputs/mmlu_qwen_rd30_os3.json",    "outputs/eval_qwen_rd30_os3_wikifact.json","qwen_mgnm"),
]

# ── Llama models (v2) ─────────────────────────────────────────────────────
LLAMA_ROWS = [
    ("Base (no fine-tuning)", V2.format("llama_base"),            None,
     None,                                 None,                              "llama_base"),
    ("Vanilla SFT",           V2.format("llama_vanilla"),         None,
     "outputs/mmlu_llama_vanilla_3ep.json",None,                              None),
    ("DPO",                   V2.format("llama_dpo"),             "llama_dpo",
     "outputs/mmlu_llama_dpo.json",        "outputs/eval_llama_dpo_wikifact.json",   "llama_dpo"),
    ("NC-SFT",                V2.format("llama_nc_sft"),          "llama_nc_sft",
     "outputs/mmlu_llama_nc_sft.json",     "outputs/eval_llama_nc_sft_wikifact.json","llama_nc_sft"),
    ("MGNM (ours)",           V2.format("llama_mgnm"),            "llama_mgnm",
     "outputs/mmlu_llama_rd10_os2.json",   "outputs/eval_llama_rd10_os2_wikifact.json","llama_mgnm"),
]

# ── Mistral models ────────────────────────────────────────────────────────
MISTRAL_ROWS = [
    ("MGNM (ours)",           V2.format("mistral_lr5e5_os1"),     "mistral_lr5e5_os1",
     "outputs/mmlu_mistral_lr5e5_os1.json","outputs/eval_mistral_lr5e5_os1_wikifact.json", None),
]

# ── Ablation rows (no BoolQ key) ─────────────────────────────────────────
QWEN_ABL_ROWS = [
    ("MGNM (full)",  V2.format("qwen_mgnm"),   "qwen_mgnm",   None, None, None),
    ("w/o L_sup",    V2.format("qwen_nosup"),  "qwen_nosup",  None, None, None),
    ("w/o L_pre",    V2.format("qwen_nopre"),  "qwen_nopre",  None, None, None),
]
LLAMA_ABL_ROWS = [
    ("MGNM (full)",  V2.format("llama_mgnm"),   "llama_mgnm",   None, None, None),
    ("w/o L_sup",    V2.format("llama_nosup"),  "llama_nosup",  None, None, None),
    ("w/o L_pre",    V2.format("llama_nopre"),  "llama_nopre",  None, None, None),
]


def print_table(title, rows, show_wikifact=True, boolq_file=None):
    W = 101 if (show_wikifact and boolq_file) else (87 if show_wikifact else 54)
    print(f"\n{'='*W}")
    print(f"  {title}  (test set: v2, 525 records)")
    print(f"{'='*W}")
    cols = f"{'Method':<22} {'NegRank':>7} {'FlipAcc':>7} {'ScopeCtrl':>9} {'OverNeg':>7}"
    if show_wikifact:
        cols += f" {'WikiFact':>8} {'MMLU Δ':>7}"
    if boolq_file:
        cols += f" {'BoolQ†':>13}"
    print(cols)
    print("-" * W)
    for row_data in rows:
        label, lt_path, lt_key, mmlu_path, wf_path = row_data[:5]
        boolq_key = row_data[5] if len(row_data) > 5 else None
        s = load_summary(lt_path, lt_key)
        nr = g(s, "NegRankAcc")
        fa = g(s, "FlipAcc")
        sc = g(s, "ScopeControlAcc")
        on = g(s, "OverNegationRate")
        avail = "✓" if s else " "
        row = f"{avail}{label:<21} {nr:>7} {fa:>7} {sc:>9} {on:>7}"
        if show_wikifact:
            mm = load_mmlu(mmlu_path) if mmlu_path else "—"
            wf = load_wikifact(wf_path) if wf_path else "—"
            row += f" {wf:>8} {mm:>7}"
        if boolq_file:
            bq = load_boolq(boolq_file, boolq_key)
            row += f" {bq:>13}"
        print(row)
    if boolq_file:
        print(f"  † BoolQ full val (3270). Format: raw(PMI) when |prior No-bias|>0.5.")


print_table("Qwen2.5-7B — Main Results",       QWEN_ROWS,    boolq_file=BOOLQ_Q)
print_table("Llama-3.1-8B — Main Results",     LLAMA_ROWS,   boolq_file=BOOLQ_L)
print_table("Mistral-7B — Cross-Model Transfer", MISTRAL_ROWS)
print_table("Qwen2.5-7B — Ablation Study",     QWEN_ABL_ROWS, show_wikifact=False)
print_table("Llama-3.1-8B — Ablation Study",   LLAMA_ABL_ROWS, show_wikifact=False)

# ── Significance tests (v2) ──────────────────────────────────────────────
print(f"\n{'='*101}")
print("  Significance Tests — MGNM vs Baselines (v2)")
print(f"{'='*101}")

sig_pairs = [
    ("Qwen MGNM vs DPO",     "outputs/stats_sig_v2_qwen_mgnm_vs_dpo.json"),
    ("Qwen MGNM vs NC-SFT",  "outputs/stats_sig_v2_qwen_mgnm_vs_ncsft.json"),
    ("Llama MGNM vs DPO",    "outputs/stats_sig_v2_llama_mgnm_vs_dpo.json"),
    ("Llama MGNM vs NC-SFT", "outputs/stats_sig_v2_llama_mgnm_vs_ncsft.json"),
    ("Qwen MGNM vs nosup",   "outputs/stats_sig_qwen_mgnm_vs_qwen_nosup_v2.json"),
    ("Qwen MGNM vs nopre",   "outputs/stats_sig_qwen_mgnm_vs_qwen_nopre_v2.json"),
    ("Llama MGNM vs nosup",  "outputs/stats_sig_llama_mgnm_vs_llama_nosup_v2.json"),
    ("Llama MGNM vs nopre",  "outputs/stats_sig_llama_mgnm_vs_llama_nopre_v2.json"),
]
print(f"  {'Comparison':<28} FlipAcc Δ          ScopeCtrl Δ")
print("  " + "-"*70)
for label, path in sig_pairs:
    fa_sig = load_sig(path, "FlipAcc")
    sc_sig = load_sig(path, "ScopeControlAcc")
    print(f"  {label:<28} {fa_sig:<20} {sc_sig}")
