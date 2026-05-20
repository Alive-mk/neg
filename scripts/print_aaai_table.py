"""Print the full AAAI comparison table once all baselines are available."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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

# ── Qwen models ────────────────────────────────────────────────────────────
QWEN_ROWS = [
    ("Base (Qwen)",        "outputs/eval_prompt_baseline_largetest.json", "none",           None,                             None),
    ("Prompt+Warning",     "outputs/eval_prompt_baseline_largetest.json", "warning",        None,                             None),
    ("Prompt+Persona",     "outputs/eval_prompt_baseline_largetest.json", "persona",        None,                             None),
    ("Prompt+CoT",         "outputs/eval_cot_baseline_largetest.json",    "cot",            None,                             None),
    ("Vanilla SFT",        "outputs/eval_vanilla_3ep_largetest.json",     None,             "outputs/mmlu_vanilla_3ep.json",  None),
    ("DPO",                "outputs/eval_qwen_dpo_largetest.json",        "qwen_dpo",       "outputs/mmlu_qwen_dpo.json",     "outputs/eval_qwen_dpo_wikifact.json"),
    ("NC-SFT (ours)",      "outputs/eval_qwen_nc_sft_largetest.json",     "qwen_nc_sft",    "outputs/mmlu_qwen_nc_sft.json",  "outputs/eval_qwen_nc_sft_wikifact.json"),
    ("MGNM (ours)",        "outputs/eval_qwen_rd30_os3_largetest.json",   "qwen_rd30_os3",  "outputs/mmlu_qwen_rd30_os3.json","outputs/eval_qwen_rd30_os3_wikifact.json"),
]

# ── Llama models ───────────────────────────────────────────────────────────
LLAMA_ROWS = [
    ("Vanilla SFT",        "outputs/eval_llama_vanilla_3ep_largetest.json", None,              "outputs/mmlu_llama_vanilla_3ep.json", None),
    ("DPO",                "outputs/eval_llama_dpo_largetest.json",         "llama_dpo",       "outputs/mmlu_llama_dpo.json",         "outputs/eval_llama_dpo_wikifact.json"),
    ("NC-SFT (ours)",      "outputs/eval_llama_nc_sft_largetest.json",      "llama_nc_sft",    "outputs/mmlu_llama_nc_sft.json",      "outputs/eval_llama_nc_sft_wikifact.json"),
    ("MGNM (ours)",        "outputs/eval_llama_ret01_largetest.json",        "llama_ret01",     "outputs/mmlu_llama_ret01.json",       "outputs/eval_llama_ret01_wikifact.json"),
]

def print_table(title, rows):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    hdr = f"{'Method':<20} {'NegRank':>7} {'FlipAcc':>7} {'ScopeCtrl':>9} {'OverNeg':>7} {'WikiFact':>8} {'MMLU Δ':>7}"
    print(hdr)
    print("-" * 75)
    for label, lt_path, lt_key, mmlu_path, wf_path in rows:
        s = load_summary(lt_path, lt_key)
        nr = g(s, "NegRankAcc")
        fa = g(s, "FlipAcc")
        sc = g(s, "ScopeControlAcc")
        on = g(s, "OverNegationRate")
        mm = load_mmlu(mmlu_path) if mmlu_path else "—"
        wf = "—"
        if wf_path:
            ws = load_summary(wf_path)
            if ws:
                wf = g(ws, "FlipAcc")
        avail = "✓" if s else " "
        print(f"{avail}{label:<19} {nr:>7} {fa:>7} {sc:>9} {on:>7} {wf:>8} {mm:>7}")

os.chdir(ROOT)
print_table("Qwen2.5-7B", QWEN_ROWS)
print_table("Llama-3.1-8B", LLAMA_ROWS)

# ── Significance summary ───────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  Significance tests (MGNM vs baseline, Qwen)")
print(f"{'='*90}")
sig_files = {
    "vs Vanilla SFT": "outputs/stats_sig_mgnm_vs_vanilla.json",
    "vs DPO":         "outputs/stats_sig_mgnm_vs_dpo.json",
    "vs NC-SFT":      "outputs/stats_sig_mgnm_vs_nc_sft.json",
}
for label, path in sig_files.items():
    if not os.path.exists(path):
        print(f"  {label}: (pending)")
        continue
    d = json.load(open(path))
    def fmt(m):
        return load_sig(path, m)
    print(f"  {label:<20} FlipAcc={fmt('FlipAcc')}  ScopeCtrl={fmt('ScopeControlAcc')}")
