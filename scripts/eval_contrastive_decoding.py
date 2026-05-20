"""Contrastive Decoding baseline on E4 v2 test set.

Expert model  : Qwen2.5-7B (base, no adapter)
Amateur model : Qwen2.5-0.5B (smaller, same family)
CD score      : log p_expert(c|prompt) - alpha * log p_amateur(c|prompt)

Li et al. (2022) "Contrastive Decoding: Open-ended Text Generation as Optimization"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, join_prompt_and_continuation
from neg_blindness.evaluation import build_candidate_sets
from neg_blindness.io_utils import load_records, write_json
from neg_blindness.metrics import bootstrap_ci


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(model_path: str, dtype=torch.float16):
    print(f"  Loading {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return tok, model


def score_continuation_local(
    model, tokenizer, prompt: str, continuation: str
) -> float:
    full_text = join_prompt_and_continuation(prompt, continuation)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids   = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = int(prompt_ids.shape[1])
    full_len   = int(full_ids.shape[1])
    if full_len <= prompt_len:
        return -1e9

    device = next(model.parameters()).device
    input_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, prompt_len - 1 : -1, :]
        log_probs = logits.log_softmax(dim=-1)
        targets = input_ids[:, prompt_len:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.mean().item())


def cd_score(
    expert_model, expert_tok, amateur_model, amateur_tok,
    prompt: str, continuation: str, alpha: float,
    cache_expert: ScoreCache, cache_amateur: ScoreCache,
) -> float:
    from hashlib import sha1
    cache_key = sha1(f"{prompt}|||{continuation}".encode()).hexdigest()

    cached_e = cache_expert.get(cache_key)
    if cached_e is None:
        cached_e = score_continuation_local(expert_model, expert_tok, prompt, continuation)
        cache_expert.set(cache_key, cached_e)

    cached_a = cache_amateur.get(cache_key)
    if cached_a is None:
        cached_a = score_continuation_local(amateur_model, amateur_tok, prompt, continuation)
        cache_amateur.set(cache_key, cached_a)

    return float(cached_e) - alpha * float(cached_a)


# ── eval loop ─────────────────────────────────────────────────────────────────

def evaluate(
    records,
    expert_model, expert_tok,
    amateur_model, amateur_tok,
    alpha: float,
    cache_dir: str,
) -> dict:
    cache_e = ScoreCache(cache_dir, "cd_expert")
    cache_a = ScoreCache(cache_dir, "cd_amateur")

    per_record = []
    for i, record in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(records)}", flush=True)

        pos_cands, neg_cands = build_candidate_sets(record)

        # Positive prompt ranking (expert only — no CD needed for pos)
        pos_scores = {
            c: score_continuation_local(expert_model, expert_tok, record.prompt_pos, c)
            for c in pos_cands
        }
        pos_best = max(pos_scores, key=pos_scores.__getitem__)
        pos_correct = pos_best in set(record.gold_pos)

        # Negative prompt ranking via CD
        neg_scores = {
            c: cd_score(
                expert_model, expert_tok, amateur_model, amateur_tok,
                record.prompt_neg, c, alpha, cache_e, cache_a,
            )
            for c in neg_cands
        }
        neg_best = max(neg_scores, key=neg_scores.__getitem__)

        effective_forbidden = (
            record.forbidden_neg if record.forbidden_neg
            else (record.gold_pos if record.expected_neg_behavior == "suppress_target" else [])
        )
        forbidden_sc = [neg_scores[x] for x in effective_forbidden if x in neg_scores]
        allowed_sc   = [s for x, s in neg_scores.items() if x not in set(effective_forbidden)]
        neg_suppressed = bool(allowed_sc) and bool(forbidden_sc) and max(allowed_sc) > max(forbidden_sc)
        neg_rank_correct  = neg_best in set(record.gold_neg)
        preserve_positive = neg_best in set(record.gold_pos)

        if record.expected_neg_behavior == "select_gold_neg":
            neg_correct = neg_rank_correct
        elif record.expected_neg_behavior == "preserve_positive":
            neg_correct = preserve_positive
        else:
            neg_correct = neg_suppressed

        flip_required = record.expected_neg_behavior in {"suppress_target", "select_gold_neg"}
        flip_correct  = flip_required and pos_correct and neg_correct
        over_negation = (record.expected_neg_behavior == "preserve_positive") and (not preserve_positive)

        per_record.append({
            "id": record.id,
            "expected_neg_behavior": record.expected_neg_behavior,
            "pos_correct": pos_correct,
            "neg_correct": neg_correct,
            "flip_correct": flip_correct,
            "flip_required": flip_required,
            "over_negation": over_negation,
            "preserve_positive": preserve_positive,
            "neg_suppressed": neg_suppressed,
            "neg_rank_correct": neg_rank_correct,
        })

    flip_values = [r["flip_correct"] for r in per_record if r["flip_required"]]
    scope_values = [r["preserve_positive"] for r in per_record if r["expected_neg_behavior"] == "preserve_positive"]
    neg_rank_values = [r["neg_rank_correct"] for r in per_record if r["expected_neg_behavior"] == "select_gold_neg"]
    over_neg_values = [r["over_negation"] for r in per_record if r["expected_neg_behavior"] == "preserve_positive"]

    summary = {
        "count": len(per_record),
        "alpha": alpha,
        "FlipAcc": bootstrap_ci(flip_values),
        "ScopeControlAcc": bootstrap_ci(scope_values),
        "NegRankAcc": bootstrap_ci(neg_rank_values),
        "OverNegationRate": bootstrap_ci(over_neg_values),
    }
    fa = summary["FlipAcc"]["mean"] * 100
    sc = summary["ScopeControlAcc"]["mean"] * 100
    nr = summary["NegRankAcc"]["mean"] * 100
    print(f"    α={alpha:.2f}  FlipAcc={fa:.1f}%  ScopeCtrl={sc:.1f}%  NegRank={nr:.1f}%", flush=True)
    return {"summary": summary, "per_record": per_record}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-path",  required=True)
    parser.add_argument("--amateur-path", required=True)
    parser.add_argument("--input",  default="outputs/test_v2.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="outputs/score_cache_cd")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Amateur downweight coefficient (default 0.5)")
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    print("Loading expert model...")
    expert_tok, expert_model = load_model(args.expert_path)
    print("Loading amateur model...")
    amateur_tok, amateur_model = load_model(args.amateur_path)

    print(f"\nRunning CD (α={args.alpha})...")
    result = evaluate(
        records, expert_model, expert_tok, amateur_model, amateur_tok,
        alpha=args.alpha, cache_dir=args.cache_dir,
    )

    write_json(args.output, {"qwen_cd": result})
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
