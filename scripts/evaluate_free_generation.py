"""
N1: Free-generation evaluation for suppress_target records.

For each suppress_target record, we give the model the negated prompt
(e.g. "Is metal NOT known for its high conductivity?") and let it generate
a free-text response with greedy decoding. We then check whether the model's
response asserts the forbidden property (negation failure).

Detection heuristic (two-level):
  1. Hard-fail: model's first-token response is 'No' / 'no' (for yes/no questions
     this directly asserts the forbidden fact: "No, metal IS known for conductivity")
  2. Content-fail: generated text contains key content words from the forbidden_neg
     string (case-insensitive, stemmed to root), suggesting the model reproduced
     the forbidden fact.

Free-gen FlipAcc = proportion of records that PASS (neither hard-fail nor content-fail).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_free_generation.py \
      --model-path model/Qwen2.5-7B \
      --adapter-path outputs/e4_qwen_r3v5_lp15_rs3_boost_rd30_os3/adapter \
      --input data/processed/validated_largetest_v2.jsonl \
      --output outputs/eval_free_gen_qwen_mgnm.json \
      --use-behavior-token \
      [--n 100]
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_STOPWORDS = {"the", "a", "an", "is", "not", "for", "its", "it", "to", "of",
              "in", "on", "at", "by", "be", "was", "are", "have", "has", "do"}

_BEHAVIOR_TOKENS = {
    "suppress_target":   "[SUPPRESS]",
    "preserve_positive": "[PRESERVE]",
    "select_gold_neg":   "[SELECT]",
}


def extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 3}


def is_negation_failure(response: str, forbidden_neg: list[str]) -> tuple[bool, str]:
    """Return (failure, reason)."""
    resp_lower = response.lower().strip()

    # Hard-fail: starts with 'no,' / 'no.' / 'no ' → asserts forbidden fact
    if re.match(r"^no[,.\s!]", resp_lower):
        return True, "starts_with_no"

    # Content-fail: forbidden keywords appear in response
    for forbidden in forbidden_neg:
        fkw = extract_keywords(forbidden)
        resp_kw = extract_keywords(response)
        # Require ≥60% keyword overlap (robust to paraphrasing)
        if fkw and len(fkw & resp_kw) / len(fkw) >= 0.6:
            return True, f"keyword_overlap({fkw & resp_kw})"

    return False, ""


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        trust_remote_code=True, device_map=device
    )
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    records = [json.loads(l) for l in Path(args.input).open()]
    records = [r for r in records if r.get("expected_neg_behavior") == "suppress_target"]
    if args.n:
        records = records[:args.n]
    print(f"Evaluating {len(records)} suppress_target records")

    results = []
    n_pass = 0
    for i, rec in enumerate(records):
        prompt = rec["prompt_neg"]
        if args.use_behavior_token:
            token = _BEHAVIOR_TOKENS.get(rec["expected_neg_behavior"], "")
            if token:
                prompt = f"{token} {prompt}"

        # Format as chat if tokenizer has chat template
        try:
            chat = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,          # greedy
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        failed, reason = is_negation_failure(response, rec.get("forbidden_neg", []))
        passed = not failed
        n_pass += int(passed)

        results.append({
            "id": rec["id"],
            "prompt_neg": rec["prompt_neg"],
            "forbidden_neg": rec["forbidden_neg"],
            "response": response,
            "passed": passed,
            "failure_reason": reason,
        })

        if (i + 1) % 10 == 0:
            acc = n_pass / (i + 1)
            print(f"  [{i+1}/{len(records)}] free-gen FlipAcc so far: {acc:.1%}")

    acc = n_pass / len(records)
    out = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "use_behavior_token": args.use_behavior_token,
        "n_records": len(records),
        "n_pass": n_pass,
        "free_gen_flip_acc": acc,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, Path(args.output).open("w"), indent=2)
    print(f"\nFree-gen FlipAcc: {acc:.1%}  ({n_pass}/{len(records)} passed)")
    print(f"Saved: {args.output}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--use-behavior-token", action="store_true")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N records (default: all)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
