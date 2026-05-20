"""
A2: Free-generation evaluation for preserve_positive records.

For each preserve_positive record, the model receives the negated prompt (+ [PRESERVE]
token for the oracle) and generates a free-text response. We check whether the response
contains key content words from gold_pos — indicating the model preserved the correct
answer despite the negating surface form.

Detection heuristic:
  PASS: response contains ≥1 non-trivial keyword (len>3, not stopword) from gold_pos.
  FAIL: response contains none of the gold_pos keywords.

Preserve-FlipAcc = proportion of records that PASS.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_STOPWORDS = {"the", "a", "an", "is", "not", "for", "its", "it", "to", "of",
              "in", "on", "at", "by", "be", "was", "are", "have", "has", "do",
              "and", "with", "that", "this", "from", "are", "but", "use"}

def extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 3}

def contains_gold(response: str, gold_pos: list[str]) -> tuple[bool, str]:
    resp_kw = extract_keywords(response)
    for gold in gold_pos:
        gkw = extract_keywords(gold)
        overlap = gkw & resp_kw
        if overlap:
            return True, f"matched({overlap})"
    return False, "no_gold_keywords"

def keyword_rerank_score(response: str, gold_pos: list[str]) -> tuple[float, str]:
    resp_kw = extract_keywords(response)
    best_score = 0.0
    best_overlap: set[str] = set()
    for gold in gold_pos:
        gkw = extract_keywords(gold)
        if not gkw:
            continue
        overlap = gkw & resp_kw
        score = len(overlap) / len(gkw)
        if score > best_score:
            best_score = score
            best_overlap = overlap

    penalty = 0.0
    lower = response.lower()
    control_hits = []
    for token in ("[preserve]", "[suppress]", "[select]", "<control>", "</control>"):
        if token in lower:
            penalty += 0.25
            control_hits.append(token)
    score = best_score - penalty
    reason = f"keyword_coverage={best_score:.3f}; overlap={sorted(best_overlap)}"
    if control_hits:
        reason += f"; control_echo={control_hits}; penalty={penalty:.2f}"
    return score, reason

def continuation_logprob_score(
    model,
    sequence: torch.Tensor,
    input_len: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[float, str]:
    with torch.no_grad():
        outputs = model(input_ids=sequence.unsqueeze(0))
        logits = outputs.logits[:, :-1, :]
        target_ids = sequence.unsqueeze(0)[:, 1:]
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(0).squeeze(-1)

    generated_ids = sequence[input_len:]
    generated_logprobs = token_logprobs[input_len - 1:]
    mask = torch.ones_like(generated_ids, dtype=torch.bool)
    if pad_token_id is not None:
        mask &= generated_ids != pad_token_id
    if eos_token_id is not None:
        mask &= generated_ids != eos_token_id
    if not mask.any():
        return float("-inf"), "avg_logprob=-inf; tokens=0"

    selected = generated_logprobs[mask]
    avg_logprob = selected.mean().item()
    return avg_logprob, f"avg_logprob={avg_logprob:.4f}; tokens={selected.numel()}"

def render_plain_chat(chat: list[dict[str, str]]) -> str:
    parts = []
    for msg in chat:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts)

def render_answer_only(prompt: str, language: str) -> str:
    if language == "zh":
        return f"问题：{prompt}\n只输出答案短语，不要解释："
    return f"Question: {prompt}\nAnswer with only the key phrase, no explanation:"

def evaluate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        trust_remote_code=True, device_map=device
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    print(f"Model ready on {device}.")

    records = [json.loads(l) for l in Path(args.input).open()]
    records = [r for r in records if r.get("expected_neg_behavior") == "preserve_positive"]
    if args.n:
        records = records[:args.n]
    print(f"Evaluating {len(records)} preserve_positive records")

    results = []
    n_pass = 0
    for i, rec in enumerate(records):
        prompt = rec["prompt_neg"]
        chat = None
        control_mode = args.control_mode
        if args.use_behavior_token:
            control_mode = "token"
        if control_mode == "token":
            if args.answer_only != "none":
                prompt = f"[PRESERVE] {render_answer_only(prompt, args.answer_only)}"
            else:
                prompt = f"[PRESERVE] {prompt}"
        elif control_mode == "hidden":
            prompt = (
                "Behavior mode: PRESERVE. Do not print the behavior mode or any "
                f"control token.\n\n{prompt}"
            )
        elif control_mode == "system_hidden":
            chat = [
                {
                    "role": "system",
                    "content": "The behavior mode is PRESERVE. Do not output the behavior label.",
                },
                {"role": "user", "content": f"{prompt}\nAnswer:"},
            ]
        elif control_mode == "system_token":
            chat = [
                {
                    "role": "system",
                    "content": "[PRESERVE]\nDo not output the behavior label or any control token.",
                },
                {"role": "user", "content": f"{prompt}\nAnswer:"},
            ]
        elif control_mode == "tag":
            prompt = f"<control>PRESERVE</control>\nQuestion: {prompt}\nAnswer:"
        elif control_mode == "tag_lower":
            prompt = f"<control>preserve</control>\nQuestion: {prompt}\nAnswer:"
        elif control_mode == "instruction":
            prompt = (
                "Instruction type: preserve the positive answer when negation is out of scope.\n"
                f"Question: {prompt}\nAnswer:"
            )
        elif control_mode != "none":
            raise ValueError(f"Unknown control mode: {control_mode}")

        if args.answer_only != "none" and control_mode != "token" and chat is None:
            prompt = render_answer_only(prompt, args.answer_only)

        if args.no_chat_template:
            input_text = render_plain_chat(chat) if chat is not None else prompt
            if args.answer_only == "none" and not input_text.rstrip().endswith("Answer:"):
                input_text = f"{input_text}\nAnswer:"
        else:
            try:
                if chat is None:
                    chat = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = prompt

        if not input_text:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        do_sample = args.do_sample or args.num_return_sequences > 1
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 1.0,
                num_return_sequences=args.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )
        candidates = []
        for seq in out:
            generated_ids = seq[inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            keyword_score, keyword_reason = keyword_rerank_score(text, rec.get("gold_pos", []))
            logprob_score, logprob_reason = continuation_logprob_score(
                model,
                seq,
                inputs["input_ids"].shape[1],
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            )
            candidates.append({
                "response": text,
                "keyword_rerank_score": keyword_score,
                "keyword_rerank_reason": keyword_reason,
                "model_logprob_score": logprob_score,
                "model_logprob_reason": logprob_reason,
            })

        if args.rerank in {"keyword", "pass_at_5_keyword"}:
            best = max(
                enumerate(candidates),
                key=lambda x: (x[1]["keyword_rerank_score"], -x[0]),
            )
            response = best[1]["response"]
            selected_index = best[0]
        elif args.rerank == "model_logprob":
            best = max(
                enumerate(candidates),
                key=lambda x: (x[1]["model_logprob_score"], -x[0]),
            )
            response = best[1]["response"]
            selected_index = best[0]
        else:
            response = candidates[0]["response"]
            selected_index = 0

        gold_pos = rec.get("gold_pos", [])
        passed, reason = contains_gold(response, gold_pos)
        n_pass += int(passed)

        row = {
            "id": rec["id"],
            "prompt_neg": rec["prompt_neg"],
            "gold_pos": gold_pos,
            "response": response,
            "selected_index": selected_index,
            "passed": passed,
            "reason": reason,
        }
        if args.num_return_sequences > 1 or args.rerank != "none":
            row["candidates"] = candidates
        results.append(row)

        if (i + 1) % 10 == 0:
            acc = n_pass / (i + 1)
            print(f"  [{i+1}/{len(records)}] Preserve-FlipAcc so far: {acc:.1%}")

    acc = n_pass / len(records)
    out = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "input": args.input,
        "use_behavior_token": args.use_behavior_token,
        "control_mode": args.control_mode,
        "answer_only": args.answer_only,
        "no_chat_template": args.no_chat_template,
        "do_sample": do_sample,
        "temperature": args.temperature,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "rerank": args.rerank,
        "diagnostic": args.rerank == "pass_at_5_keyword",
        "n_records": len(records),
        "n_pass": n_pass,
        "preserve_flip_acc": acc,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, Path(args.output).open("w"), indent=2)
    print(f"\nPreserve-FlipAcc: {acc:.1%}  ({n_pass}/{len(records)} passed)")
    print(f"Saved: {args.output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-behavior-token", action="store_true")
    parser.add_argument("--device", default=None, help="Explicit device_map/device, e.g. cuda:2")
    parser.add_argument(
        "--control-mode",
        choices=[
            "none",
            "token",
            "hidden",
            "system_hidden",
            "system_token",
            "tag",
            "tag_lower",
            "instruction",
        ],
        default="none",
        help="Preserve control style. --use-behavior-token remains an alias for token.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--rerank",
        choices=["none", "keyword", "pass_at_5_keyword", "model_logprob"],
        default="none",
        help=(
            "Select sampled output by model_logprob, or run pass_at_5_keyword as "
            "a gold-keyword upper-bound diagnostic. 'keyword' is kept as a legacy alias."
        ),
    )
    parser.add_argument(
        "--answer-only",
        choices=["none", "en", "zh"],
        default="none",
        help="Constrain open generation to answer-only phrase format.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Render prompts as plain text instead of tokenizer chat template.",
    )
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
