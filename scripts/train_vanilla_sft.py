"""
Vanilla Negation SFT baseline — same data, same LoRA setup as MGNM, but uses a
uniform pairwise ranking loss without the mechanism-guided loss decomposition.

Differences from train_mgnm.py (for the ablation table):
  - suppress_target: picks the first candidate_pool_neg item as pseudo-gold_neg
    and applies a plain ranking loss (pseudo_gold > gold_pos under neg_prompt)
  - select_gold_neg: same ranking loss as MGNM (no change)
  - preserve_positive: skipped entirely (vanilla SFT has no scope-control signal)
  - NO L_sup loss (the mechanism-targeted suppression term)
  - NO L_preserve loss

Everything else is identical: L_pos, L_ret, LoRA config, optimizer, grad accum.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import join_prompt_and_continuation, load_model_configs
from neg_blindness.io_utils import load_records, read_json, write_json


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def continuation_score(model, tokenizer, prompt, continuation, device):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_enc = tokenizer(
        join_prompt_and_continuation(prompt, continuation),
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_enc["input_ids"].shape[1])
    if full_len <= prompt_len:
        raise ValueError(f"continuation tokenized empty: {continuation!r}")
    input_ids = full_enc["input_ids"].to(device)
    attn_mask = full_enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits[:, prompt_len - 1 : -1, :]
    lp = logits.log_softmax(dim=-1)
    target = input_ids[:, prompt_len:]
    return lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()


def score_many(model, tokenizer, prompt, continuations, device):
    return [continuation_score(model, tokenizer, prompt, c, device) for c in continuations]


def retention_kl(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with model.disable_adapter():
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask, use_cache=False)
    cur = model(input_ids=ids, attention_mask=mask, use_cache=False)
    ref_l = ref.logits[:, :-1, :]
    cur_l = cur.logits[:, :-1, :]
    m = mask[:, 1:].float()
    kl = F.kl_div(cur_l.log_softmax(-1), ref_l.softmax(-1), reduction="none").sum(-1)
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def record_loss_vanilla(model, tokenizer, record, device, margin, lambda_pos, lambda_rank, lambda_ret, retention_text):
    """
    Vanilla loss:
      L_pos   — preserve gold_pos rank under positive prompt (same as MGNM)
      L_rank  — rank gold_neg (or pseudo-gold_neg) above gold_pos under neg prompt
                applies to suppress_target and select_gold_neg only
      L_ret   — KL retention (same as MGNM)

    No L_sup, no L_preserve.
    """
    losses: dict[str, torch.Tensor] = {}

    # ── L_pos ──────────────────────────────────────────────────────────────
    pos_scores = score_many(model, tokenizer, record.prompt_pos, record.gold_pos, device)
    pos_target = torch.stack(pos_scores).max()
    competitors = unique_preserve_order(
        record.gold_neg + record.candidate_pool_neg + record.distractors
    )
    if competitors:
        pos_comp = torch.stack(score_many(model, tokenizer, record.prompt_pos, competitors, device)).max()
        losses["L_pos"] = F.relu(
            torch.tensor(margin, device=device, dtype=pos_target.dtype) - pos_target + pos_comp
        )

    # ── L_rank (vanilla negation) ────────────────────────────────────────
    if record.expected_neg_behavior == "suppress_target":
        # pick first pool candidate as pseudo-gold_neg
        pseudo = record.candidate_pool_neg[:1]
        if pseudo:
            neg_gold = torch.stack(score_many(model, tokenizer, record.prompt_neg, pseudo, device)).max()
            neg_forbidden = torch.stack(
                score_many(model, tokenizer, record.prompt_neg, record.gold_pos, device)
            ).max()
            losses["L_rank"] = F.relu(
                torch.tensor(margin, device=device, dtype=neg_gold.dtype) - neg_gold + neg_forbidden
            )

    elif record.expected_neg_behavior == "select_gold_neg":
        neg_gold = torch.stack(score_many(model, tokenizer, record.prompt_neg, record.gold_neg, device)).max()
        neg_forbidden = torch.stack(
            score_many(model, tokenizer, record.prompt_neg, record.gold_pos, device)
        ).max()
        losses["L_rank"] = F.relu(
            torch.tensor(margin, device=device, dtype=neg_gold.dtype) - neg_gold + neg_forbidden
        )

    # preserve_positive: no negation loss in vanilla SFT

    # ── L_ret ────────────────────────────────────────────────────────────
    if retention_text and lambda_ret > 0:
        losses["L_ret"] = retention_kl(model, tokenizer, retention_text, device)

    total = torch.tensor(0.0, device=device)
    total = total + lambda_pos * losses.get("L_pos", torch.tensor(0.0, device=device))
    total = total + lambda_rank * losses.get("L_rank", torch.tensor(0.0, device=device))
    total = total + lambda_ret * losses.get("L_ret", torch.tensor(0.0, device=device))
    return total, {k: float(v.detach().item()) for k, v in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--eval-input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lambda-pos", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-ret", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configs = load_model_configs(args.models)
    config = configs[args.model_name]
    model_path = config.model_path or config.model
    if not model_path:
        raise ValueError(f"{args.model_name} missing model_path")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("requires peft and transformers") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.train_input)
    if args.train_limit is not None:
        train_records = train_records[: args.train_limit]
    eval_records = load_records(args.eval_input) if args.eval_input else []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=config.trust_remote_code,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda"
        else torch.amp.GradScaler("cpu", enabled=False)
    )

    retention_texts = unique_preserve_order(
        join_prompt_and_continuation(r.prompt_pos, r.gold_pos[0])
        for r in train_records if r.gold_pos
    )[:64]

    history: list[dict] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(train_records)
        epoch_totals: list[float] = []
        epoch_terms: dict[str, list[float]] = {}

        for idx, record in enumerate(train_records):
            ret_text = retention_texts[(global_step + idx) % len(retention_texts)] if retention_texts else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                total_loss, loss_terms = record_loss_vanilla(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    device=device,
                    margin=args.margin,
                    lambda_pos=args.lambda_pos,
                    lambda_rank=args.lambda_rank,
                    lambda_ret=args.lambda_ret,
                    retention_text=ret_text,
                )
                loss_for_step = total_loss / args.grad_accum_steps

            scaler.scale(loss_for_step).backward()
            epoch_totals.append(float(total_loss.detach().item()))
            for k, v in loss_terms.items():
                epoch_terms.setdefault(k, []).append(v)

            if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == len(train_records):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        summary: dict = {
            "epoch": epoch + 1,
            "train_mean_total_loss": mean(epoch_totals) if epoch_totals else 0.0,
            "global_step": global_step,
        }
        for k, vs in epoch_terms.items():
            summary[f"train_mean_{k}"] = mean(vs)
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    raw_config = read_json(args.models)
    adapted_name = f"{args.model_name}_vanilla"
    adapted_entry = {
        "name": adapted_name,
        "mode": "hf_local",
        "model_path": model_path,
        "adapter_path": str(adapter_dir),
        "device_map": getattr(config, "device_map", "auto"),
        "torch_dtype": getattr(config, "torch_dtype", "auto"),
        "trust_remote_code": config.trust_remote_code,
    }
    write_json(output_dir / "model_config_with_adapter.json", {"models": raw_config["models"] + [adapted_entry]})
    write_json(output_dir / "training_summary.json", {
        "model_name": args.model_name,
        "adapted_model_name": adapted_name,
        "baseline_type": "vanilla_negation_sft",
        "train_input": args.train_input,
        "eval_input": args.eval_input,
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "history": history,
        "args": vars(args),
    })


if __name__ == "__main__":
    main()
