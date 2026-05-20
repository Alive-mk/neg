"""
Negation Contrastive SFT (NC-SFT) baseline.

Adds a cross-prompt contrastive loss to vanilla SFT, teaching the model that
the same candidate should score differently under positive vs. negated prompts.
This is the most natural training-based baseline beyond DPO/vanilla-SFT.

Loss decomposition (no mechanism guidance, no behavior tokens):
  L_pos      — gold_pos ranks first under pos_prompt  (all records)
  L_neg_rank — gold_neg ranks above gold_pos under neg_prompt  (in-scope only)
  L_cross    — gold_neg scores HIGHER under neg_prompt than under pos_prompt
               gold_pos scores HIGHER under pos_prompt than under neg_prompt
               (cross-prompt contrastive, in-scope only)
  L_preserve — gold_pos stays high under neg_prompt  (out-of-scope / double-neg)
  L_ret      — KL retention vs frozen base  (all records)

Compared with Vanilla SFT: adds L_cross and L_preserve.
Compared with MGNM: no L_sup, no behavior tokens, no mechanism-guided weighting.
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


def unique_preserve_order(items) -> list[str]:
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


def record_loss_nc_sft(
    model, tokenizer, record, device,
    margin, lambda_pos, lambda_neg_rank, lambda_cross, lambda_preserve, lambda_ret,
    retention_text,
):
    """
    NC-SFT loss per record.

    L_cross is the key differentiator from Vanilla SFT:
      For gold_neg: score(neg_prompt, gold_neg) > score(pos_prompt, gold_neg) + margin
      For gold_pos: score(pos_prompt, gold_pos) > score(neg_prompt, gold_pos) + margin
    This teaches that the same candidate's score depends on whether the prompt is negated.
    """
    losses: dict[str, torch.Tensor] = {}
    behavior = record.expected_neg_behavior

    # ── L_pos: gold_pos ranks first under positive prompt ──────────────────
    pos_scores = score_many(model, tokenizer, record.prompt_pos, record.gold_pos, device)
    pos_target = torch.stack(pos_scores).max()
    competitors = unique_preserve_order(
        record.gold_neg + record.candidate_pool_neg + record.distractors
    )
    if competitors:
        pos_comp = torch.stack(
            score_many(model, tokenizer, record.prompt_pos, competitors, device)
        ).max()
        losses["L_pos"] = F.relu(
            torch.tensor(margin, device=device, dtype=pos_target.dtype) - pos_target + pos_comp
        )

    # ── In-scope negation: L_neg_rank + L_cross ────────────────────────────
    if behavior in ("suppress_target", "select_gold_neg"):

        if behavior == "suppress_target":
            gold_neg_items = record.candidate_pool_neg[:1]  # pseudo gold_neg
        else:
            gold_neg_items = record.gold_neg

        if gold_neg_items:
            s_neg_under_neg = torch.stack(
                score_many(model, tokenizer, record.prompt_neg, gold_neg_items, device)
            ).max()
            s_pos_under_neg = torch.stack(
                score_many(model, tokenizer, record.prompt_neg, record.gold_pos, device)
            ).max()

            # L_neg_rank: gold_neg > gold_pos under neg_prompt
            losses["L_neg_rank"] = F.relu(
                torch.tensor(margin, device=device, dtype=s_neg_under_neg.dtype)
                - s_neg_under_neg + s_pos_under_neg
            )

            # L_cross part 1: gold_neg scores higher under neg_prompt than pos_prompt
            s_neg_under_pos = torch.stack(
                score_many(model, tokenizer, record.prompt_pos, gold_neg_items, device)
            ).max()
            losses["L_cross_neg"] = F.relu(
                torch.tensor(margin, device=device, dtype=s_neg_under_neg.dtype)
                - s_neg_under_neg + s_neg_under_pos
            )

        # L_cross part 2: gold_pos scores higher under pos_prompt than neg_prompt
        s_pos_under_pos = torch.stack(
            score_many(model, tokenizer, record.prompt_pos, record.gold_pos, device)
        ).max()
        s_pos_under_neg2 = torch.stack(
            score_many(model, tokenizer, record.prompt_neg, record.gold_pos, device)
        ).max()
        losses["L_cross_pos"] = F.relu(
            torch.tensor(margin, device=device, dtype=s_pos_under_pos.dtype)
            - s_pos_under_pos + s_pos_under_neg2
        )

    # ── Out-of-scope: L_preserve ───────────────────────────────────────────
    elif behavior == "preserve_positive":
        # gold_pos should score similarly (high) under neg_prompt as under pos_prompt
        s_gp_neg = torch.stack(
            score_many(model, tokenizer, record.prompt_neg, record.gold_pos, device)
        ).max()
        # competing neg candidates should score lower than gold_pos under neg_prompt
        neg_candidates = unique_preserve_order(record.candidate_pool_neg + record.distractors)
        if neg_candidates:
            s_neg_cand = torch.stack(
                score_many(model, tokenizer, record.prompt_neg, neg_candidates, device)
            ).max()
            losses["L_preserve"] = F.relu(
                torch.tensor(margin, device=device, dtype=s_gp_neg.dtype)
                - s_gp_neg + s_neg_cand
            )

    # ── L_ret ──────────────────────────────────────────────────────────────
    if retention_text and lambda_ret > 0:
        losses["L_ret"] = retention_kl(model, tokenizer, retention_text, device)

    m_t = torch.tensor(0.0, device=device)
    m_t = m_t + lambda_pos       * losses.get("L_pos",        torch.tensor(0.0, device=device))
    m_t = m_t + lambda_neg_rank  * losses.get("L_neg_rank",   torch.tensor(0.0, device=device))
    m_t = m_t + lambda_cross     * losses.get("L_cross_neg",  torch.tensor(0.0, device=device))
    m_t = m_t + lambda_cross     * losses.get("L_cross_pos",  torch.tensor(0.0, device=device))
    m_t = m_t + lambda_preserve  * losses.get("L_preserve",   torch.tensor(0.0, device=device))
    m_t = m_t + lambda_ret       * losses.get("L_ret",        torch.tensor(0.0, device=device))
    return m_t, {k: float(v.detach().item()) for k, v in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",         required=True)
    parser.add_argument("--model-name",     required=True)
    parser.add_argument("--train-input",    required=True)
    parser.add_argument("--eval-input")
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--learning-rate",  type=float, default=2e-4)
    parser.add_argument("--weight-decay",   type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--train-limit",    type=int)
    parser.add_argument("--margin",         type=float, default=0.5)
    parser.add_argument("--lambda-pos",     type=float, default=1.0)
    parser.add_argument("--lambda-neg-rank",type=float, default=1.5)
    parser.add_argument("--lambda-cross",   type=float, default=1.5,
                        help="Weight for cross-prompt contrastive loss")
    parser.add_argument("--lambda-preserve",type=float, default=1.5,
                        help="Weight for out-of-scope preserve loss")
    parser.add_argument("--lambda-ret",     type=float, default=0.1)
    parser.add_argument("--scope-oversample", type=int, default=2,
                        help="Repeat out_of_scope and double_negation records N times")
    parser.add_argument("--lora-r",         type=int,   default=16)
    parser.add_argument("--lora-alpha",     type=int,   default=32)
    parser.add_argument("--lora-dropout",   type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configs = load_model_configs(args.models)
    config  = configs[args.model_name]
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

    # Scope oversampling to balance in/out-of-scope distribution
    if args.scope_oversample > 1:
        extra = [
            r for r in train_records
            if r.scope_type in ("out_of_scope", "double_negation")
        ] * (args.scope_oversample - 1)
        train_records = train_records + extra
        print(f"[scope-oversample] out_of_scope/double_negation x{args.scope_oversample}; "
              f"added {len(extra)} records, total={len(train_records)}", flush=True)

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
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)", flush=True)

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
        epoch_terms:  dict[str, list[float]] = {}

        for idx, record in enumerate(train_records):
            ret_text = retention_texts[(global_step + idx) % len(retention_texts)] if retention_texts else None
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    total_loss, loss_terms = record_loss_nc_sft(
                        model=model,
                        tokenizer=tokenizer,
                        record=record,
                        device=device,
                        margin=args.margin,
                        lambda_pos=args.lambda_pos,
                        lambda_neg_rank=args.lambda_neg_rank,
                        lambda_cross=args.lambda_cross,
                        lambda_preserve=args.lambda_preserve,
                        lambda_ret=args.lambda_ret,
                        retention_text=ret_text,
                    )
                    loss_for_step = total_loss / args.grad_accum_steps

                scaler.scale(loss_for_step).backward()
                epoch_totals.append(float(total_loss.detach().item()))
                for k, v in loss_terms.items():
                    epoch_terms.setdefault(k, []).append(v)

            except Exception as exc:
                print(f"[warn] record {record.id} skipped: {exc}", flush=True)
                continue

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

    raw_config   = read_json(args.models)
    adapted_name = f"{args.model_name}_nc_sft"
    adapted_entry = {
        "name": adapted_name,
        "mode": "hf_local",
        "model_path": model_path,
        "adapter_path": str(adapter_dir),
        "device_map": getattr(config, "device_map", "auto"),
        "torch_dtype": getattr(config, "torch_dtype", "auto"),
        "trust_remote_code": config.trust_remote_code,
    }
    write_json(output_dir / "model_config_with_adapter.json",
               {"models": raw_config["models"] + [adapted_entry]})
    write_json(output_dir / "training_summary.json", {
        "model_name":        args.model_name,
        "adapted_model_name":adapted_name,
        "baseline_type":     "negation_contrastive_sft",
        "train_input":       args.train_input,
        "num_train_records": len(train_records),
        "trainable_params":  trainable_params,
        "total_params":      total_params,
        "history":           history,
        "args":              vars(args),
    })
    print(f"[nc-sft] saved adapter → {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
