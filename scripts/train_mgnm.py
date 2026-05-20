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
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def continuation_score(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    full_inputs = tokenizer(
        join_prompt_and_continuation(prompt, continuation),
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_len = int(full_inputs["input_ids"].shape[1])
    if full_len <= prompt_len:
        raise ValueError(f"continuation tokenized to empty sequence for {continuation!r}")

    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    log_probs = logits.log_softmax(dim=-1)
    target_ids = input_ids[:, prompt_len:]
    token_logprobs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    return token_logprobs.mean()


def score_many(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    continuations: list[str],
    device: torch.device,
) -> list[torch.Tensor]:
    return [continuation_score(model, tokenizer, prompt, item, device) for item in continuations]


def retention_kl(
    model: torch.nn.Module,
    tokenizer: object,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with model.disable_adapter():
        with torch.no_grad():
            ref_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    cur_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    ref_logits = ref_outputs.logits[:, :-1, :]
    cur_logits = cur_outputs.logits[:, :-1, :]
    mask = attention_mask[:, 1:].float()
    token_kl = F.kl_div(
        cur_logits.log_softmax(dim=-1),
        ref_logits.softmax(dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)


_BEHAVIOR_TOKENS: dict[str, str] = {
    "suppress_target":   "[SUPPRESS]",
    "select_gold_neg":   "[SELECT]",
    "preserve_positive": "[PRESERVE]",
}


def _neg_prompt(record: object, use_behavior_token: bool) -> str:
    if not use_behavior_token:
        return record.prompt_neg
    token = _BEHAVIOR_TOKENS.get(record.expected_neg_behavior, "")
    return f"{token} {record.prompt_neg}" if token else record.prompt_neg


def record_loss(
    model: torch.nn.Module,
    tokenizer: object,
    record: object,
    device: torch.device,
    margins: dict[str, float],
    lambdas: dict[str, float],
    retention_text: str | None,
    use_behavior_token: bool = False,
    listwise_rank: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}
    prompt_neg = _neg_prompt(record, use_behavior_token)

    pos_target_scores = score_many(model, tokenizer, record.prompt_pos, record.gold_pos, device)
    pos_target = torch.stack(pos_target_scores).max()
    pos_competitors = unique_preserve_order(
        record.gold_neg + record.candidate_pool_neg + record.distractors
    )
    if pos_competitors:
        pos_comp = torch.stack(
            score_many(model, tokenizer, record.prompt_pos, pos_competitors, device)
        ).max()
        losses["L_pos"] = F.relu(
            torch.tensor(margins["pos"], device=device, dtype=pos_target.dtype)
            - pos_target
            + pos_comp
        )

    if record.expected_neg_behavior == "suppress_target":
        neg_target = torch.stack(
            score_many(model, tokenizer, prompt_neg, record.gold_pos, device)
        ).max()
        neg_pool = unique_preserve_order(record.candidate_pool_neg + record.distractors)
        if neg_pool:
            pool_mean = torch.stack(
                score_many(model, tokenizer, prompt_neg, neg_pool, device)
            ).mean()
            losses["L_sup"] = F.relu(
                torch.tensor(margins["sup"], device=device, dtype=neg_target.dtype)
                + neg_target
                - pool_mean
            )
    elif record.expected_neg_behavior == "select_gold_neg":
        neg_gold = torch.stack(
            score_many(model, tokenizer, prompt_neg, record.gold_neg, device)
        ).max()
        if listwise_rank:
            gold_neg_set = set(record.gold_neg)
            competing = [c for c in unique_preserve_order(
                record.gold_pos + record.candidate_pool_neg + record.distractors
            ) if c not in gold_neg_set]
            if competing:
                comp_scores = torch.stack(
                    score_many(model, tokenizer, prompt_neg, competing, device)
                )
                all_scores = torch.cat([neg_gold.unsqueeze(0), comp_scores])
                losses["L_rank"] = -neg_gold + torch.logsumexp(all_scores, dim=0)
            else:
                losses["L_rank"] = torch.tensor(0.0, device=device)
        else:
            neg_target = torch.stack(
                score_many(model, tokenizer, prompt_neg, record.gold_pos, device)
            ).max()
            losses["L_rank"] = F.relu(
                torch.tensor(margins["rank"], device=device, dtype=neg_target.dtype)
                - neg_gold
                + neg_target
            )
        # L_rank only ensures gold_neg > gold_pos; push gold_neg above distractors too
        if lambdas.get("rank_dist", 0.0) > 0:
            gold_neg_set = set(record.gold_neg)
            dist_pool = [d for d in unique_preserve_order(
                record.candidate_pool_neg + record.distractors
            ) if d not in gold_neg_set]
            if dist_pool:
                dist_scores = torch.stack(
                    score_many(model, tokenizer, prompt_neg, dist_pool, device)
                )
                neg_dist_max = dist_scores.max()
                losses["L_rank_dist"] = F.relu(
                    torch.tensor(margins["rank"], device=device, dtype=neg_gold.dtype)
                    - neg_gold
                    + neg_dist_max
                )
    elif record.expected_neg_behavior == "preserve_positive":
        neg_pos = torch.stack(
            score_many(model, tokenizer, prompt_neg, record.gold_pos, device)
        ).max()
        neg_competitors = unique_preserve_order(
            record.gold_neg + record.candidate_pool_neg + record.distractors
        )
        if neg_competitors:
            neg_comp = torch.stack(
                score_many(model, tokenizer, prompt_neg, neg_competitors, device)
            ).max()
            losses["L_preserve"] = F.relu(
                torch.tensor(margins["preserve"], device=device, dtype=neg_pos.dtype)
                - neg_pos
                + neg_comp
            )

    if retention_text and lambdas["ret"] > 0:
        losses["L_ret"] = retention_kl(model, tokenizer, retention_text, device)

    eff_lambda_rank = (
        lambdas.get("rank_select", lambdas["rank"])
        if record.expected_neg_behavior == "select_gold_neg"
        else lambdas["rank"]
    )

    total = torch.tensor(0.0, device=device)
    total = total + lambdas["pos"] * losses.get("L_pos", torch.tensor(0.0, device=device))
    total = total + lambdas["sup"] * losses.get("L_sup", torch.tensor(0.0, device=device))
    total = total + eff_lambda_rank * losses.get("L_rank", torch.tensor(0.0, device=device))
    total = total + lambdas.get("rank_dist", 0.0) * losses.get("L_rank_dist", torch.tensor(0.0, device=device))
    total = total + lambdas["preserve"] * losses.get(
        "L_preserve", torch.tensor(0.0, device=device)
    )
    total = total + lambdas["ret"] * losses.get("L_ret", torch.tensor(0.0, device=device))

    return total, {name: float(value.detach().item()) for name, value in losses.items()}


def evaluate_objective(
    model: torch.nn.Module,
    tokenizer: object,
    records: list[object],
    device: torch.device,
    margins: dict[str, float],
    lambdas: dict[str, float],
    retention_texts: list[str],
    max_records: int | None = None,
    use_behavior_token: bool = False,
    listwise_rank: bool = False,
) -> dict[str, float]:
    subset = records[:max_records] if max_records else records
    if not subset:
        return {"mean_total_loss": 0.0}

    totals: list[float] = []
    with torch.no_grad():
        for idx, record in enumerate(subset):
            retention_text = retention_texts[idx % len(retention_texts)] if retention_texts else None
            total, _ = record_loss(
                model=model,
                tokenizer=tokenizer,
                record=record,
                device=device,
                margins=margins,
                lambdas=lambdas,
                retention_text=retention_text,
                use_behavior_token=use_behavior_token,
                listwise_rank=listwise_rank,
            )
            totals.append(float(total.item()))
    return {"mean_total_loss": mean(totals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--eval-input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=0,
                        help="Print training progress every N records. Default 0 disables step logs.")
    parser.add_argument("--margin-pos", type=float, default=0.5)
    parser.add_argument("--margin-sup", type=float, default=0.5)
    parser.add_argument("--margin-rank", type=float, default=0.5)
    parser.add_argument("--margin-preserve", type=float, default=0.5)
    parser.add_argument("--lambda-pos", type=float, default=1.0)
    parser.add_argument("--lambda-sup", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-ret", type=float, default=0.05)
    parser.add_argument("--lambda-rank-select", type=float, default=None,
                        help="λ_rank override for select_gold_neg records only. "
                             "Default: same as --lambda-rank.")
    parser.add_argument("--lambda-rank-dist", type=float, default=0.0,
                        help="λ for L_rank_dist: pairwise margin loss pushing gold_neg above "
                             "the hardest distractor/pool item on neg prompt (select_gold_neg only). "
                             "Default 0 = disabled. Addresses rank->flip gap caused by unpunished distractors.")
    parser.add_argument("--listwise-rank", action="store_true", default=False,
                        help="Use cross-entropy over full candidate pool for L_rank instead of "
                             "pairwise margin loss vs gold_pos only.")
    parser.add_argument("--behavior-token", action="store_true", default=False,
                        help="Prefix neg prompts with [SUPPRESS]/[SELECT]/[PRESERVE] during training.")
    parser.add_argument("--init-adapter-path",
                        help="Load an existing LoRA adapter instead of creating a new one. "
                             "Use for phase-2 fine-tuning on top of a trained adapter.")
    parser.add_argument("--behavior-filter",
                        help="If set, keep only training records with this expected_neg_behavior "
                             "(e.g. select_gold_neg). Useful for phase-2 rank-only fine-tuning.")
    parser.add_argument("--scope-oversample", type=int, default=1,
                        help="Duplicate records with out_of_scope or double_negation scope N times "
                             "to compensate for the 3:1:1 in_scope training imbalance (default 1 = no change).")
    parser.add_argument("--dn-oversample", type=int, default=None,
                        help="Oversample factor specifically for double_negation records, overriding "
                             "--scope-oversample for that scope. E.g. --scope-oversample 2 --dn-oversample 4 "
                             "gives out_of_scope x2, double_negation x4.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configs = load_model_configs(args.models)
    config = configs[args.model_name]
    model_path = config.model_path or config.model
    if not model_path:
        raise ValueError(f"{args.model_name} is missing model_path")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("E4 training requires peft and transformers") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.train_input)
    if args.behavior_filter:
        train_records = [r for r in train_records if r.expected_neg_behavior == args.behavior_filter]
        print(f"[behavior-filter] kept {len(train_records)} records with behavior={args.behavior_filter!r}", flush=True)
    if args.scope_oversample > 1 or args.dn_oversample:
        oos_factor = args.scope_oversample - 1
        dn_factor = (args.dn_oversample if args.dn_oversample is not None else args.scope_oversample) - 1
        extras = (
            [r for r in train_records if r.scope_type == "out_of_scope"] * oos_factor
            + [r for r in train_records if r.scope_type == "double_negation"] * dn_factor
        )
        train_records = train_records + extras
        print(
            f"[scope-oversample] out_of_scope x{oos_factor+1}, double_negation x{dn_factor+1}; "
            f"added {len(extras)} records, total={len(train_records)}",
            flush=True,
        )
    if args.train_limit is not None:
        train_records = train_records[: args.train_limit]
    eval_records = load_records(args.eval_input) if args.eval_input else []

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config.trust_remote_code,
    )
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

    if args.init_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
        print(f"[phase2] Loaded adapter from {args.init_adapter_path}", flush=True)
    else:
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

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    margins = {
        "pos": args.margin_pos,
        "sup": args.margin_sup,
        "rank": args.margin_rank,
        "preserve": args.margin_preserve,
    }
    lambdas = {
        "pos": args.lambda_pos,
        "sup": args.lambda_sup,
        "rank": args.lambda_rank,
        "preserve": args.lambda_preserve,
        "ret": args.lambda_ret,
        "rank_select": args.lambda_rank_select if args.lambda_rank_select is not None else args.lambda_rank,
        "rank_dist": args.lambda_rank_dist,
    }

    retention_texts = unique_preserve_order(
        join_prompt_and_continuation(record.prompt_pos, record.gold_pos[0])
        for record in train_records
        if record.gold_pos
    )
    retention_texts = retention_texts[: min(len(retention_texts), 64)]

    history: list[dict[str, float | int]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.shuffle(train_records)
        epoch_totals: list[float] = []
        epoch_loss_terms: dict[str, list[float]] = {}
        for idx, record in enumerate(train_records):
            retention_text = (
                retention_texts[(global_step + idx) % len(retention_texts)]
                if retention_texts
                else None
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                total_loss, loss_terms = record_loss(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    device=device,
                    margins=margins,
                    lambdas=lambdas,
                    retention_text=retention_text,
                    use_behavior_token=args.behavior_token,
                    listwise_rank=args.listwise_rank,
                )
                loss_for_step = total_loss / args.grad_accum_steps
            scaler.scale(loss_for_step).backward()
            epoch_totals.append(float(total_loss.detach().item()))
            for name, value in loss_terms.items():
                epoch_loss_terms.setdefault(name, []).append(value)

            if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == len(train_records):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if args.log_every and (idx + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "record": idx + 1,
                            "total_records": len(train_records),
                            "global_step": global_step,
                            "last_loss": float(total_loss.detach().item()),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        eval_metrics = evaluate_objective(
            model=model,
            tokenizer=tokenizer,
            records=eval_records,
            device=device,
            margins=margins,
            lambdas=lambdas,
            retention_texts=retention_texts,
            max_records=args.eval_limit,
            use_behavior_token=args.behavior_token,
            listwise_rank=args.listwise_rank,
        )
        epoch_summary: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_mean_total_loss": mean(epoch_totals) if epoch_totals else 0.0,
            "global_step": global_step,
        }
        for name, values in epoch_loss_terms.items():
            epoch_summary[f"train_mean_{name}"] = mean(values)
        epoch_summary.update(eval_metrics)
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    raw_model_config = read_json(args.models)
    adapted_name = f"{args.model_name}_e4"
    adapted_entry = {
        "name": adapted_name,
        "mode": "hf_local",
        "model_path": model_path,
        "adapter_path": str(adapter_dir),
        "device_map": "auto",
        "torch_dtype": config.torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    model_config_out = {
        "models": raw_model_config["models"] + [adapted_entry],
    }
    write_json(output_dir / "model_config_with_adapter.json", model_config_out)
    write_json(
        output_dir / "training_summary.json",
        {
            "model_name": args.model_name,
            "adapted_model_name": adapted_name,
            "train_input": args.train_input,
            "eval_input": args.eval_input,
            "num_train_records": len(train_records),
            "num_eval_records": len(eval_records),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "history": history,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
