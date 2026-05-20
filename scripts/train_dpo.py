"""DPO baseline for negation blindness.

Converts E4 training records into preference pairs:
  select_gold_neg   → (prompt_neg, gold_neg[0], gold_pos[0])
  preserve_positive → (prompt_neg, gold_pos[0], candidate_pool_neg[0])
  suppress_target   → (prompt_neg, alternative, gold_pos[0])   [if forbidden_neg available]

Trains with TRL DPOTrainer + LoRA on the same base model used for MGNM.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from neg_blindness.api import load_model_configs


def build_dpo_pairs(records: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    skipped = 0
    for r in records:
        behavior = r.get("expected_neg_behavior", "")
        prompt   = r.get("prompt_neg", "")
        gold_pos = r.get("gold_pos", [])
        gold_neg = r.get("gold_neg", [])
        cpool    = r.get("candidate_pool_neg", [])
        distractors = r.get("distractors", [])

        if behavior == "select_gold_neg":
            if not gold_neg or not gold_pos:
                skipped += 1; continue
            chosen   = rng.choice(gold_neg)
            # rejected: gold_pos (model should NOT pick the pos answer for a neg query)
            rejected = rng.choice(gold_pos)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        elif behavior == "preserve_positive":
            if not gold_pos or not cpool:
                skipped += 1; continue
            chosen   = rng.choice(gold_pos)
            rejected = rng.choice(cpool)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        elif behavior == "suppress_target":
            # chosen = a distractor (model should prefer anything over the target)
            # rejected = gold_pos (which is what the base model naively outputs)
            alts = distractors or cpool
            if not alts or not gold_pos:
                skipped += 1; continue
            chosen   = rng.choice(alts)
            rejected = rng.choice(gold_pos)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"[dpo] built {len(pairs)} pairs  (skipped {skipped} incomplete records)")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",       required=True)
    parser.add_argument("--model-name",   required=True)
    parser.add_argument("--train-input",  required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--beta",         type=float, default=0.1,
                        help="DPO beta (KL regularization strength)")
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--max-length",   type=int,   default=256)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    configs   = load_model_configs(args.models)
    model_cfg = configs[args.model_name]
    model_path = model_cfg.model_path

    records = [json.loads(l) for l in open(args.train_input)]
    pairs   = build_dpo_pairs(records, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(pairs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "model_name": args.model_name,
        "train_input": args.train_input,
        "num_pairs": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "lr": args.lr,
    }
    json.dump(summary, open(out_dir / "training_summary.json", "w"), indent=2)
    print(f"[dpo] saved adapter → {adapter_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
