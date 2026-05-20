"""
MMLU 5-shot evaluation for capability-preservation check.

Scores each A/B/C/D option as a single-token logprob continuation.
Samples a stratified subset (--n-per-subject questions per subject) for speed.
Compares base model vs fine-tuned adapter side-by-side.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_mmlu.py \
      --base-model-path  /data/mingkai/neg/model/Qwen2.5-7B \
      --adapter-path     outputs/e4_qwen_bal_full_3ep/adapter \
      --output           outputs/mmlu_capability_check.json \
      --n-per-subject    10 \
      --seed             42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


LETTERS = ["A", "B", "C", "D"]

MMLU_SUBJECT_GROUPS = {
    "stem": ["abstract_algebra","college_chemistry","college_mathematics","college_physics",
             "computer_security","electrical_engineering","high_school_mathematics",
             "high_school_physics","high_school_chemistry","high_school_computer_science",
             "machine_learning"],
    "humanities": ["formal_logic","high_school_european_history","high_school_us_history",
                   "high_school_world_history","jurisprudence","logical_fallacies",
                   "moral_disputes","philosophy","prehistory","world_religions"],
    "social_sciences": ["econometrics","high_school_geography","high_school_government_and_politics",
                        "high_school_macroeconomics","high_school_microeconomics",
                        "high_school_psychology","human_sexuality","political_science",
                        "professional_psychology","sociology"],
    "other": ["business_ethics","clinical_knowledge","global_facts","management",
              "marketing","medical_genetics","miscellaneous","nutrition",
              "professional_accounting","professional_medicine","virology"],
}


def format_question(question: str, choices: list[str], include_answer: bool = False,
                    answer_idx: int = 0) -> str:
    lines = [question]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    if include_answer:
        lines.append(f"Answer: {LETTERS[answer_idx]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_fewshot_prompt(dev_examples: list[dict], subject: str) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    shots = [format_question(ex["question"], ex["choices"], include_answer=True, answer_idx=ex["answer"])
             for ex in dev_examples[:5]]
    return header + "\n\n".join(shots) + "\n\n"


def score_options(model, tokenizer, prompt: str, device: str) -> list[float]:
    import torch
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    last_logits = out.logits[0, -1, :]
    log_probs = last_logits.log_softmax(dim=-1)
    scores = []
    for letter in LETTERS:
        tid = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if tid:
            scores.append(float(log_probs[tid[0]].item()))
        else:
            scores.append(float("-inf"))
    return scores


def evaluate_model(model, tokenizer, questions: list[dict], fewshot_cache: dict[str, str],
                   device: str) -> dict:
    correct = 0
    per_subject: dict[str, list[bool]] = defaultdict(list)
    for q in questions:
        subject = q["subject"]
        prefix = fewshot_cache[subject]
        prompt = prefix + format_question(q["question"], q["choices"])
        scores = score_options(model, tokenizer, prompt, device)
        pred = scores.index(max(scores))
        hit = pred == q["answer"]
        per_subject[subject].append(hit)
        correct += hit

    subject_accs = {s: mean(v) for s, v in per_subject.items()}
    group_accs: dict[str, float] = {}
    for group, subjects in MMLU_SUBJECT_GROUPS.items():
        hits = [v for s in subjects for v in per_subject.get(s, [])]
        group_accs[group] = mean(hits) if hits else 0.0

    return {
        "overall_accuracy": correct / len(questions),
        "n_questions": len(questions),
        "per_subject": subject_accs,
        "per_group": group_accs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-subject", type=int, default=10,
                        help="Questions per subject in the stratified sample (default 10 → ~570 total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)

    print("[mmlu] loading dataset...")
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds  = load_dataset("cais/mmlu", "all", split="dev")

    # Group test questions by subject, stratified sample
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for item in test_ds:
        by_subject[item["subject"]].append(item)

    questions: list[dict] = []
    for subject, items in sorted(by_subject.items()):
        sample = rng.sample(items, min(args.n_per_subject, len(items)))
        questions.extend(sample)
    print(f"[mmlu] sampled {len(questions)} questions across {len(by_subject)} subjects")

    # Build 5-shot prompts from dev set, per subject
    dev_by_subject: dict[str, list[dict]] = defaultdict(list)
    for item in dev_ds:
        dev_by_subject[item["subject"]].append(item)

    fewshot_cache: dict[str, str] = {}
    for subject in by_subject:
        dev_examples = dev_by_subject.get(subject, [])
        fewshot_cache[subject] = build_fewshot_prompt(dev_examples, subject)

    print("[mmlu] loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    base_model.eval()

    print("[mmlu] evaluating base model...")
    base_results = evaluate_model(base_model, tokenizer, questions, fewshot_cache, args.device)
    print(f"[mmlu] base overall accuracy: {base_results['overall_accuracy']:.4f}")

    ft_results = None
    if args.adapter_path:
        print("[mmlu] loading fine-tuned adapter...")
        from peft import PeftModel
        ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        ft_model.eval()
        print("[mmlu] evaluating fine-tuned model...")
        ft_results = evaluate_model(ft_model, tokenizer, questions, fewshot_cache, args.device)
        print(f"[mmlu] fine-tuned overall accuracy: {ft_results['overall_accuracy']:.4f}")
        delta = ft_results["overall_accuracy"] - base_results["overall_accuracy"]
        print(f"[mmlu] delta (ft - base): {delta:+.4f}")

    output = {
        "base": base_results,
        "finetuned": ft_results,
        "delta": {
            "overall_accuracy": (ft_results["overall_accuracy"] - base_results["overall_accuracy"])
            if ft_results else None,
            "per_group": {
                g: ft_results["per_group"][g] - base_results["per_group"][g]
                for g in base_results["per_group"]
            } if ft_results else None,
        },
        "config": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"[mmlu] saved -> {args.output}")


if __name__ == "__main__":
    main()
