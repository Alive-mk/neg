"""Generate mechanistic-analysis probe records.

Each record uses a short incomplete prompt (e.g. "The sky is") and
single-token word candidates (e.g. " blue").  All candidates are verified
against the Qwen2.5-7B tokenizer before being written.

Usage:
    python scripts/generate_mech_probes.py \
        --models configs/model_config.json \
        --tokenizer model/Qwen2.5-7B \
        --output data/manual/mech_probes.jsonl \
        --target 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neg_blindness.api import load_model_configs


# ---------------------------------------------------------------------------
# Single-token check
# ---------------------------------------------------------------------------

def make_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    return tok


def is_single_token(tokenizer: Any, text: str) -> bool:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids) == 1


def all_single_token(tokenizer: Any, texts: list[str]) -> bool:
    return all(is_single_token(tokenizer, t) for t in texts)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You generate mechanistic-analysis probe records for a negation-blindness study.
Rules:
- prompt_pos: a short INCOMPLETE English sentence that expects a single-word adjective or noun continuation.
  Example: "The fire is", "The whale is", "The knife is"
- prompt_neg: the same prompt with a negation word inserted.
  Example: "The fire is not", "The whale is not", "The knife is not"
- gold_pos: exactly ONE single common English word (with a leading space) that naturally completes prompt_pos.
  Example: [" hot"], [" large"], [" sharp"]
- forbidden_neg: must equal gold_pos — the word that should be SUPPRESSED under negation.
- candidate_pool_neg: 2–3 single words (with leading space) that are plausible completions UNDER NEGATION.
  These must NOT be in forbidden_neg.
- distractors: 1–2 single words that are clearly unrelated (wrong category).
- ALL candidate strings (gold_pos, forbidden_neg, candidate_pool_neg, distractors) must be single common
  English words that tokenize to EXACTLY ONE subword token in standard BPE tokenizers.
  Use only short, common adjectives and nouns: hot, cold, warm, large, small, sharp, soft, hard, wet, dry,
  fast, slow, loud, quiet, light, dark, heavy, old, new, red, blue, black, white, tall, short, etc.
- neg_type: the grammatical negation type. Use "sentential" when "not" is added after a copula.
  Use "verb" when the main verb is negated (e.g., "doesn't", "cannot"). Use "adverb" when an adverb like
  "never" or "rarely" is used. Use "noun" when a noun phrase is negated (e.g., "no fire").
- scope_type: use "in_scope" for standard negation, "out_of_scope" for cases where negation is semantically
  cancelled or irrelevant, "double_negation" for double negation.
- semantic_mode: use "suppression_only" (gold_neg empty) when there is no unique correct answer under negation,
  "contrastive_resolution" when there IS a unique correct answer under negation.
- expected_neg_behavior: "suppress_target" for suppression_only, "select_gold_neg" for contrastive_resolution.
- domain: "commonsense", "factual", "lexical", or "physical".
- template_id, family_id, entity_id: short snake_case identifiers.

Return ONLY a JSON object with key "items" containing an array of records.
"""

RECORD_SCHEMA = {
    "id": "string",
    "neg_type": "sentential|verb|adverb|noun",
    "scope_type": "in_scope|out_of_scope|double_negation",
    "semantic_mode": "suppression_only|contrastive_resolution",
    "expected_neg_behavior": "suppress_target|select_gold_neg",
    "domain": "commonsense|factual|lexical|physical",
    "prompt_pos": "...",
    "prompt_neg": "...",
    "gold_pos": ["single word with leading space"],
    "gold_neg": ["single word or empty list"],
    "forbidden_neg": ["same as gold_pos"],
    "candidate_pool_neg": ["word1", "word2"],
    "distractors": ["word"],
    "template_id": "...",
    "family_id": "...",
    "entity_id": "...",
}

SEED_BATCHES = [
    {
        "topic": "temperature and heat",
        "entities": ["fire", "oven", "ice", "snow", "soup", "coffee", "winter", "summer"],
        "target_words": ["hot", "cold", "warm", "cool", "frozen", "boiling"],
    },
    {
        "topic": "size and weight",
        "entities": ["elephant", "ant", "mountain", "pebble", "whale", "mouse", "castle", "coin"],
        "target_words": ["large", "small", "heavy", "light", "tall", "short", "huge", "tiny"],
    },
    {
        "topic": "physical properties",
        "entities": ["knife", "pillow", "rock", "foam", "glass", "cotton", "steel", "butter"],
        "target_words": ["sharp", "soft", "hard", "smooth", "rough", "brittle", "flexible"],
    },
    {
        "topic": "speed and motion",
        "entities": ["cheetah", "turtle", "rocket", "snail", "bullet", "glacier", "river"],
        "target_words": ["fast", "slow", "quick", "swift", "rapid", "sluggish"],
    },
    {
        "topic": "color and light",
        "entities": ["sky", "coal", "snow", "grass", "sun", "night", "blood", "milk"],
        "target_words": ["blue", "black", "white", "green", "yellow", "dark", "bright", "red"],
    },
    {
        "topic": "sound and noise",
        "entities": ["library", "concert", "whisper", "thunder", "mouse", "drum"],
        "target_words": ["quiet", "loud", "silent", "noisy", "soft", "harsh"],
    },
    {
        "topic": "moisture and dryness",
        "entities": ["desert", "ocean", "sponge", "towel", "rain", "cactus", "fish"],
        "target_words": ["dry", "wet", "damp", "moist", "arid", "soaked"],
    },
    {
        "topic": "age and freshness",
        "entities": ["antique", "puppy", "fossil", "baby", "ruin", "flower", "bread"],
        "target_words": ["old", "young", "new", "ancient", "fresh", "stale", "aged"],
    },
    {
        "topic": "danger and safety",
        "entities": ["lion", "lamb", "poison", "medicine", "gun", "toy", "cliff"],
        "target_words": ["dangerous", "safe", "harmful", "gentle", "lethal", "benign"],
    },
    {
        "topic": "factual properties — geography",
        "entities": ["Sahara", "Amazon", "Arctic", "equator", "Himalayas", "Dead Sea"],
        "target_words": ["hot", "wet", "cold", "deep", "high", "low", "vast", "narrow"],
    },
]

NEG_TYPE_BATCHES = [
    {
        "neg_type": "sentential",
        "instruction": (
            'Use "is not" or "are not" as negation. '
            'prompt_pos ends with "is" or "are", prompt_neg ends with "is not" or "are not".'
        ),
    },
    {
        "neg_type": "verb",
        "instruction": (
            'Use auxiliary-verb negation like "does not", "cannot", "will not". '
            'prompt_pos uses affirmative verb form, prompt_neg negates it. '
            'Example pos: "Fire burns", neg: "Fire does not burn". '
            'gold_pos is an adjective or noun that naturally follows both prompts as a continuation — '
            'rephrase so that the continuation is still a single word.'
        ),
    },
    {
        "neg_type": "adverb",
        "instruction": (
            'Use adverb negation like "never", "rarely", "seldom", "hardly". '
            'prompt_pos: "X is always ...", prompt_neg: "X is never ..." — continuation is a single word.'
        ),
    },
    {
        "neg_type": "noun",
        "instruction": (
            'Use "no" before a noun in the negative version. '
            'Example pos: "There is fire in the room", neg: "There is no fire in the room —" '
            'but rephrase so the continuation is still a single word completing a sentence fragment.'
        ),
    },
]


def call_api(config: Any, user_content: str) -> str:
    api_key = config.resolved_api_key()
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens or 3000,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        config.endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=config.timeout_sec) as resp:
        return resp.read().decode("utf-8")


def build_user_prompt(seed: dict, neg_type_spec: dict, batch_size: int) -> str:
    return json.dumps(
        {
            "task": "Generate mechanistic probe records.",
            "seed_topic": seed["topic"],
            "seed_entities": seed["entities"],
            "target_words_hint": seed.get("target_words", []),
            "neg_type": neg_type_spec["neg_type"],
            "neg_type_instruction": neg_type_spec["instruction"],
            "batch_size": batch_size,
            "output_schema": RECORD_SCHEMA,
            "important": [
                "ALL candidate words must be short common English words with leading space.",
                "All candidates must tokenize to exactly ONE BPE subword token.",
                "Prefer: hot cold warm cool large small heavy light sharp soft hard fast slow quiet loud wet dry old new red blue black white tall short.",
                "Do NOT use multi-word phrases or sentences as candidates.",
                f"Generate exactly {batch_size} items.",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


NEG_PATTERN = re.compile(r"\b(?:not|no|never|none|without|rarely|seldom|hardly)\b|n't|does not|do not|cannot|will not|did not", re.IGNORECASE)


def validate_record(rec: dict, tokenizer: Any) -> tuple[bool, str]:
    required = ["id", "neg_type", "scope_type", "semantic_mode", "expected_neg_behavior",
                "domain", "prompt_pos", "prompt_neg", "gold_pos", "forbidden_neg",
                "candidate_pool_neg", "distractors", "template_id", "family_id", "entity_id"]
    for field in required:
        if field not in rec:
            return False, f"missing field: {field}"

    all_cands = (
        list(rec.get("gold_pos", []))
        + list(rec.get("gold_neg", []))
        + list(rec.get("forbidden_neg", []))
        + list(rec.get("candidate_pool_neg", []))
        + list(rec.get("distractors", []))
    )
    if not all_cands:
        return False, "no candidates"

    for cand in all_cands:
        if not is_single_token(tokenizer, cand):
            ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            return False, f"multi-token candidate {cand!r} -> {ids}"

    if not NEG_PATTERN.search(rec.get("prompt_neg", "")):
        return False, "no negation token detected in prompt_neg"

    gold_pos = set(rec.get("gold_pos", []))
    forbidden = set(rec.get("forbidden_neg", []))
    if not gold_pos:
        return False, "gold_pos is empty"
    if not forbidden:
        return False, "forbidden_neg is empty"
    if not gold_pos & forbidden:
        return False, "forbidden_neg must overlap gold_pos"

    pool = set(rec.get("candidate_pool_neg", []))
    if pool & forbidden:
        return False, "candidate_pool_neg overlaps forbidden_neg"

    mode = rec.get("semantic_mode", "")
    behavior = rec.get("expected_neg_behavior", "")
    gold_neg = rec.get("gold_neg", [])
    if mode == "suppression_only" and behavior != "suppress_target":
        return False, "suppression_only must use suppress_target"
    if mode == "contrastive_resolution" and behavior != "select_gold_neg":
        return False, "contrastive_resolution must use select_gold_neg"
    if mode == "contrastive_resolution" and not gold_neg:
        return False, "contrastive_resolution must have gold_neg"

    return True, "ok"


def dedup_key(rec: dict) -> str:
    raw = " || ".join([
        rec.get("prompt_pos", "").strip().lower(),
        rec.get("prompt_neg", "").strip().lower(),
    ])
    return sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    configs = load_model_configs(args.models)
    if "generator" not in configs:
        print("ERROR: need a 'generator' model in model_config.json", file=sys.stderr)
        sys.exit(1)
    gen_config = configs["generator"]

    print(f"Loading tokenizer from {args.tokenizer} ...", flush=True)
    tokenizer = make_tokenizer(args.tokenizer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: set[str] = set()
    accepted: list[dict] = []

    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                accepted.append(rec)
                seen_keys.add(dedup_key(rec))
        print(f"Resuming: {len(accepted)} records already written.", flush=True)

    seed_idx = 0
    neg_type_idx = 0
    retry_count = 0

    while len(accepted) < args.target:
        seed = SEED_BATCHES[seed_idx % len(SEED_BATCHES)]
        neg_spec = NEG_TYPE_BATCHES[neg_type_idx % len(NEG_TYPE_BATCHES)]
        user_prompt = build_user_prompt(seed, neg_spec, args.batch_size)

        try:
            raw_resp = call_api(gen_config, user_prompt)
            resp_json = json.loads(raw_resp)
            content = resp_json["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise ValueError("items is not a list")
        except Exception as exc:  # noqa: BLE001
            retry_count += 1
            print(f"  API error ({exc}), retry {retry_count}", flush=True)
            if retry_count >= args.max_retries:
                seed_idx += 1
                neg_type_idx += 1
                retry_count = 0
            time.sleep(2)
            continue

        retry_count = 0
        batch_accepted = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            key = dedup_key(item)
            if key in seen_keys:
                continue
            ok, reason = validate_record(item, tokenizer)
            if not ok:
                print(f"  skip [{item.get('id', '?')}]: {reason}", flush=True)
                continue
            seen_keys.add(key)
            accepted.append(item)
            batch_accepted += 1
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        total = len(accepted)
        print(
            f"[seed={seed_idx % len(SEED_BATCHES)} neg={neg_spec['neg_type']}] "
            f"+{batch_accepted} -> total {total}/{args.target}",
            flush=True,
        )

        seed_idx += 1
        neg_type_idx += 1
        time.sleep(0.5)

    print(f"\nDone. {len(accepted)} records written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
