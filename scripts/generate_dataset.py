from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neg_blindness.api import ScoreCache, chat_json_request, load_model_configs
from neg_blindness.io_utils import read_json, write_json
from neg_blindness.prompts import GENERATOR_SYSTEM_PROMPT, generation_user_prompt


def build_specs(plan: dict) -> list[dict]:
    rng = random.Random(plan.get("random_seed", 42))
    target_examples = int(plan["target_examples"])
    batch_size = int(plan["batch_size"])
    total_batches = max(1, math.ceil(target_examples / batch_size))

    semantic_mix = plan["semantic_mode_mix"]
    semantic_choices = list(semantic_mix.keys())
    semantic_weights = list(semantic_mix.values())
    neg_types = list(plan["neg_types"])
    scope_types = list(plan["scope_types"])
    domains = list(plan["domains"])

    scope_weights = [plan.get("scope_weights", {}).get(s, 1.0) for s in scope_types]

    specs: list[dict] = []
    for _ in range(total_batches):
        scope_type = rng.choices(scope_types, weights=scope_weights, k=1)[0]
        if scope_type == "in_scope":
            semantic_mode = rng.choices(semantic_choices, weights=semantic_weights, k=1)[0]
        else:
            # out_of_scope and double_negation → preserve_positive behavior
            # allow contrastive_resolution too so gold_neg can still be set
            semantic_mode = rng.choices(
                ["exclusive_choice", "contrastive_resolution"],
                weights=[0.7, 0.3],
                k=1,
            )[0]
        specs.append(
            {
                "semantic_mode": semantic_mode,
                "neg_type": rng.choice(neg_types),
                "scope_type": scope_type,
                "domain": rng.choice(domains),
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="outputs/generation_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = read_json(args.plan)
    model_configs = load_model_configs(args.models)
    generator_name = plan["generation_model"]
    if generator_name not in model_configs:
        raise ValueError(f"generation model {generator_name} not found in model config")

    specs = build_specs(plan)
    seed_topics = read_json(ROOT / plan["seed_topics_path"])

    if args.dry_run:
        write_json(args.report, {"num_specs": len(specs), "spec_preview": specs[:20]})
        return

    generator_config = model_configs[generator_name]
    cache = ScoreCache("outputs/generation_cache", generator_name)
    failures: list[dict] = []
    combo_counter = Counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_rows = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for batch_idx, spec in enumerate(specs):
            seeds = seed_topics.get(spec["domain"], [])[:3]
            combo_key = f"{spec['semantic_mode']}|{spec['neg_type']}|{spec['scope_type']}|{spec['domain']}"
            combo_counter[combo_key] += 1
            try:
                payload = chat_json_request(
                    config=generator_config,
                    system_prompt=GENERATOR_SYSTEM_PROMPT,
                    user_prompt=generation_user_prompt(
                        spec=spec,
                        seeds=seeds,
                        batch_size=int(plan["batch_size"]),
                    ),
                    cache=cache,
                    max_retries=int(plan["max_retries_per_batch"]),
                )
                items = payload["items"] if isinstance(payload, dict) else payload
                if not isinstance(items, list):
                    raise ValueError("generator did not return a JSON list under items")
                for item in items:
                    metadata = dict(item.get("metadata", {}))
                    metadata["_generation_spec"] = spec
                    metadata["_batch_index"] = batch_idx
                    item["metadata"] = metadata
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    generated_rows += 1
                handle.flush()
                print(
                    f"[generate_dataset] batch={batch_idx + 1}/{len(specs)} "
                    f"items={len(items)} total_rows={generated_rows}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"batch_index": batch_idx, "spec": spec, "error": str(exc)})
                print(
                    f"[generate_dataset] batch={batch_idx + 1}/{len(specs)} "
                    f"ERROR={exc}",
                    flush=True,
                )

    write_json(
        args.report,
        {
            "generated_rows": generated_rows,
            "failed_batches": len(failures),
            "failures": failures[:100],
            "spec_distribution": dict(combo_counter),
        },
    )


if __name__ == "__main__":
    main()
