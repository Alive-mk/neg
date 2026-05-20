from __future__ import annotations

import json
from typing import Any


GENERATOR_SYSTEM_PROMPT = """You are building a high-quality negation reasoning dataset for a research paper.
Return only valid JSON. Do not wrap the answer in markdown.
Make each example semantically coherent and natural.
"""


def generation_user_prompt(spec: dict[str, Any], seeds: list[str], batch_size: int) -> str:
    scope_type = spec["scope_type"]
    preserve_positive_hint = (
        "All items must use expected_neg_behavior=preserve_positive. "
        "gold_neg should be empty or equal to gold_pos because the negation is cancelled, out of scope, or doubled."
        if scope_type in {"out_of_scope", "double_negation"}
        else "For in_scope items, expected_neg_behavior must be suppress_target or select_gold_neg depending on whether the negative answer is unique."
    )
    schema = {
        "items": [
            {
                "id": "string",
                "neg_type": spec["neg_type"],
                "scope_type": spec["scope_type"],
                "semantic_mode": spec["semantic_mode"],
                "expected_neg_behavior": "suppress_target | select_gold_neg | preserve_positive",
                "domain": spec["domain"],
                "prompt_pos": "positive prompt",
                "prompt_neg": "negated or scope-controlled prompt",
                "gold_pos": ["positive continuation"],
                "gold_neg": ["negative-only correct continuation if unique"],
                "forbidden_neg": ["positive target that should not stay high under negation"],
                "candidate_pool_neg": ["plausible non-target alternatives"],
                "distractors": ["clearly wrong options"],
                "template_id": "shared prompt template id",
                "family_id": "semantic family id",
                "entity_id": "main entity id",
                "metadata": {
                    "topic": "seed topic",
                    "notes": "short generation note"
                }
            }
        ]
    }
    double_neg_hint = (
        "For double_negation items: the prompt MUST contain exactly two explicit negation markers "
        "(e.g. 'not...never', 'cannot...without', 'doesn\\'t...no'). "
        "A single 'not' is NOT sufficient for double_negation."
        if spec["scope_type"] == "double_negation"
        else ""
    )
    requirements = [
        f"Generate exactly {batch_size} items.",
        f"All items must use semantic_mode={spec['semantic_mode']}.",
        f"All items must use neg_type={spec['neg_type']}.",
        f"All items must use scope_type={spec['scope_type']}.",
        f"All items must use domain={spec['domain']}.",
        "Use explicit English prompts because the evaluation models will be scored on continuations.",
        "Keep prompt_pos and prompt_neg minimally different, but make the semantic change clear.",
        preserve_positive_hint,
        "CRITICAL: forbidden_neg MUST contain every string that appears in gold_pos. "
        "This is the most important rule — failure to include gold_pos in forbidden_neg will invalidate the item.",
        "For suppression_only: gold_neg MUST be an empty list [], forbidden_neg MUST equal gold_pos, "
        "expected_neg_behavior MUST be 'suppress_target', and candidate_pool_neg must be non-empty.",
        "For contrastive_resolution or exclusive_choice with in_scope negation: gold_neg must be non-empty and uniquely correct.",
        "For preserve_positive cases: forbidden_neg must still include the positive target, but candidate_pool_neg and distractors must not overlap forbidden_neg.",
        "candidate_pool_neg and distractors must be distinct from forbidden_neg in every item.",
        double_neg_hint,
        "Avoid malformed double negation like bare 'not not' questions unless they read naturally; scope-control prompts should still be fluent English.",
        "Each item must be self-contained without requiring external context.",
        "Avoid duplicates and trivial paraphrases.",
    ]
    requirements = [r for r in requirements if r]  # drop empty strings
    prompt = {
        "task": "Generate negation-reasoning dataset items.",
        "seed_topics": seeds,
        "requirements": requirements,
        "output_schema": schema,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


VERIFIER_SYSTEM_PROMPT = """You verify whether a negation-reasoning example is semantically consistent.
Return JSON only with fields:
{
  "pass": true or false,
  "score": number between 0 and 1,
  "reasons": ["short reason"]
}
"""


def verifier_user_prompt(item: dict[str, Any]) -> str:
    instructions = {
        "checklist": [
            "prompt_pos and prompt_neg must reflect a meaningful semantic difference",
            "gold_pos must match prompt_pos",
            "forbidden_neg must include the positive target",
            "semantic_mode-specific constraints must hold",
            "preserve_positive should only be used when negation is cancelled or out of scope",
            "candidate_pool_neg and distractors must be distinct from forbidden_neg",
        ],
        "item": item,
    }
    return json.dumps(instructions, ensure_ascii=False, indent=2)
