from __future__ import annotations

import json
from typing import Any

from neg_blindness.api import ModelConfig, ScoreCache, chat_json_request, score_continuation
from neg_blindness.metrics import bootstrap_ci
from neg_blindness.schema import ExperimentRecord


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_candidate_sets(record: ExperimentRecord) -> tuple[list[str], list[str]]:
    pos_candidates = _unique(
        record.gold_pos + record.gold_neg + record.candidate_pool_neg + record.distractors
    )

    if record.expected_neg_behavior == "select_gold_neg":
        neg_candidates = _unique(
            record.gold_neg
            + record.valid_negatives
            + record.forbidden_neg
            + record.candidate_pool_neg
            + record.distractors
        )
    elif record.expected_neg_behavior == "preserve_positive":
        neg_candidates = _unique(
            record.gold_pos
            + record.gold_neg
            + record.candidate_pool_neg
            + record.distractors
        )
    else:
        # For suppress_target with no explicit forbidden_neg (e.g. WikiFact OOD),
        # fall back to gold_pos as the item to suppress so neg_suppressed is computable.
        effective_forbidden = record.forbidden_neg if record.forbidden_neg else record.gold_pos
        neg_candidates = _unique(
            effective_forbidden + record.candidate_pool_neg + record.distractors
        )

    return pos_candidates, neg_candidates


def _choice_prompt(prompt: str, candidates: list[str]) -> tuple[str, str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(candidates) > len(letters):
        raise ValueError("too many candidates for choice mode")
    option_map = {letters[idx]: candidate for idx, candidate in enumerate(candidates)}
    system = (
        "You are evaluating continuation candidates. "
        "Return JSON only in the format {\"answer\": \"A\"}."
    )
    user = {
        "prompt": prompt,
        "question": "Which continuation best completes the prompt?",
        "options": option_map,
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def score_candidates(
    config: ModelConfig,
    prompt: str,
    candidates: list[str],
    cache: ScoreCache | None,
) -> dict[str, float]:
    if config.mode in {"logprob", "hf_local"}:
        return {
            candidate: score_continuation(config, prompt, candidate, cache=cache)
            for candidate in candidates
        }
    if config.mode in {"choice", "chat_json"}:
        system, user = _choice_prompt(prompt, candidates)
        result = chat_json_request(config, system, user, cache=cache)
        answer = str(result["answer"]).strip().upper()
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        selected_idx = letters.index(answer)
        selected_candidate = candidates[selected_idx]
        return {candidate: (1.0 if candidate == selected_candidate else 0.0) for candidate in candidates}
    raise ValueError(f"Unsupported evaluation mode: {config.mode}")


def _best_candidate(scores: dict[str, float]) -> str:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[0][0]


def evaluate_record(
    record: ExperimentRecord,
    config: ModelConfig,
    cache: ScoreCache | None = None,
    neg_prefix: str = "",
    use_multi_answer_negatives: bool = False,
) -> dict[str, Any]:
    pos_candidates, neg_candidates = build_candidate_sets(record)
    pos_scores = score_candidates(config, record.prompt_pos, pos_candidates, cache)
    prompt_neg = f"{neg_prefix} {record.prompt_neg}" if neg_prefix else record.prompt_neg
    neg_scores = score_candidates(config, prompt_neg, neg_candidates, cache)

    pos_best = _best_candidate(pos_scores)
    neg_best = _best_candidate(neg_scores)

    pos_correct = pos_best in set(record.gold_pos)

    effective_forbidden = (
        record.forbidden_neg if record.forbidden_neg
        else (record.gold_pos if record.expected_neg_behavior == "suppress_target" else [])
    )
    forbidden_scores = [neg_scores[item] for item in effective_forbidden if item in neg_scores]
    allowed_scores = [
        score
        for item, score in neg_scores.items()
        if item not in set(effective_forbidden)
    ]
    neg_suppressed = bool(allowed_scores) and bool(forbidden_scores) and max(allowed_scores) > max(forbidden_scores)
    neg_rank_correct = neg_best in set(record.gold_neg)
    valid_negative_set = set(record.gold_neg + record.valid_negatives)
    multi_neg_rank_correct = neg_best in valid_negative_set
    select_ambiguous = (
        record.expected_neg_behavior == "select_gold_neg"
        and len(valid_negative_set) > len(set(record.gold_neg))
    )
    preserve_positive = neg_best in set(record.gold_pos)

    if record.expected_neg_behavior == "select_gold_neg":
        neg_correct = multi_neg_rank_correct if use_multi_answer_negatives else neg_rank_correct
    elif record.expected_neg_behavior == "preserve_positive":
        neg_correct = preserve_positive
    else:
        neg_correct = neg_suppressed

    distractor_reject = (pos_best not in set(record.distractors)) and (
        neg_best not in set(record.distractors)
    )
    flip_required = record.expected_neg_behavior in {"suppress_target", "select_gold_neg"}
    flip_correct = flip_required and pos_correct and neg_correct
    over_negation = (record.expected_neg_behavior == "preserve_positive") and (not preserve_positive)

    return {
        "id": record.id,
        "semantic_mode": record.semantic_mode,
        "scope_type": record.scope_type,
        "expected_neg_behavior": record.expected_neg_behavior,
        "pos_best": pos_best,
        "neg_best": neg_best,
        "pos_correct": pos_correct,
        "neg_suppressed": neg_suppressed,
        "neg_rank_correct": neg_rank_correct,
        "multi_neg_rank_correct": multi_neg_rank_correct,
        "select_ambiguous": select_ambiguous,
        "preserve_positive": preserve_positive,
        "neg_correct": neg_correct,
        "flip_correct": flip_correct,
        "flip_required": flip_required,
        "distractor_reject": distractor_reject,
        "over_negation": over_negation,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    pos_values = [item["pos_correct"] for item in results]
    neg_suppression_values = [
        item["neg_suppressed"]
        for item in results
        if item["expected_neg_behavior"] == "suppress_target"
    ]
    neg_rank_values = [
        item["neg_rank_correct"]
        for item in results
        if item["expected_neg_behavior"] == "select_gold_neg"
    ]
    multi_neg_rank_values = [
        item.get("multi_neg_rank_correct", item["neg_rank_correct"])
        for item in results
        if item["expected_neg_behavior"] == "select_gold_neg"
    ]
    select_ambiguous_values = [
        item.get("select_ambiguous", False)
        for item in results
        if item["expected_neg_behavior"] == "select_gold_neg"
    ]
    flip_values = [item["flip_correct"] for item in results if item["flip_required"]]
    distractor_values = [item["distractor_reject"] for item in results]
    scope_control_values = [
        item["preserve_positive"]
        for item in results
        if item["expected_neg_behavior"] == "preserve_positive"
    ]
    double_negation_values = [
        item["preserve_positive"]
        for item in results
        if item["scope_type"] == "double_negation"
    ]
    over_negation_values = [
        item["over_negation"]
        for item in results
        if item["expected_neg_behavior"] == "preserve_positive"
    ]

    return {
        "count": len(results),
        "PosAcc": bootstrap_ci(pos_values),
        "NegSuppRate": bootstrap_ci(neg_suppression_values),
        "NegRankAcc": bootstrap_ci(neg_rank_values),
        "MultiAnswerNegRankAcc": bootstrap_ci(multi_neg_rank_values),
        "SelectAmbiguousRate": bootstrap_ci(select_ambiguous_values),
        "FlipAcc": bootstrap_ci(flip_values),
        "DistractorReject": bootstrap_ci(distractor_values),
        "ScopeControlAcc": bootstrap_ci(scope_control_values),
        "DoubleNegationAcc": bootstrap_ci(double_negation_values),
        "OverNegationRate": bootstrap_ci(over_negation_values),
    }
