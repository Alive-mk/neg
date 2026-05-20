from __future__ import annotations

import re
from dataclasses import dataclass
from copy import deepcopy
from typing import Iterable

from neg_blindness.schema import (
    EXPECTED_NEG_BEHAVIORS,
    NEG_TYPES,
    SCOPE_TYPES,
    SEMANTIC_MODES,
    ExperimentRecord,
    normalize_text,
)


NEGATION_MARKERS = (" not ", "n't", " no ", " never ", " without ", " none ")
NEGATION_REGEX = re.compile(r"\b(?:not|no|never|none|without)\b|n't")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def has_negation_marker(text: str) -> bool:
    padded = f" {normalize_text(text)} "
    return any(marker in padded for marker in NEGATION_MARKERS)


def negation_marker_count(text: str) -> int:
    return len(NEGATION_REGEX.findall(normalize_text(text)))


def _check_nonempty_list(name: str, items: Iterable[str], errors: list[str]) -> None:
    if not list(items):
        errors.append(f"{name} must be non-empty")


def validate_record(record: ExperimentRecord) -> ValidationResult:
    errors: list[str] = []

    if record.semantic_mode not in SEMANTIC_MODES:
        errors.append(f"invalid semantic_mode={record.semantic_mode}")
    if record.neg_type not in NEG_TYPES:
        errors.append(f"invalid neg_type={record.neg_type}")
    if record.scope_type not in SCOPE_TYPES:
        errors.append(f"invalid scope_type={record.scope_type}")
    if record.expected_neg_behavior not in EXPECTED_NEG_BEHAVIORS:
        errors.append(
            f"invalid expected_neg_behavior={record.expected_neg_behavior}"
        )

    if not record.prompt_pos.strip():
        errors.append("prompt_pos must be non-empty")
    if not record.prompt_neg.strip():
        errors.append("prompt_neg must be non-empty")
    if normalize_text(record.prompt_pos) == normalize_text(record.prompt_neg):
        errors.append("prompt_pos and prompt_neg must differ")
    if not has_negation_marker(record.prompt_neg):
        errors.append("prompt_neg must contain an explicit negation marker")
    if record.scope_type == "double_negation":
        neg_count = negation_marker_count(record.prompt_neg)
        if neg_count < 2 and "double" not in normalize_text(record.prompt_neg):
            errors.append("double_negation sample should express two negation cues")

    _check_nonempty_list("gold_pos", record.gold_pos, errors)
    _check_nonempty_list("forbidden_neg", record.forbidden_neg, errors)
    _check_nonempty_list("distractors", record.distractors, errors)

    norm_gold_pos = {normalize_text(x) for x in record.gold_pos}
    norm_gold_neg = {normalize_text(x) for x in record.gold_neg}
    norm_forbidden = {normalize_text(x) for x in record.forbidden_neg}
    norm_candidates = {normalize_text(x) for x in record.candidate_pool_neg}
    norm_distractors = {normalize_text(x) for x in record.distractors}

    if not norm_gold_pos.issubset(norm_forbidden):
        errors.append("forbidden_neg must include all gold_pos answers")
    if norm_candidates & norm_forbidden:
        errors.append("candidate_pool_neg must not overlap forbidden_neg")
    if norm_distractors & norm_forbidden:
        errors.append("distractors must not overlap forbidden_neg")

    if record.semantic_mode == "suppression_only":
        if record.gold_neg:
            errors.append("suppression_only samples should keep gold_neg empty")
        if record.expected_neg_behavior != "suppress_target":
            errors.append(
                "suppression_only samples must use expected_neg_behavior=suppress_target"
            )
        _check_nonempty_list("candidate_pool_neg", record.candidate_pool_neg, errors)

    if record.semantic_mode in {"contrastive_resolution", "exclusive_choice"}:
        if record.expected_neg_behavior != "preserve_positive":
            _check_nonempty_list("gold_neg", record.gold_neg, errors)

    if record.expected_neg_behavior == "select_gold_neg" and not record.gold_neg:
        errors.append("select_gold_neg samples require non-empty gold_neg")

    if record.expected_neg_behavior == "preserve_positive":
        if record.scope_type == "in_scope":
            errors.append("preserve_positive should only appear in scope-control cases")
        if norm_gold_neg and norm_gold_neg != norm_gold_pos:
            errors.append(
                "preserve_positive samples should keep gold_neg empty or equal to gold_pos"
            )

    if not record.template_id.strip():
        errors.append("template_id must be non-empty")
    if not record.family_id.strip():
        errors.append("family_id must be non-empty")
    if not record.entity_id.strip():
        errors.append("entity_id must be non-empty")

    return ValidationResult(ok=not errors, errors=errors)


def dedupe_records(records: list[ExperimentRecord]) -> tuple[list[ExperimentRecord], dict]:
    seen_keys: dict[str, str] = {}
    deduped: list[ExperimentRecord] = []
    duplicate_pairs: list[dict[str, str]] = []
    for record in records:
        key = record.canonical_key()
        if key in seen_keys:
            duplicate_pairs.append({"kept": seen_keys[key], "dropped": record.id})
            continue
        seen_keys[key] = record.id
        deduped.append(record)
    return deduped, {
        "input_records": len(records),
        "output_records": len(deduped),
        "duplicates_removed": len(records) - len(deduped),
        "duplicate_pairs": duplicate_pairs,
    }


def ensure_unique_record_ids(
    records: list[ExperimentRecord],
) -> tuple[list[ExperimentRecord], dict]:
    seen_counts: dict[str, int] = {}
    uniquified: list[ExperimentRecord] = []
    renamed: list[dict[str, str]] = []

    for record in records:
        original_id = record.id
        count = seen_counts.get(original_id, 0)
        seen_counts[original_id] = count + 1
        if count == 0:
            uniquified.append(record)
            continue

        updated = deepcopy(record)
        updated.id = f"{original_id}__dup{count + 1:03d}"
        updated.metadata = {
            **updated.metadata,
            "source_id": original_id,
        }
        uniquified.append(updated)
        renamed.append({"from": original_id, "to": updated.id})

    return uniquified, {
        "input_records": len(records),
        "output_records": len(uniquified),
        "duplicate_id_groups": sum(1 for count in seen_counts.values() if count > 1),
        "renamed_records": len(renamed),
        "renamed_samples": renamed[:100],
    }
