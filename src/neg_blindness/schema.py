from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any


SEMANTIC_MODES = {
    "suppression_only",
    "contrastive_resolution",
    "exclusive_choice",
}

NEG_TYPES = {
    "verb",
    "noun",
    "adverb",
    "sentential",
}

SCOPE_TYPES = {
    "in_scope",
    "out_of_scope",
    "double_negation",
}

EXPECTED_NEG_BEHAVIORS = {
    "suppress_target",
    "select_gold_neg",
    "preserve_positive",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


@dataclass
class ExperimentRecord:
    id: str
    neg_type: str
    scope_type: str
    semantic_mode: str
    expected_neg_behavior: str
    domain: str
    prompt_pos: str
    prompt_neg: str
    gold_pos: list[str]
    gold_neg: list[str]
    forbidden_neg: list[str]
    candidate_pool_neg: list[str]
    distractors: list[str]
    template_id: str
    family_id: str
    entity_id: str
    valid_negatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_key(self) -> str:
        key = " || ".join(
            [
                normalize_text(self.prompt_pos),
                normalize_text(self.prompt_neg),
                self.semantic_mode,
                self.neg_type,
                self.scope_type,
            ]
        )
        return sha1(key.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "neg_type": self.neg_type,
            "scope_type": self.scope_type,
            "semantic_mode": self.semantic_mode,
            "expected_neg_behavior": self.expected_neg_behavior,
            "domain": self.domain,
            "prompt_pos": self.prompt_pos,
            "prompt_neg": self.prompt_neg,
            "gold_pos": self.gold_pos,
            "gold_neg": self.gold_neg,
            "forbidden_neg": self.forbidden_neg,
            "candidate_pool_neg": self.candidate_pool_neg,
            "valid_negatives": self.valid_negatives,
            "distractors": self.distractors,
            "template_id": self.template_id,
            "family_id": self.family_id,
            "entity_id": self.entity_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        def _load_text_list(key: str) -> list[str]:
            values: list[str] = []
            for item in data.get(key, []):
                text = str(item)
                if text.strip():
                    values.append(text)
            return values

        record = cls(
            id=str(data.get("id") or ""),
            neg_type=str(data.get("neg_type") or ""),
            scope_type=str(data.get("scope_type") or ""),
            semantic_mode=str(data.get("semantic_mode") or ""),
            expected_neg_behavior=str(data.get("expected_neg_behavior") or ""),
            domain=str(data.get("domain") or ""),
            prompt_pos=str(data.get("prompt_pos") or ""),
            prompt_neg=str(data.get("prompt_neg") or ""),
            gold_pos=_load_text_list("gold_pos"),
            gold_neg=_load_text_list("gold_neg"),
            forbidden_neg=_load_text_list("forbidden_neg"),
            candidate_pool_neg=_load_text_list("candidate_pool_neg"),
            valid_negatives=_load_text_list("valid_negatives"),
            distractors=_load_text_list("distractors"),
            template_id=str(data.get("template_id") or ""),
            family_id=str(data.get("family_id") or ""),
            entity_id=str(data.get("entity_id") or ""),
            metadata=dict(data.get("metadata", {})),
        )
        return record.with_defaults()

    def with_defaults(self) -> "ExperimentRecord":
        if not self.id:
            base = " || ".join(
                [
                    self.prompt_pos,
                    self.prompt_neg,
                    self.template_id,
                    self.family_id,
                    self.entity_id,
                ]
            )
            self.id = sha1(base.encode("utf-8")).hexdigest()[:16]
        self.gold_pos = dedupe_preserve_order([x for x in self.gold_pos if x])
        self.gold_neg = dedupe_preserve_order([x for x in self.gold_neg if x])
        self.forbidden_neg = dedupe_preserve_order([x for x in self.forbidden_neg if x])
        self.candidate_pool_neg = dedupe_preserve_order(
            [x for x in self.candidate_pool_neg if x]
        )
        self.valid_negatives = dedupe_preserve_order(
            [x for x in self.valid_negatives if x]
        )
        self.distractors = dedupe_preserve_order([x for x in self.distractors if x])
        return self
