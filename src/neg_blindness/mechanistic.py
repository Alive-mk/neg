from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

from neg_blindness.api import ModelConfig
from neg_blindness.evaluation import build_candidate_sets
from neg_blindness.schema import ExperimentRecord


NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|none|without)\b|n't", re.IGNORECASE)


@dataclass
class CandidateTokenInfo:
    text: str
    token_ids: list[int]

    @property
    def first_token_id(self) -> int:
        return self.token_ids[0]

    @property
    def is_single_token(self) -> bool:
        return len(self.token_ids) == 1


class LocalMechanisticRunner:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Mechanistic analysis requires torch and transformers."
            ) from exc

        self.torch = torch
        self.config = config
        model_path = config.model_path or config.model
        if not model_path:
            raise ValueError(f"{config.name} is missing model_path for hf_local mode")

        dtype = self._resolve_dtype(config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        if config.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Loading adapter_path requires peft to be installed."
                ) from exc
            self.model = PeftModel.from_pretrained(
                self.model,
                config.adapter_path,
            )
        self.model.eval()
        self.final_norm = self._resolve_final_norm()
        self.layer_modules = self._resolve_layers()
        self.attention_model: Any | None = None

    def _resolve_dtype(self, dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        if not hasattr(self.torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return getattr(self.torch, dtype_name)

    def _resolve_layers(self) -> list[Any]:
        # plain HF model: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # PEFT/LoRA wrapper: base_model.model.model.layers
        if (
            hasattr(self.model, "base_model")
            and hasattr(self.model.base_model, "model")
            and hasattr(self.model.base_model.model, "model")
            and hasattr(self.model.base_model.model.model, "layers")
        ):
            return list(self.model.base_model.model.model.layers)
        raise ValueError("Unsupported architecture: could not locate decoder layers")

    def _resolve_final_norm(self) -> Any | None:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if (
            hasattr(self.model, "base_model")
            and hasattr(self.model.base_model, "model")
            and hasattr(self.model.base_model.model, "model")
            and hasattr(self.model.base_model.model.model, "norm")
        ):
            return self.model.base_model.model.model.norm
        return None

    def _model_input_device(self) -> Any:
        try:
            return self.model.device
        except Exception:  # noqa: BLE001
            return next(self.model.parameters()).device

    def encode_prompt(self, prompt: str, include_offsets: bool = False) -> dict[str, Any]:
        kwargs = {
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if include_offsets:
            kwargs["return_offsets_mapping"] = True
        encoded = self.tokenizer(prompt, **kwargs)
        device = self._model_input_device()
        encoded["input_ids"] = encoded["input_ids"].to(device)
        encoded["attention_mask"] = encoded["attention_mask"].to(device)
        return encoded

    def _load_attention_model(self) -> Any:
        if self.attention_model is not None:
            return self.attention_model
        from transformers import AutoModelForCausalLM

        model_path = self.config.model_path or self.config.model
        dtype = self._resolve_dtype(self.config.torch_dtype)
        # Put attention model on a separate GPU from the main model to avoid OOM.
        # With CUDA_VISIBLE_DEVICES=0,1 the main model lands on cuda:0 and this
        # model lands on cuda:1 (or falls back to auto if only one GPU is visible).
        import torch as _torch
        attn_device = (
            "cuda:1"
            if _torch.cuda.is_available() and _torch.cuda.device_count() >= 2
            else "auto"
        )
        self.attention_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=attn_device,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation="eager",
        )
        if self.config.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Loading adapter_path requires peft to be installed."
                ) from exc
            self.attention_model = PeftModel.from_pretrained(
                self.attention_model,
                self.config.adapter_path,
            )
        self.attention_model.eval()
        return self.attention_model

    def forward(
        self,
        encoded: dict[str, Any],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Any:
        with self.torch.no_grad():
            return self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=False,
            )

    def forward_attention(self, prompt: str) -> Any:
        attention_model = self._load_attention_model()
        try:
            device = attention_model.device
        except Exception:  # noqa: BLE001
            device = next(attention_model.parameters()).device
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with self.torch.no_grad():
            return attention_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=True,
                use_cache=False,
            )

    def _resolve_lm_head(self) -> Any:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            if hasattr(self.model.base_model.model, "lm_head"):
                return self.model.base_model.model.lm_head
        raise ValueError("Could not locate lm_head")

    def apply_logit_lens(self, hidden_state: Any) -> Any:
        if self.final_norm is not None:
            hidden_state = self.final_norm(hidden_state)
        return self._resolve_lm_head()(hidden_state)

    def tokenize_candidate(self, text: str) -> CandidateTokenInfo:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"candidate tokenized to empty sequence: {text!r}")
        return CandidateTokenInfo(text=text, token_ids=list(token_ids))

    def patch_final_from_source(
        self,
        encoded: dict[str, Any],
        layer_index: int,
        source_vector: Any,
        blend_weight: float = 1.0,
    ) -> Any:
        layer_module = self.layer_modules[layer_index]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()
            patched = hidden_states.clone()
            patched[:, -1, :] = (
                blend_weight * source_vector.to(patched.device)
                + (1.0 - blend_weight) * patched[:, -1, :]
            )
            if isinstance(output, tuple):
                return (patched, *rest)
            return patched

        handle = layer_module.register_forward_hook(hook)
        try:
            return self.forward(encoded, output_hidden_states=False, output_attentions=False)
        finally:
            handle.remove()


def _candidate_score_summary(logits: Any, token_ids: list[int]) -> dict[str, float]:
    values = [float(logits[token_id].item()) for token_id in token_ids]
    return {
        "mean": mean(values),
        "max": max(values),
    }


def _char_spans_for_negation(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in NEGATION_PATTERN.finditer(text)]


def find_negation_token_positions(
    prompt: str,
    offsets: list[tuple[int, int]],
) -> list[int]:
    spans = _char_spans_for_negation(prompt)
    positions: list[int] = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                positions.append(idx)
                break
    return positions


def summarize_layer_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"first": 0.0, "peak": 0.0, "last": 0.0, "delta_last_first": 0.0}
    peak = max(values)
    return {
        "first": values[0],
        "peak": peak,
        "last": values[-1],
        "delta_last_first": values[-1] - values[0],
        "delta_last_peak": values[-1] - peak,
    }


def analyze_record_mechanistically(
    record: ExperimentRecord,
    runner: LocalMechanisticRunner,
    patch_blend_weight: float = 1.0,
) -> dict[str, Any]:
    pos_candidates, neg_candidates = build_candidate_sets(record)
    encoded_neg = runner.encode_prompt(record.prompt_neg, include_offsets=True)
    encoded_pos = runner.encode_prompt(record.prompt_pos, include_offsets=False)

    offsets = [tuple(item) for item in encoded_neg["offset_mapping"][0].tolist()]
    neg_token_positions = find_negation_token_positions(record.prompt_neg, offsets)
    if not neg_token_positions:
        raise ValueError(f"could not align negation token positions for record {record.id}")

    candidate_infos = {
        text: runner.tokenize_candidate(text)
        for text in sorted(set(pos_candidates + neg_candidates))
    }
    multi_token_candidates = {
        text: info.token_ids
        for text, info in candidate_infos.items()
        if not info.is_single_token
    }
    if multi_token_candidates:
        raise ValueError(
            f"multi-token candidates are not supported for mechanism analysis: "
            f"{multi_token_candidates}"
        )

    neg_allowed = [text for text in neg_candidates if text not in set(record.forbidden_neg)]
    pos_competitors = [text for text in pos_candidates if text not in set(record.gold_pos)]
    if not neg_allowed:
        raise ValueError(f"record {record.id} has no allowed negative candidates")
    if not pos_competitors:
        raise ValueError(f"record {record.id} has no positive competitors")

    neg_outputs = runner.forward(
        encoded_neg,
        output_hidden_states=True,
        output_attentions=False,
    )
    pos_outputs = runner.forward(
        encoded_pos,
        output_hidden_states=True,
        output_attentions=False,
    )
    attention_outputs = runner.forward_attention(record.prompt_neg)

    neg_hidden_states = list(neg_outputs.hidden_states[1:])
    pos_hidden_states = list(pos_outputs.hidden_states[1:])
    if len(neg_hidden_states) != len(runner.layer_modules):
        raise ValueError("unexpected hidden state depth from model forward")

    e1_layers: list[dict[str, Any]] = []
    e2_layers: list[dict[str, Any]] = []
    baseline_neg_margin = 0.0

    for layer_index, hidden_state in enumerate(neg_hidden_states):
        logits = runner.apply_logit_lens(hidden_state[:, -1, :])[0]
        forbidden_max = max(
            _candidate_score_summary(logits, [candidate_infos[text].first_token_id])["max"]
            for text in record.forbidden_neg
        )
        allowed_scores = [
            _candidate_score_summary(logits, [candidate_infos[text].first_token_id])["mean"]
            for text in neg_allowed
        ]
        neg_best_text = max(
            neg_candidates,
            key=lambda text: float(logits[candidate_infos[text].first_token_id].item()),
        )
        neg_margin = mean(allowed_scores) - forbidden_max
        if layer_index == len(neg_hidden_states) - 1:
            baseline_neg_margin = neg_margin

        pos_logits = runner.apply_logit_lens(pos_hidden_states[layer_index][:, -1, :])[0]
        pos_target_scores = [
            _candidate_score_summary(pos_logits, [candidate_infos[text].first_token_id])["mean"]
            for text in record.gold_pos
        ]
        pos_competitor_scores = [
            _candidate_score_summary(pos_logits, [candidate_infos[text].first_token_id])["mean"]
            for text in pos_competitors
        ]
        pos_best_text = max(
            pos_candidates,
            key=lambda text: float(pos_logits[candidate_infos[text].first_token_id].item()),
        )

        e1_layers.append(
            {
                "layer": layer_index,
                "neg_margin": neg_margin,
                "forbidden_max_logit": forbidden_max,
                "allowed_mean_logit": mean(allowed_scores),
                "neg_best_candidate": neg_best_text,
                "neg_best_is_allowed": neg_best_text in set(neg_allowed),
                "pos_margin": mean(pos_target_scores) - max(pos_competitor_scores),
                "pos_best_candidate": pos_best_text,
                "pos_best_is_gold": pos_best_text in set(record.gold_pos),
            }
        )

    for layer_index, layer_attention in enumerate(attention_outputs.attentions):
        raw_values = [
            float(layer_attention[0, head_idx, -1, neg_token_positions].mean().item())
            for head_idx in range(layer_attention.shape[1])
        ]
        head_values = [v for v in raw_values if v == v]  # drop NaN
        if not head_values:
            head_values = [0.0]
        top_heads = sorted(
            (
                {"head": head_idx, "attention_to_negation": value}
                for head_idx, value in enumerate(head_values)
            ),
            key=lambda item: item["attention_to_negation"],
            reverse=True,
        )[:3]
        e2_layers.append(
            {
                "layer": layer_index,
                "mean_attention_to_negation": mean(head_values),
                "max_head_attention_to_negation": max(head_values),
                "top_heads": top_heads,
            }
        )

    e3_layers: list[dict[str, Any]] = []
    for layer_index, hidden_state in enumerate(neg_hidden_states):
        source_vector = hidden_state[0, neg_token_positions, :].mean(dim=0)
        patched_outputs = runner.patch_final_from_source(
            encoded=encoded_neg,
            layer_index=layer_index,
            source_vector=source_vector,
            blend_weight=patch_blend_weight,
        )
        patched_logits = patched_outputs.logits[0, -1, :]
        patched_forbidden_max = max(
            float(patched_logits[candidate_infos[text].first_token_id].item())
            for text in record.forbidden_neg
        )
        patched_allowed_scores = [
            float(patched_logits[candidate_infos[text].first_token_id].item())
            for text in neg_allowed
        ]
        patched_margin = mean(patched_allowed_scores) - patched_forbidden_max
        patched_best_text = max(
            neg_candidates,
            key=lambda text: float(patched_logits[candidate_infos[text].first_token_id].item()),
        )
        e3_layers.append(
            {
                "layer": layer_index,
                "baseline_neg_margin": baseline_neg_margin,
                "patched_neg_margin": patched_margin,
                "recovery": patched_margin - baseline_neg_margin,
                "patched_best_candidate": patched_best_text,
                "patched_best_is_allowed": patched_best_text in set(neg_allowed),
            }
        )

    return {
        "id": record.id,
        "scope_type": record.scope_type,
        "semantic_mode": record.semantic_mode,
        "expected_neg_behavior": record.expected_neg_behavior,
        "prompt_neg": record.prompt_neg,
        "negation_token_positions": neg_token_positions,
        "negation_token_text": [
            record.prompt_neg[offsets[pos][0] : offsets[pos][1]]
            for pos in neg_token_positions
        ],
        "candidate_token_ids": {
            text: info.token_ids for text, info in candidate_infos.items()
        },
        "e1_layers": e1_layers,
        "e2_layers": e2_layers,
        "e3_layers": e3_layers,
    }


def aggregate_mechanistic_results(
    record_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not record_results:
        return {
            "num_records": 0,
            "e1": {"per_layer": [], "summary": {}},
            "e2": {"per_layer": [], "summary": {}},
            "e3": {"per_layer": [], "summary": {}},
        }

    layer_count = len(record_results[0]["e1_layers"])
    e1_per_layer: list[dict[str, Any]] = []
    e2_per_layer: list[dict[str, Any]] = []
    e3_per_layer: list[dict[str, Any]] = []
    e2_head_accumulator: dict[str, list[float]] = defaultdict(list)

    for layer_index in range(layer_count):
        neg_margins = [item["e1_layers"][layer_index]["neg_margin"] for item in record_results]
        pos_margins = [item["e1_layers"][layer_index]["pos_margin"] for item in record_results]
        neg_allowed_rate = [
            item["e1_layers"][layer_index]["neg_best_is_allowed"] for item in record_results
        ]
        pos_gold_rate = [
            item["e1_layers"][layer_index]["pos_best_is_gold"] for item in record_results
        ]
        e1_per_layer.append(
            {
                "layer": layer_index,
                "neg_margin_mean": mean(neg_margins),
                "neg_margin_median": median(neg_margins),
                "neg_best_allowed_rate": mean(float(value) for value in neg_allowed_rate),
                "pos_margin_mean": mean(pos_margins),
                "pos_best_gold_rate": mean(float(value) for value in pos_gold_rate),
            }
        )

        e2_values = [
            item["e2_layers"][layer_index]["mean_attention_to_negation"]
            for item in record_results
        ]
        e2_per_layer.append(
            {
                "layer": layer_index,
                "mean_attention_to_negation": mean(e2_values),
                "median_attention_to_negation": median(e2_values),
                "max_head_attention_mean": mean(
                    item["e2_layers"][layer_index]["max_head_attention_to_negation"]
                    for item in record_results
                ),
            }
        )
        for item in record_results:
            for head_item in item["e2_layers"][layer_index]["top_heads"]:
                key = f"L{layer_index}H{head_item['head']}"
                e2_head_accumulator[key].append(head_item["attention_to_negation"])

        e3_values = [item["e3_layers"][layer_index]["recovery"] for item in record_results]
        e3_success = [
            item["e3_layers"][layer_index]["patched_best_is_allowed"] for item in record_results
        ]
        e3_per_layer.append(
            {
                "layer": layer_index,
                "recovery_mean": mean(e3_values),
                "recovery_median": median(e3_values),
                "positive_recovery_rate": mean(float(value > 0.0) for value in e3_values),
                "patched_best_allowed_rate": mean(float(value) for value in e3_success),
            }
        )

    e1_curve = [item["neg_margin_mean"] for item in e1_per_layer]
    e2_curve = [item["mean_attention_to_negation"] for item in e2_per_layer]
    e3_curve = [item["recovery_mean"] for item in e3_per_layer]
    top_heads = sorted(
        (
            {"head": key, "mean_attention_to_negation": mean(values)}
            for key, values in e2_head_accumulator.items()
        ),
        key=lambda item: item["mean_attention_to_negation"],
        reverse=True,
    )[:10]

    return {
        "num_records": len(record_results),
        "e1": {
            "per_layer": e1_per_layer,
            "summary": summarize_layer_series(e1_curve),
        },
        "e2": {
            "per_layer": e2_per_layer,
            "summary": summarize_layer_series(e2_curve),
            "top_heads": top_heads,
        },
        "e3": {
            "per_layer": e3_per_layer,
            "summary": summarize_layer_series(e3_curve),
        },
    }
