from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from neg_blindness.io_utils import read_jsonl


@dataclass
class ModelConfig:
    name: str
    mode: str
    endpoint: str = ""
    model: str = ""
    api_key: str = "EMPTY"
    api_key_env: str = ""
    timeout_sec: int = 120
    temperature: float = 0.0
    max_tokens: int = 0
    model_path: str = ""
    adapter_path: str = ""
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(data["name"]),
            endpoint=str(data.get("endpoint", "")),
            model=str(data.get("model", "")),
            mode=str(data["mode"]),
            api_key=str(data.get("api_key", "EMPTY")),
            api_key_env=str(data.get("api_key_env", "")),
            timeout_sec=int(data.get("timeout_sec", 120)),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 0)),
            model_path=str(data.get("model_path", "")),
            adapter_path=str(data.get("adapter_path", "")),
            device_map=str(data.get("device_map", "auto")),
            torch_dtype=str(data.get("torch_dtype", "auto")),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        if self.api_key == "EMPTY":
            return ""
        return self.api_key


class ScoreCache:
    def __init__(self, cache_dir: str | Path | None, model_name: str) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name
        self.memory: dict[str, Any] = {}
        self.cache_path = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = self.cache_dir / f"{model_name}.jsonl"
            if self.cache_path.exists():
                for item in read_jsonl(self.cache_path):
                    self.memory[item["key"]] = item["value"]

    def get(self, key: str) -> Any | None:
        return self.memory.get(key)

    def set(self, key: str, value: Any) -> None:
        self.memory[key] = value
        if not self.cache_path:
            return
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
            )


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        item["name"]: ModelConfig.from_dict(item)
        for item in payload.get("models", [])
    }


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_sec: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(endpoint, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request to {endpoint} failed: {exc}") from exc


def _extract_json_text(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty response text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start : end + 1])
        raise


def join_prompt_and_continuation(prompt: str, continuation: str) -> str:
    if not continuation:
        return prompt
    if not prompt:
        return continuation
    if prompt[-1].isspace() or continuation[0].isspace():
        return prompt + continuation
    if continuation[0] in ",.;:!?)]}":
        return prompt + continuation
    return prompt + " " + continuation


_LOCAL_MODEL_RUNNERS: dict[str, "LocalCausalLMRunner"] = {}


class LocalCausalLMRunner:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local model evaluation requires torch and transformers. "
                "Install them in the Python environment used to run the scripts."
            ) from exc

        self.torch = torch
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

    def _resolve_dtype(self, dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        if not hasattr(self.torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return getattr(self.torch, dtype_name)

    def _model_input_device(self) -> Any:
        try:
            return self.model.device
        except Exception:  # noqa: BLE001
            return next(self.model.parameters()).device

    def score_continuation(self, prompt: str, continuation: str) -> float:
        if not continuation:
            raise ValueError("continuation must be non-empty")
        full_text = join_prompt_and_continuation(prompt, continuation)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        full_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full_len = int(full_inputs["input_ids"].shape[1])
        if full_len <= prompt_len:
            raise ValueError("continuation tokenized to empty sequence")

        device = self._model_input_device()
        input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)

        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, prompt_len - 1 : -1, :]
            log_probs = logits.log_softmax(dim=-1)
            target_ids = input_ids[:, prompt_len:]
            token_logprobs = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
        return float(token_logprobs.mean().item())


def get_local_runner(config: ModelConfig) -> LocalCausalLMRunner:
    cache_key = config.name
    runner = _LOCAL_MODEL_RUNNERS.get(cache_key)
    if runner is None:
        runner = LocalCausalLMRunner(config)
        _LOCAL_MODEL_RUNNERS[cache_key] = runner
    return runner


def chat_json_request(
    config: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    cache: ScoreCache | None = None,
    max_retries: int = 3,
) -> Any:
    cache_key = sha1(
        json.dumps(
            {
                "endpoint": config.endpoint,
                "model": config.model,
                "system": system_prompt,
                "user": user_prompt,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = _post_json(
                endpoint=config.endpoint,
                payload=payload,
                api_key=config.resolved_api_key(),
                timeout_sec=config.timeout_sec,
            )
            content = response["choices"][0]["message"]["content"]
            parsed = _extract_json_text(content)
            if cache:
                cache.set(cache_key, parsed)
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"chat_json_request failed after retries: {last_error}")


def score_continuation(
    config: ModelConfig,
    prompt: str,
    continuation: str,
    cache: ScoreCache | None = None,
    max_retries: int = 3,
) -> float:
    cache_key = sha1(
        json.dumps(
            {
                "endpoint": config.endpoint,
                "model": config.model,
                "prompt": prompt,
                "continuation": continuation,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return float(cached)

    if config.mode == "hf_local":
        score = get_local_runner(config).score_continuation(prompt, continuation)
        if cache:
            cache.set(cache_key, score)
        return float(score)

    full_prompt = join_prompt_and_continuation(prompt, continuation)
    payload = {
        "model": config.model,
        "prompt": full_prompt,
        "max_tokens": 0,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = _post_json(
                endpoint=config.endpoint,
                payload=payload,
                api_key=config.resolved_api_key(),
                timeout_sec=config.timeout_sec,
            )
            choice = response["choices"][0]
            logprobs = choice["logprobs"]
            offsets = logprobs["text_offset"]
            token_logprobs = logprobs["token_logprobs"]
            prompt_len = len(prompt)

            continuation_scores = [
                lp
                for offset, lp in zip(offsets, token_logprobs)
                if offset >= prompt_len and lp is not None
            ]
            if not continuation_scores:
                raise ValueError("no continuation token logprobs returned")
            average = sum(continuation_scores) / len(continuation_scores)
            if cache:
                cache.set(cache_key, average)
            return float(average)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"score_continuation failed after retries: {last_error}")
