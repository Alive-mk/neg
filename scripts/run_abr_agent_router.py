"""Run an Agentic Behavior Router over prepared ABR JSONL inputs.

The router only predicts SUPPRESS/PRESERVE/SELECT. It never receives
gold_behavior and never chooses a final answer candidate.
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


ROUTES = {"SUPPRESS", "PRESERVE", "SELECT"}

ABR_TEMPLATE = """You are a negation behavior router.

Your task is NOT to answer the user question.
Your task is NOT to rank the candidate answers.
Your task is only to choose which behavior the downstream model should use.

Choose exactly one label:

SUPPRESS:
Use this when the prompt asks for something that does NOT belong to a category,
does NOT have a property, is NOT used for a function, or should avoid a target concept.

PRESERVE:
Use this when the prompt contains explicit negation, but the correct answer should preserve
the target concept because the negation scopes over an auxiliary action, condition,
prohibition, or instruction, such as "do not skip X" or "do not ignore X".

SELECT:
Use this when the prompt asks which option, step, answer, or statement is invalid,
false, incorrect, or should not be used.

Return JSON only:
{{
  "route": "SUPPRESS" | "PRESERVE" | "SELECT",
  "confidence": 0.0-1.0,
  "rationale": "one short sentence"
}}

Prompt:
{prompt}

Candidates:
{candidate_pool}
"""


DOUBLE_NEGATION_RE = re.compile(
    r"\b("
    r"not\s+not|"
    r"not\s+(?:be\s+)?without|"
    r"cannot\s+not|can\s+not\s+not|"
    r"not\s+fail|never\s+fail|"
    r"not\s+(?:neglect|ignore|forget|skip|overlook|omit)|"
    r"not\s+(?:incorrect|untrue|uncommon)|"
    r"not\s+the\s+case\s+that.*never|"
    r"isn'?t\s+it\s+true.*cannot\s+never"
    r")\b",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_candidates(record: dict[str, Any], mode: str) -> str:
    if mode == "prompt_only":
        return "Not provided."
    return json.dumps(record.get("candidate_pool", []), ensure_ascii=False)


def build_router_prompt(record: dict[str, Any], mode: str) -> str:
    return ABR_TEMPLATE.format(
        prompt=str(record.get("prompt", "")),
        candidate_pool=format_candidates(record, mode),
    )


def normalize_route(value: Any) -> str | None:
    route = str(value or "").strip().upper()
    route = route.strip("[]`\"' ")
    return route if route in ROUTES else None


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fence:
        stripped = fence.group(1)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found")


def parse_router_response(text: str) -> tuple[str, float | None, str, bool]:
    try:
        parsed = extract_json_object(text)
        route = normalize_route(parsed.get("route"))
        if route is None:
            raise ValueError("missing or invalid route")
        raw_conf = parsed.get("confidence")
        confidence = None
        if raw_conf is not None:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        rationale = str(parsed.get("rationale", "")).strip()
        return route, confidence, rationale, True
    except Exception:  # noqa: BLE001
        upper = text.upper()
        hits = [route for route in ROUTES if route in upper]
        if len(hits) == 1:
            return hits[0], None, "Recovered a single route label from non-JSON output.", True
        return "PARSE_ERROR", None, "", False


def dry_run_rule_route(record: dict[str, Any], mode: str) -> dict[str, Any]:
    text = str(record.get("prompt", "")).lower()
    candidates = " ".join(str(x) for x in record.get("candidate_pool", [])).lower()
    if DOUBLE_NEGATION_RE.search(text) or " without " in f" {text} ":
        route = "PRESERVE"
        rationale = "The negation appears out of scope or double-negative."
        confidence = 0.72
    elif re.search(r"\b(which|what|select|identify|choose)\b.*\bnot\b", text):
        route = "SELECT"
        rationale = "The prompt asks for an option that is not valid or not applicable."
        confidence = 0.70
    elif re.search(r"\b(incorrect|invalid|inappropriate|unsuitable|wrong|false)\b", text):
        route = "SELECT"
        rationale = "The prompt asks for an invalid or false option."
        confidence = 0.72
    elif re.search(
        r"\bnot\s+(?:the\s+)?"
        r"(?:correct|valid|good|effective|appropriate|suitable|best|largest|"
        r"smallest|fastest|slowest|oldest|newest|official|famous|known|used|"
        r"essential|necessary|required|needed)\b",
        text,
    ):
        route = "SELECT"
        rationale = "The negated superlative or validity cue points to option selection."
        confidence = 0.70
    elif re.search(r"\b(is|does|do|should|can|could|would|are)\b.*\bnot\b", text):
        comparative = re.search(
            r"\b(more|less|larger|smaller|higher|lower|better|worse|than|opposite|reverse)\b",
            text,
        )
        comparative_pool = re.search(r"\b(more|less|larger|smaller|higher|lower|than)\b", candidates)
        if mode == "prompt_candidates" and comparative and comparative_pool:
            route = "SELECT"
            rationale = "Candidate context suggests a reversed option should be selected."
            confidence = 0.68
        else:
            route = "SUPPRESS"
            rationale = "The prompt denies a target property or relation."
            confidence = 0.68
    else:
        route = "SUPPRESS"
        rationale = "The prompt asks to avoid the target concept."
        confidence = 0.60
    return {"route": route, "confidence": confidence, "rationale": rationale}


def api_key_from_args(args: argparse.Namespace) -> str:
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if key:
            return key
    if args.api_key_file:
        path = Path(args.api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def call_openai_compatible(prompt: str, args: argparse.Namespace) -> str:
    api_key = api_key_from_args(args)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(args.endpoint, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    with urlopen(request, timeout=args.timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def cache_key(record: dict[str, Any], mode: str, args: argparse.Namespace) -> str:
    payload = {
        "id": record.get("id"),
        "prompt": record.get("prompt"),
        "candidate_pool": record.get("candidate_pool") if mode == "prompt_candidates" else None,
        "mode": mode,
        "endpoint": "" if args.dry_run_rule_router else args.endpoint,
        "model": "dry-run-rule-router" if args.dry_run_rule_router else args.model,
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    out: dict[str, str] = {}
    for row in read_jsonl(path):
        out[str(row["key"])] = str(row["raw_response"])
    return out


def append_cache(path: Path | None, key: str, raw_response: str) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "raw_response": raw_response}, ensure_ascii=False) + "\n")


def route_record(record: dict[str, Any], args: argparse.Namespace, cache: dict[str, str]) -> dict[str, Any]:
    prompt = build_router_prompt(record, args.mode)
    key = cache_key(record, args.mode, args)
    raw_response = cache.get(key)
    if raw_response is None:
        if args.dry_run_rule_router:
            raw_response = json.dumps(dry_run_rule_route(record, args.mode), ensure_ascii=False)
        else:
            raw_response = call_openai_compatible(prompt, args)
        cache[key] = raw_response
        append_cache(Path(args.cache) if args.cache else None, key, raw_response)

    pred_behavior, confidence, rationale, parsed = parse_router_response(raw_response)
    out = {
        "id": record.get("id"),
        "prompt": record.get("prompt", ""),
        "candidate_pool": record.get("candidate_pool", []),
        "gold_behavior": record.get("gold_behavior", ""),
        "pred_behavior": pred_behavior,
        "confidence": confidence,
        "rationale": rationale,
        "raw_response": raw_response,
        "mode": args.mode,
    }
    if not parsed:
        out["parse_error"] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["prompt_only", "prompt_candidates"])
    parser.add_argument("--dry-run-rule-router", action="store_true")
    parser.add_argument("--endpoint", default="https://api.ai-gaochao.cn/v1/chat/completions")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-key-env", default="NEG_OPENAI_API_KEY")
    parser.add_argument("--api-key-file", default="/tmp/.neg_api_key")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--parse-errors", default="outputs/abr_router_preexp/parse_errors.jsonl")
    args = parser.parse_args()

    if not args.dry_run_rule_router and not api_key_from_args(args):
        raise SystemExit("No API key found. Set --api-key-env/--api-key-file or use --dry-run-rule-router.")

    records = read_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    cache = load_cache(Path(args.cache) if args.cache else None)

    outputs: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                out = route_record(record, args, cache)
                outputs.append(out)
                if out.get("parse_error"):
                    parse_errors.append(out)
                break
            except (HTTPError, URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raw = f"ROUTER_ERROR: {exc}"
                out = {
                    "id": record.get("id"),
                    "prompt": record.get("prompt", ""),
                    "candidate_pool": record.get("candidate_pool", []),
                    "gold_behavior": record.get("gold_behavior", ""),
                    "pred_behavior": "PARSE_ERROR",
                    "confidence": None,
                    "rationale": "",
                    "raw_response": raw,
                    "mode": args.mode,
                    "parse_error": True,
                }
                outputs.append(out)
                parse_errors.append(out)
        if args.sleep:
            time.sleep(args.sleep)
        if idx % 50 == 0:
            print(f"Routed {idx}/{len(records)}", flush=True)
        if last_error and outputs[-1].get("pred_behavior") == "PARSE_ERROR":
            print(f"Router failed for {record.get('id')}: {last_error}", file=sys.stderr)

    write_jsonl(Path(args.output), outputs)
    if parse_errors:
        write_jsonl(Path(args.parse_errors), parse_errors)
    print(f"Saved {len(outputs)} routes to {args.output}")
    print(f"Parse/router errors: {len(parse_errors)}")


if __name__ == "__main__":
    main()
