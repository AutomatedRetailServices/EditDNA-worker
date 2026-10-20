"""Anthropic call for the simple engine. Only env var NAMES live here; the key stays in the host's secrets."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import time

import requests

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
STEPS = ("pass", "decision", "haul_review", "sale_review", "fixwords")
_STEP: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_step", default="")
_LOG = logging.getLogger("cutsell.simple_engine.llm")


@contextlib.contextmanager
def step(name: str):
    """Marks which engine step the calls inside belong to (only used to pick that step's thinking effort)."""
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


# Measured on the Gold (2026-10-17, 26 videos x 2 runs): less thinking in the pass step and the haul review keeps the
# cut quality; the decision, the sale review and the word fix stay at the model default (high). A lower effort on the
# word fix wrote a wrong brand name into a caption, and a lower effort on the decision lost sales lines.
DEFAULT_EFFORT = "pass=medium,haul_review=low"


def effort_for(name: str = "") -> str:
    """Thinking effort for a step from CUTSELL_SIMPLE_ENGINE_EFFORT (else DEFAULT_EFFORT): "medium" (every step) or
    "pass=medium,fixwords=low" (only those steps). "high" = every step at the model default, exactly as in 1.2.1
    (the rollback without a redeploy). A step not listed / an unknown value -> "" = the model default (high)."""
    raw = (os.environ.get("CUTSELL_SIMPLE_ENGINE_EFFORT", "").strip() or DEFAULT_EFFORT).lower()
    if raw == "high":
        return ""
    if "=" not in raw:
        return raw if raw in EFFORTS else ""
    for part in raw.split(","):
        key, _, value = part.partition("=")
        if key.strip() == (name or _STEP.get()) and value.strip() in EFFORTS:
            return value.strip()
    return ""


class Prompt(str):
    """A prompt that is still a plain string (tests and replays see the same text) but remembers its parts: the fixed
    instructions of a step, then the video's own text, then the run tag. call_anthropic sends the parts as separate
    content blocks and marks the fixed instructions for prompt caching (identical for every video, so the next video
    within 5 minutes reads them at the cache price). The text the model reads is exactly the same."""

    def __new__(cls, *parts: str, schema: dict | None = None):
        obj = super().__new__(cls, "".join(parts))
        obj.parts = tuple(p for p in parts if p)
        obj.schema = schema
        return obj


def _content(prompt):
    parts = getattr(prompt, "parts", None)
    if not parts or not cache_enabled():
        return str(prompt)
    blocks = [{"type": "text", "text": p} for p in parts]
    blocks[0]["cache_control"] = {"type": "ephemeral", **({"ttl": "1h"} if cache_ttl() == "1h" else {})}
    return blocks


def cache_ttl() -> str:
    """CUTSELL_SIMPLE_ENGINE_CACHE_TTL=1h writes the cache for 1 hour (2x write price) instead of 5 minutes (1.25x)."""
    return "1h" if os.environ.get("CUTSELL_SIMPLE_ENGINE_CACHE_TTL", "").strip().lower() == "1h" else "5m"


def json_enabled() -> bool:
    """CUTSELL_SIMPLE_ENGINE_JSON=0 lets the model answer in free text again (rollback without a redeploy)."""
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_JSON", "1").strip() != "0"


_CACHE_ON: contextvars.ContextVar = contextvars.ContextVar("cutsell_simple_engine_cache", default=False)


@contextlib.contextmanager
def cache_scope(on: bool):
    """Prompt caching only inside this block. Product Owner decision (2026-10-19): on only for a video that is part of
    a Bulk send (the next videos of the same send read the cached instructions); a lone video never pays the cache-write
    premium."""
    token = _CACHE_ON.set(bool(on))
    try:
        yield
    finally:
        _CACHE_ON.reset(token)


def cache_enabled() -> bool:
    """On inside cache_scope(True) only. CUTSELL_SIMPLE_ENGINE_CACHE=0 turns it off everywhere (no redeploy)."""
    if os.environ.get("CUTSELL_SIMPLE_ENGINE_CACHE", "1").strip() == "0":
        return False
    # CUTSELL_SIMPLE_ENGINE_CACHE_SCOPE=all caches every video, not only a Bulk send (measurement / high volume).
    return _CACHE_ON.get() or os.environ.get("CUTSELL_SIMPLE_ENGINE_CACHE_SCOPE", "").strip().lower() == "all"


def model_name() -> str:
    return os.environ.get("CUTSELL_SIMPLE_ENGINE_MODEL", "").strip() or DEFAULT_MODEL


KIE_URL = "https://api.kie.ai/claude/v1/messages"
KIE_TIMEOUT_SEC = 60
FALLBACK_CHECK_STEPS = ("decision", "sale_review", "fixwords")   # steps that think at the model default (high)


def provider() -> str:
    """CUTSELL_LLM_PROVIDER=kie sends each call to kie.ai first (same model, same request), with Anthropic direct as
    the automatic fallback. Default and rollback: anthropic."""
    return "kie" if os.environ.get("CUTSELL_LLM_PROVIDER", "").strip().lower() == "kie" else "anthropic"


def call_anthropic(prompt, *, max_tokens: int = 16000, attempts: int = 4) -> tuple[str, dict]:
    """One user message in, (text, usage) out, through the configured provider. The request (model, prompt, thinking,
    effort, JSON format, cache marks) is the same for both providers; kie.ai answers that are late, broken, or that
    dropped the thinking or the cache are asked again from Anthropic direct and logged as provider_fallback."""
    body = _body(prompt, max_tokens)
    if provider() == "kie":
        started = time.time()
        reason, text, usage = _try_kie(body, prompt)
        usage_kie = usage
        if reason is None:
            usage.update({"provider": "kie", "latency_s": round(time.time() - started, 2)})
            return text, usage
        _LOG.warning("provider_fallback reason=%s step=%s", reason, _STEP.get() or "other")
        text, usage = _call_direct(_body(prompt, max_tokens), prompt, max_tokens=max_tokens, attempts=attempts)
        usage.update({"provider": "anthropic", "provider_fallback": reason,
                      "kie_dropped": {k: usage_kie.get(k) for k in ("input_tokens", "output_tokens",
                                                                     "cache_creation_input_tokens",
                                                                     "cache_read_input_tokens", "credits_consumed")}})
        return text, usage
    return _call_direct(body, prompt, max_tokens=max_tokens, attempts=attempts)


def _body(prompt, max_tokens: int) -> dict:
    body = {"model": model_name(), "max_tokens": max_tokens, "messages": [{"role": "user", "content": _content(prompt)}]}
    effort = effort_for()
    if effort:
        body["output_config"] = {"effort": effort}
    schema = getattr(prompt, "schema", None)
    if schema and json_enabled():
        # Structured output: the visible answer is only the JSON (no notes before it). Thinking is not affected.
        body.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": schema}
    return body


def _usage_of(data: dict, body: dict, text: str, max_tokens: int) -> dict:
    effort = (body.get("output_config") or {}).get("effort")
    usage = dict(data.get("usage") or {})
    usage.update({
        "step": _STEP.get() or "other", "max_tokens": max_tokens,
        "thinking": "adaptive", "effort": effort or "default(high)",
        "thinking_blocks": sum(1 for b in data.get("content", []) if b.get("type") == "thinking"),
        "answer_chars": len(text), "stop_reason": data.get("stop_reason"),
        "json_format": "format" in body.get("output_config", {}),
        "cache": isinstance(body["messages"][0]["content"], list),
    })
    if "credits_consumed" in data:
        usage["credits_consumed"] = data.get("credits_consumed")
    return usage


def _try_kie(body: dict, prompt) -> tuple:
    """-> (fallback reason or None, text, usage). Never raises."""
    key = os.environ.get("KIE_API_KEY", "").strip()
    if not key:
        return "no_kie_key", "", {}
    headers = {"Authorization": "Bearer " + key, "content-type": "application/json", "anthropic-version": "2023-06-01"}
    try:
        response = requests.post(KIE_URL, json=body, headers=headers, timeout=(10, KIE_TIMEOUT_SEC))
    except requests.Timeout:
        return "timeout", "", {}
    except requests.RequestException as exc:
        return "network_" + exc.__class__.__name__.lower(), "", {}
    if response.status_code != 200:
        return f"http_{response.status_code}", "", {}
    try:
        data = response.json()
    except ValueError:
        return "not_json_envelope", "", {}
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        return "bad_envelope_" + str((data or {}).get("code") if isinstance(data, dict) else "x")[:20], "", {}
    text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
    usage = _usage_of(data, body, text, int(body["max_tokens"]))
    reason = _kie_answer_problem(body, text, usage)
    if reason:
        return reason, "", usage                       # usage kept: what kie.ai charged for the answer we dropped
    _LOG.info("simple_engine_llm_call %s", json.dumps({k: usage.get(k) for k in (
        "step", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        "max_tokens", "effort", "answer_chars", "stop_reason", "cache", "json_format", "credits_consumed")} | {"provider": "kie"}))
    return None, text, usage


def _kie_answer_problem(body: dict, text: str, usage: dict):
    """Why a kie.ai answer cannot be trusted as the same call Anthropic would have answered (None = fine)."""
    if usage.get("stop_reason") == "max_tokens":
        return "truncated"
    if "format" in (body.get("output_config") or {}):
        try:
            json.loads(text)
        except ValueError:
            return "json_broken"
    elif "{" not in text and "[" not in text:
        return "json_broken"
    if (usage.get("step") in FALLBACK_CHECK_STEPS and usage.get("effort") == "default(high)"
            and not usage.get("thinking_blocks") and int(usage.get("output_tokens") or 0) <= len(text) / 2.5 + 30):
        return "thinking_ignored"
    content = body["messages"][0]["content"]
    if isinstance(content, list):
        fixed_chars = len(content[0].get("text", ""))
        cached = int(usage.get("cache_creation_input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
        if fixed_chars >= 2400 and cached == 0:          # >= ~600 tokens: above Sonnet 5.5's 512-token minimum
            return "cache_ignored"
    return None


def _call_direct(body: dict, prompt, *, max_tokens: int = 16000, attempts: int = 4) -> tuple[str, dict]:
    """Anthropic direct. Retries transient failures; never logs the prompt or the key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last: Exception | None = None
    json_refused = False
    for attempt in range(attempts):
        try:
            started = time.time()
            response = requests.post(API_URL, json=body, headers=headers, timeout=(20, 600))
            if response.status_code == 200:
                data = response.json()
                text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
                usage = _usage_of(data, body, text, max_tokens)
                usage.update({"json_refused": json_refused, "provider": "anthropic",
                              "latency_s": round(time.time() - started, 2)})
                _LOG.info("simple_engine_llm_call %s", json.dumps({k: usage.get(k) for k in (
                    "step", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                    "max_tokens", "thinking", "effort", "answer_chars", "stop_reason", "cache", "json_format",
                    "provider")}))
                return text, usage
            if response.status_code == 400 and "format" in body.get("output_config", {}):
                # The JSON format was refused: ask again in free text (the parsers read both) and say so in the logs.
                _LOG.warning("simple_engine_llm_json_refused step=%s; asking again without the JSON format", _STEP.get())
                body["output_config"].pop("format")
                if not body["output_config"]:
                    body.pop("output_config")
                json_refused = True
                continue
            if response.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"Anthropic HTTP {response.status_code}")
            last = RuntimeError(f"Anthropic HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Anthropic call failed: {last}")
