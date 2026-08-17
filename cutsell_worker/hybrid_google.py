"""Pure request/response helpers for Google Gemini Hybrid transport.

No HTTP, secret reads, or spending occur here. The request shape mirrors the live
bake-off that succeeded with Gemini 3.5 Flash-Lite.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


_ALLOWED_LABELS = ["winner", "alternate", "failed", "bts", "uncertain", "keep"]


def editorial_response_schema(candidate_count: int | None = None) -> dict[str, Any]:
    """Return the smallest safe vendor schema for ordered editorial decisions.

    Gemini does not need to echo clip IDs: the caller already knows the deterministic
    candidate order. Omitting IDs materially reduces structured-output size and avoids
    truncation while downstream code can reattach IDs by index before validation.
    """
    decisions_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": _ALLOWED_LABELS},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        },
    }
    if candidate_count is not None and int(candidate_count) >= 0:
        decisions_schema["minItems"] = int(candidate_count)
        decisions_schema["maxItems"] = int(candidate_count)
    return {
        "type": "object",
        "properties": {"decisions": decisions_schema},
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _prompt_text(compact_payload: Mapping[str, Any]) -> str:
    return (
        "You are the CutSell editorial judge. Classify only the supplied candidates "
        "inside this already-bounded creator group/session. Do not invent clips, do "
        "not create timestamps, and do not compare against any other creator/session. "
        "A winner is the strongest complete intended delivery. Alternate is usable but "
        "not the best. Failed is a stumble, false start, word-search, incomplete or "
        "broken delivery. BTS is creator self-talk, recording-process commentary, "
        "frustration, self-review or breaking character. Return exactly one compact "
        "decision per candidate, in the exact same order as the candidates array, using "
        "only label and confidence. Do not echo clip IDs. Exactly one winner only when "
        "justified; use uncertain when evidence is insufficient.\n\n"
        + json.dumps(dict(compact_payload), separators=(",", ":"), ensure_ascii=False)
    )


def build_gemini_generate_content_request(
    compact_payload: Mapping[str, Any],
    *,
    max_output_tokens: int,
    thinking_level: str = "minimal",
) -> dict[str, Any]:
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise ValueError("unsupported Gemini thinking level")
    raw_candidates = compact_payload.get("candidates")
    candidate_count = len(raw_candidates) if isinstance(raw_candidates, (list, tuple)) else None
    return {
        "contents": [{"role": "user", "parts": [{"text": _prompt_text(compact_payload)}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_output_tokens),
            "thinkingConfig": {"thinkingLevel": thinking_level},
            "responseMimeType": "application/json",
            "responseJsonSchema": editorial_response_schema(candidate_count),
        },
    }


def parse_gemini_generate_content_response(response: Mapping[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise ValueError("Gemini candidate malformed")
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini response missing content")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response missing parts")
    text = "".join(
        str(part.get("text") or "")
        for part in parts
        if isinstance(part, Mapping)
    )
    if not text.strip():
        raise ValueError("Gemini structured response missing text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        finish_reason = str(first.get("finishReason") or "unknown")
        usage = response.get("usageMetadata") or {}
        output_tokens = int(usage.get("candidatesTokenCount") or 0) if isinstance(usage, Mapping) else 0
        raise ValueError(
            f"Gemini structured response is invalid JSON; finish_reason={finish_reason}; output_tokens={output_tokens}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Gemini structured response must be an object")

    usage = response.get("usageMetadata") or {}
    output_tokens = 0
    if isinstance(usage, Mapping):
        output_tokens = max(0, int(usage.get("candidatesTokenCount") or 0))
    return {**parsed, "output_tokens": output_tokens}
