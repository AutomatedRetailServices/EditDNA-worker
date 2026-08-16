"""Pure request/response helpers for a future Google Gemini Hybrid transport.

This module deliberately performs no HTTP, reads no API key, and cannot spend money.
It only builds the structured-output request body and parses the response body that a
future explicitly-enabled transport will send/receive.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


_ALLOWED_LABELS = ["winner", "alternate", "failed", "bts", "uncertain", "keep"]


def editorial_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clip_id": {"type": "string"},
                        "label": {"type": "string", "enum": _ALLOWED_LABELS},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason_code": {"type": "string"},
                    },
                    "required": ["clip_id", "label", "confidence", "reason_code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _prompt_text(compact_payload: Mapping[str, Any]) -> str:
    return (
        "You are the CutSell editorial judge. Classify only the supplied candidates "
        "inside this already-bounded creator mini-session. Do not invent clips, do "
        "not create timestamps, and do not compare against any other creator/session. "
        "Use the local evidence as supporting signals, but apply semantic judgment to "
        "identify recording failures, BTS/self-talk, alternates, and the strongest "
        "complete take. Return one decision for every candidate. Return exactly one "
        "winner only when justified; otherwise use uncertain/keep as appropriate.\n\n"
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
    return {
        "contents": [{"role": "user", "parts": [{"text": _prompt_text(compact_payload)}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_output_tokens),
            "thinkingConfig": {"thinkingLevel": thinking_level},
            "responseFormat": {
                "text": {
                    "mimeType": "application/json",
                    "schema": editorial_response_schema(),
                }
            },
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
    text = parts[0].get("text") if isinstance(parts[0], Mapping) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini structured response missing text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini structured response is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Gemini structured response must be an object")

    usage = response.get("usageMetadata") or {}
    output_tokens = 0
    if isinstance(usage, Mapping):
        output_tokens = max(0, int(usage.get("candidatesTokenCount") or 0))
    return {**parsed, "output_tokens": output_tokens}
