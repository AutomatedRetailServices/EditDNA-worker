"""D-139 -- concrete Gemini-backed `MultimodalBestTakeArbiter` (model-comparison
challenger against the D-138 OpenAI baseline).

docs/CUTSELL_DECISIONS.md D-139, D-138, D-136 (D-127 Phase 2). Bounded,
OFFLINE-EVAL-ONLY concrete implementation of the `MultimodalBestTakeArbiter`
Protocol (`multimodal_besttake_arbiter.py`) -- mirrors the repo's EXISTING
Gemini REST transport pattern (`hybrid_google_transport.GoogleGeminiTransport`:
`POST https://generativelanguage.googleapis.com/v1beta/models/{model}:
generateContent` with an `x-goog-api-key` header, `requests` already a
dependency) rather than adding a new `google-generativeai` SDK dependency.
The Protocol interface itself (`arbitrate(request) -> response`) is
UNCHANGED -- no concrete incompatibility was found that would require
touching it, exactly as D-136's own OpenAI provider proved.

DECISION-DOCTRINE PARITY BY CONSTRUCTION: this module imports and reuses the
EXACT SAME `INSTRUCTION` constant from `multimodal_besttake_openai.py`
(never copies or paraphrases it) -- guaranteeing byte-for-byte editorial
doctrine parity between the OpenAI and Gemini providers. Provider-specific
request SYNTAX differs (Gemini's `contents[].parts[]` with `inline_data`
image blocks vs. OpenAI's `input`/`input_image` content blocks); editorial
SEMANTICS do not.

THIS CLASS IS NEVER IMPORTED BY THE LIVE PIPELINE. Same structural isolation
already proven for the OpenAI provider (`test_no_live_pipeline_import_or_
instantiation`-shaped test, extended here for this module).

AUDIO PERCEPTION: NOT AVAILABLE. Same as the OpenAI provider -- this module
sends TEXT (transcript, timing, CASE B aggregate evidence, semantic/
DeliveryScorer summaries) and sampled IMAGE frames only. No audio bytes are
ever constructed, read, encoded, or sent to the provider anywhere in this
module.

MODEL AVAILABILITY IS VERIFIED, NEVER ASSUMED: `list_gemini_models`/
`model_supports_generate_content` call Google's own read-only, zero-
generation-cost `ListModels` endpoint to confirm the requested model id
actually exists and supports `generateContent` before any real generation
call is attempted -- this task's own directive requires STOPPING rather
than silently substituting a different Gemini generation if the exact
requested model is unavailable.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests

from .multimodal_besttake_arbiter import (
    MultimodalBestTakeRequest,
    MultimodalBestTakeResponse,
)
from .multimodal_besttake_openai import INSTRUCTION
from .openai_json import parse_json_object

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

#: The exact Gemini generation this task's directive requires -- never
#: silently substituted for a different generation/tier.
REQUIRED_MODEL_ID = "gemini-2.5-flash"


def list_gemini_models(
    api_key: str,
    *,
    session: Any = requests,
    timeout_sec: float = 30.0,
) -> list[dict]:
    """Read-only, zero-generation-cost model catalog listing (Google's own
    `ListModels` endpoint) -- used to verify a specific model id is actually
    available and supports `generateContent` before committing to it. Never
    a generation call; never billed."""
    response = session.get(
        f"{GEMINI_API_BASE}/models",
        headers={"x-goog-api-key": api_key},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    raw = response.json()
    models = raw.get("models") if isinstance(raw, Mapping) else None
    return list(models) if isinstance(models, list) else []


def model_supports_generate_content(models: list[dict], model_id: str) -> bool:
    """`model_id` may be given bare (`"gemini-2.5-flash"`) or fully-qualified
    (`"models/gemini-2.5-flash"`) -- checked either way, EXACT match only on
    the bare id (never a prefix/substring match that could silently accept a
    different generation, e.g. `gemini-2.5-flash-lite` when `gemini-2.5-
    flash` was requested)."""
    target_bare = model_id.split("/")[-1]
    for entry in models:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if name.split("/")[-1] != target_bare:
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if "generateContent" in methods:
            return True
    return False


def _extract_text(raw: Mapping[str, Any]) -> str:
    """Extract the model's text response from a Gemini `generateContent`
    response body. Raises `ValueError` on any malformed shape -- caught by
    `_classify_provider_call_exception` as `INVALID_RESPONSE`, exactly like
    the OpenAI provider's own `parse_json_object` failure path."""
    candidates = raw.get("candidates") if isinstance(raw, Mapping) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response has no candidates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise ValueError("Gemini candidate is not an object")
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini candidate has no content parts")
    text = parts[0].get("text") if isinstance(parts[0], Mapping) else None
    if not isinstance(text, str) or not text:
        raise ValueError("Gemini candidate part has no text")
    return text


@dataclass
class GeminiMultimodalBestTakeArbiter:
    """Concrete, OFFLINE-EVAL-ONLY `MultimodalBestTakeArbiter` implementation
    for D-139's model comparison.

    `session` mirrors `hybrid_google_transport.GoogleGeminiTransport`'s own
    injection point for testing (a raw REST provider, so `session` -- not
    OpenAI-SDK's `client_factory` convention, which does not apply here).

    After a successful `.arbitrate(...)` call, `self.last_call_metadata`
    holds `{latency_ms, request_candidate_count, frame_count, usage}` for
    the eval harness's own cost/latency observability -- same shape as the
    OpenAI provider's own field, kept OUTSIDE the `MultimodalBestTakeResponse`
    return value so the Protocol/dataclass interface stays unchanged."""

    api_key: str
    model: str = REQUIRED_MODEL_ID
    session: Any = requests
    timeout_sec: float = 60.0
    last_call_metadata: dict | None = None

    @staticmethod
    def _image_part(path: str) -> dict:
        # D-139 real-evidence fix: the Generative Language API's JSON
        # representation is camelCase throughout (confirmed by this repo's
        # own already-working `hybrid_google.build_gemini_generate_content_
        # request`, which uses `generationConfig`/`maxOutputTokens`/
        # `responseMimeType`) -- `inlineData`/`mimeType`, never snake_case.
        # The first real dispatch of this provider (CI run 34210452645)
        # used snake_case and every applicable call failed fast (~100-260ms)
        # with a real HTTP 4xx, correctly classified PROVIDER_ERROR by
        # `safe_arbitrate` but with zero real judgments obtained.
        raw = Path(path).read_bytes()
        return {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64.b64encode(raw).decode("ascii"),
            }
        }

    def _parts_for(self, request: MultimodalBestTakeRequest) -> list[dict]:
        parts: list[dict] = [{"text": INSTRUCTION}]
        parts.append({
            "text": json.dumps({
                "family_id": request.family_id,
                "proposition_context": request.proposition_context,
                "candidate_ids": [f.candidate_id for f in request.finalists],
            }, ensure_ascii=False),
        })
        for finalist in request.finalists:
            parts.append({
                "text": json.dumps({
                    "candidate_id": finalist.candidate_id,
                    "transcript": finalist.transcript_text,
                    "source_asset_id": finalist.source_asset_id,
                    "source_start_sec": finalist.source_start,
                    "source_end_sec": finalist.source_end,
                    "meaning_sufficient": finalist.meaning_sufficient,
                    "semantic_label": finalist.semantic_label,
                    "semantic_confidence": finalist.semantic_confidence,
                    "deliveryscore_summary": finalist.deliveryscore_summary,
                    "boundary_editability_note": finalist.boundary_editability_note,
                    "case_b_delivery_event_count": finalist.case_b_evidence.delivery_event_count,
                    "case_b_delivery_event_duration_total": finalist.case_b_evidence.delivery_event_duration_total,
                    "case_b_count_by_kind": dict(finalist.case_b_evidence.count_by_kind),
                    "case_b_duration_by_kind": dict(finalist.case_b_evidence.duration_by_kind),
                    "case_b_event_density": finalist.case_b_evidence.event_density,
                    "sampled_frame_count": len(finalist.sampled_frame_references),
                }, ensure_ascii=False),
            })
            for frame_path in finalist.sampled_frame_references:
                parts.append(self._image_part(frame_path))
        return parts

    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse:
        parts = self._parts_for(request)
        body = {"contents": [{"role": "user", "parts": parts}]}
        started = time.monotonic()
        response = self.session.post(
            f"{GEMINI_API_BASE}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        latency_ms = round((time.monotonic() - started) * 1000, 1)

        raw = response.json()
        if not isinstance(raw, Mapping):
            raise ValueError("Gemini HTTP response must be an object")
        usage = raw.get("usageMetadata")
        self.last_call_metadata = {
            "latency_ms": latency_ms,
            "request_candidate_count": len(request.finalists),
            "frame_count": sum(len(f.sampled_frame_references) for f in request.finalists),
            "usage": dict(usage) if isinstance(usage, Mapping) else None,
        }

        text = _extract_text(raw)
        data = parse_json_object(text)
        outcome = str(data.get("outcome") or "")
        raw_candidate_id = data.get("best_take_candidate_id")
        confidence_value = data.get("confidence")
        reason = str(data.get("reason") or "")

        return MultimodalBestTakeResponse(
            family_id=request.family_id,
            outcome=outcome,
            best_take_candidate_id=(str(raw_candidate_id) if raw_candidate_id else None),
            confidence=float(confidence_value) if confidence_value is not None else 0.0,
            reason=reason,
            provider="gemini",
            model=self.model,
            requested=True,
            available=True,
        )
