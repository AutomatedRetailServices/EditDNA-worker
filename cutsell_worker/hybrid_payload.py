"""Compact, provider-neutral payload and hard budget guards for Hybrid Flow B.

No SDK imports, network calls, or secret access live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hybrid_editorial import EditorialSession, HybridGatePolicy


@dataclass(frozen=True)
class HybridCostPolicy:
    max_calls_per_session: int = 1
    max_candidates_per_call: int = 14
    max_chars_per_candidate: int = 1_200
    max_payload_chars: int = 12_000
    max_estimated_input_tokens: int = 4_000
    max_estimated_output_tokens: int = 500


def estimate_tokens_from_chars(char_count: int) -> int:
    return max(1, (max(0, int(char_count)) + 2) // 3)


def build_compact_editorial_payload(
    session: EditorialSession,
    *,
    cost_policy: HybridCostPolicy = HybridCostPolicy(),
) -> dict[str, Any]:
    if not session.candidates:
        raise ValueError("hybrid payload requires candidates")
    if len(session.candidates) > cost_policy.max_candidates_per_call:
        raise ValueError("hybrid payload candidate budget exceeded")

    candidates = []
    for candidate in session.candidates:
        text = str(candidate.text or "").strip()
        if len(text) > cost_policy.max_chars_per_candidate:
            text = text[: cost_policy.max_chars_per_candidate]
        evidence = {str(key): value for key, value in candidate.evidence}
        candidates.append({
            "clip_id": candidate.clip_id,
            "text": text,
            "duration_sec": round(candidate.duration_sec, 3),
            "local_label": candidate.local_label,
            "local_confidence": round(float(candidate.local_confidence), 4),
            "evidence": evidence,
        })

    cleanup_task = session.task == "classify_recording_process_within_single_creator_session"
    task_rules = [
        "valid independent audience-facing speech should be keep",
        "failed means a clear stumble, false start, incomplete attempt, or word-search",
        "bts means self-talk, recording-process commentary, frustration, self-review, or breaking character",
        "do not force a winner across different ideas in the same creator session",
    ] if cleanup_task else [
        "return exactly one winner only when the supplied candidates are competing retries and evidence supports one",
        "independent valid speech may be keep instead of being forced into winner/alternate",
    ]

    payload = {
        "task": session.task,
        "session_id": session.session_id,
        "source_asset_id": session.source_asset_id,
        "local_confidence": round(float(session.local_confidence), 4),
        "conflict_score": round(float(session.conflict_score), 4),
        "allowed_labels": ["winner", "alternate", "failed", "bts", "uncertain", "keep"],
        "rules": [
            "reference every supplied clip_id exactly once",
            "never invent clip ids",
            "never create or alter timestamps",
            "reason only inside this bounded creator session",
            "use uncertain when semantic evidence is insufficient",
            *task_rules,
        ],
        "candidates": candidates,
    }

    payload_chars = len(repr(payload))
    estimated_tokens = estimate_tokens_from_chars(payload_chars)
    if payload_chars > cost_policy.max_payload_chars:
        raise ValueError("hybrid payload character budget exceeded")
    if estimated_tokens > cost_policy.max_estimated_input_tokens:
        raise ValueError("hybrid payload estimated token budget exceeded")
    return payload


def preflight_hybrid_call(
    session: EditorialSession,
    gate_policy: HybridGatePolicy,
    *,
    cost_policy: HybridCostPolicy = HybridCostPolicy(),
) -> dict[str, int | bool]:
    payload = build_compact_editorial_payload(session, cost_policy=cost_policy)
    payload_chars = len(repr(payload))
    estimated_input_tokens = estimate_tokens_from_chars(payload_chars)
    allowed = (
        len(session.candidates) <= gate_policy.max_candidates_per_request
        and estimated_input_tokens <= min(
            gate_policy.max_estimated_input_tokens,
            cost_policy.max_estimated_input_tokens,
        )
        and cost_policy.max_calls_per_session >= 1
    )
    return {
        "allowed": allowed,
        "payload_chars": payload_chars,
        "estimated_input_tokens": estimated_input_tokens,
        "max_output_tokens": min(
            gate_policy.max_estimated_output_tokens,
            cost_policy.max_estimated_output_tokens,
        ),
    }
