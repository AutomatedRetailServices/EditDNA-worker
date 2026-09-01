"""Compact, provider-neutral payload and hard budget guards for Hybrid Flow B.

No SDK imports, network calls, or secret access live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


# Whole-source context is valuable for retry/story reasoning, but Benchmark #41 proved
# that allowing every upstream field to consume its full local limit can make an otherwise
# valid 10-candidate Hybrid window fail *before* the provider is called. These limits keep
# the semantic map broad while leaving deterministic room for the actual candidate speech.
_CONTEXT_CHAR_LIMITS = {
    "summary": 2_400,
    "creator_intent": 320,
    "main_topic": 260,
    "product_or_subject": 260,
    "story_logic": 600,
    "edit_mode": 80,
}


def _compact_source_context(source_context: tuple[tuple[str, str | float], ...]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in source_context:
        name = str(key)
        if isinstance(value, str):
            normalized = " ".join(value.split())
            limit = _CONTEXT_CHAR_LIMITS.get(name, 240)
            compact[name] = normalized[:limit]
        else:
            compact[name] = value
    return compact


def _compact_evidence(raw_evidence: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Keep all ordinary evidence while preventing a malformed string from owning payload."""
    evidence: dict[str, Any] = {}
    for key, value in raw_evidence:
        name = str(key)[:80]
        if isinstance(value, str):
            evidence[name] = value[:120]
        else:
            evidence[name] = value
    return evidence


def _candidate_rows(
    session: EditorialSession,
    *,
    text_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in session.candidates:
        text = str(candidate.text or "").strip()
        if len(text) > text_limit:
            text = text[:text_limit]
        rows.append({
            "clip_id": candidate.clip_id,
            "text": text,
            "duration_sec": round(candidate.duration_sec, 3),
            "local_label": candidate.local_label,
            "local_confidence": round(float(candidate.local_confidence), 4),
            "evidence": _compact_evidence(candidate.evidence),
        })
    return rows


def _rules(cleanup_task: bool) -> list[str]:
    # The Google transport prompt already carries the detailed editorial contract. Keep
    # only the provider-neutral invariants here rather than duplicating several thousand
    # characters of prose inside every request.
    common = [
        "reference every supplied clip_id exactly once; never invent ids or timestamps",
        "use source_context only to understand the full message/story; label candidates only",
        "judge completeness, retry intent, speech structure, and visual/performance evidence together",
        "preserve unique coherent audience-facing information; use uncertain if evidence is insufficient",
    ]
    if cleanup_task:
        return common + [
            "failed = stumble, false start, word-search, incomplete/abandoned delivery, or inferior same-idea retry",
            "bts = recording-process self-talk, frustration, script/notes consultation, breaking character, or reset/waiting behavior",
            "prefer a later complete clean delivery over an earlier partial duplicate, but never delete genuinely new story coverage",
        ]
    return common + [
        "choose a winner only for genuine competing retries; otherwise independent valid speech may be keep",
    ]


def _payload(
    session: EditorialSession,
    *,
    source_context: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    rules: list[str],
) -> dict[str, Any]:
    return {
        "task": session.task,
        "session_id": session.session_id,
        "source_asset_id": session.source_asset_id,
        "source_context": dict(source_context),
        "local_confidence": round(float(session.local_confidence), 4),
        "conflict_score": round(float(session.conflict_score), 4),
        "allowed_labels": ["winner", "alternate", "failed", "bts", "uncertain", "keep"],
        "rules": rules,
        "candidates": candidates,
    }


def _shrink_context_once(source_context: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically reduce only long context strings; short identifying fields survive."""
    shrunk: dict[str, Any] = {}
    for key, value in source_context.items():
        if not isinstance(value, str) or len(value) <= 120:
            shrunk[key] = value
            continue
        target = max(120, int(len(value) * 0.75))
        shrunk[key] = value[:target]
    return shrunk


def build_compact_editorial_payload(
    session: EditorialSession,
    *,
    cost_policy: HybridCostPolicy = HybridCostPolicy(),
) -> dict[str, Any]:
    if not session.candidates:
        raise ValueError("hybrid payload requires candidates")
    if len(session.candidates) > cost_policy.max_candidates_per_call:
        raise ValueError("hybrid payload candidate budget exceeded")

    cleanup_task = session.task == "classify_recording_process_within_single_creator_session"
    rules = _rules(cleanup_task)
    source_context = _compact_source_context(session.source_context)
    text_limit = max(1, int(cost_policy.max_chars_per_candidate))
    candidates = _candidate_rows(session, text_limit=text_limit)
    payload = _payload(
        session,
        source_context=source_context,
        candidates=candidates,
        rules=rules,
    )

    # Fit valid candidate windows deterministically instead of failing open before Gemini.
    # Candidate speech shrinks first because whole-source context is shared by all members;
    # then long context strings shrink gradually if unusual evidence still consumes space.
    # No candidates, ids, labels, or evidence keys are dropped.
    target_chars = min(
        int(cost_policy.max_payload_chars),
        int(cost_policy.max_estimated_input_tokens) * 3,
    )
    target_chars = max(1, target_chars)

    for _ in range(10):
        payload_chars = len(repr(payload))
        if payload_chars <= target_chars:
            break

        candidate_count = max(1, len(session.candidates))
        # Measure fixed overhead using identical rows with empty speech. This avoids a
        # guess based on the number of candidates and adapts automatically to evidence.
        empty_rows = _candidate_rows(session, text_limit=1)
        for row in empty_rows:
            row["text"] = ""
        overhead_payload = _payload(
            session,
            source_context=source_context,
            candidates=empty_rows,
            rules=rules,
        )
        overhead = len(repr(overhead_payload))
        available_for_text = max(0, target_chars - overhead - 160)
        next_text_limit = max(48, min(text_limit - 1, available_for_text // candidate_count))

        if next_text_limit < text_limit:
            text_limit = next_text_limit
            candidates = _candidate_rows(session, text_limit=text_limit)
        else:
            source_context = _shrink_context_once(source_context)

        payload = _payload(
            session,
            source_context=source_context,
            candidates=candidates,
            rules=rules,
        )
    else:
        payload_chars = len(repr(payload))

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
