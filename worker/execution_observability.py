"""Fail-open execution observability for optional V1 intelligence layers.

The media pipeline keeps model failures non-fatal. This module turns the
resulting clip metadata into an explicit, JSON-safe Semantic V2 execution
summary so callers can distinguish a model application from a fallback path.
"""
from collections import Counter
from typing import Any, Dict, Iterable, Optional

from worker.models.openai_client import is_openai_available


def _source_status(
    clips: Iterable[Dict[str, Any]],
    *,
    source_index: int,
    requested: bool,
    provider_available: bool,
) -> Dict[str, Any]:
    source_clips = [
        clip for clip in clips
        if int(clip.get("source_index", 0)) == source_index
    ]
    semantic_records = [
        (clip.get("meta") or {}).get("semantic_v2")
        for clip in source_clips
        if isinstance((clip.get("meta") or {}).get("semantic_v2"), dict)
    ]
    applied = sum(bool(record.get("applied")) for record in semantic_records)
    abstained = sum(bool(record.get("abstain")) for record in semantic_records)
    unapplied = len(semantic_records) - applied

    fallback_reasons = Counter()
    if not requested:
        status = "not_requested"
    elif not source_clips:
        status = "no_clauses"
        fallback_reasons["no_clauses"] += 1
    elif not semantic_records:
        if provider_available:
            status = "classifier_no_result"
            fallback_reasons["provider_or_validation_error"] += 1
        else:
            status = "provider_unavailable"
            fallback_reasons["provider_unavailable"] += 1
    elif applied == len(semantic_records) and len(semantic_records) == len(source_clips):
        status = "applied"
    elif applied:
        status = "partially_applied"
        if abstained:
            fallback_reasons["model_abstained"] += abstained
        if unapplied > abstained:
            fallback_reasons["unsafe_to_apply"] += unapplied - abstained
        if len(semantic_records) < len(source_clips):
            fallback_reasons["missing_semantic_result"] += len(source_clips) - len(semantic_records)
    else:
        status = "fallback_only"
        if abstained:
            fallback_reasons["model_abstained"] += abstained
        if unapplied > abstained:
            fallback_reasons["unsafe_to_apply"] += unapplied - abstained
        if len(semantic_records) < len(source_clips):
            fallback_reasons["missing_semantic_result"] += len(source_clips) - len(semantic_records)

    return {
        "source_index": source_index,
        "status": status,
        "clip_count": len(source_clips),
        "semantic_results": len(semantic_records),
        "applied": applied,
        "abstained": abstained,
        "unapplied": unapplied,
        "fallback_reasons": dict(fallback_reasons),
    }


def attach_semantic_execution(
    result: Dict[str, Any],
    *,
    requested: bool,
    provider_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Attach Semantic V2 execution/fallback status without mutating clip state."""
    clips = list(result.get("clips") or [])
    indices = set(result.get("processed_source_indices") or [])
    indices.update(int(clip.get("source_index", 0)) for clip in clips)
    if not indices and result.get("input_file_count"):
        indices.update(range(int(result["input_file_count"])))

    available = is_openai_available() if provider_available is None else bool(provider_available)
    sources = [
        _source_status(
            clips,
            source_index=index,
            requested=bool(requested),
            provider_available=available,
        )
        for index in sorted(indices)
    ]
    status_counts = Counter(source["status"] for source in sources)
    fallback_reasons = Counter()
    for source in sources:
        fallback_reasons.update(source["fallback_reasons"])

    result["semantic_execution"] = {
        "requested": bool(requested),
        "provider_available": available,
        "sources": sources,
        "status_counts": dict(status_counts),
        "fallback_reasons": dict(fallback_reasons),
    }
    return result
