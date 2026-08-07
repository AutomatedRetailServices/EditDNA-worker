"""Fail-open execution observability for optional V1 intelligence layers.

The media pipeline keeps model failures non-fatal. This module turns the
resulting clip metadata into explicit, JSON-safe Semantic V2 execution status
without changing scoring, slot, keep/delete, or composer decisions.
"""
from collections import Counter
from typing import Any, Dict, Iterable, Optional

from worker.models.openai_client import is_openai_available


def _annotate_clip(
    clip: Dict[str, Any],
    *,
    requested: bool,
    provider_available: bool,
) -> tuple[str, Optional[str]]:
    meta = clip.setdefault("meta", {})
    semantic = meta.get("semantic_v2")

    if not requested:
        status, fallback_reason = "not_requested", None
    elif isinstance(semantic, dict):
        if bool(semantic.get("applied")):
            status, fallback_reason = "applied", None
        elif bool(semantic.get("abstain")):
            status, fallback_reason = "abstained", "model_abstained"
        else:
            status, fallback_reason = "fallback_only", "unsafe_to_apply"
    elif not provider_available:
        status, fallback_reason = "provider_unavailable", "provider_unavailable"
    else:
        status, fallback_reason = "classifier_no_result", "missing_semantic_result"

    meta["semantic_v2_execution_status"] = status
    if fallback_reason is None:
        meta.pop("semantic_v2_fallback_reason", None)
    else:
        meta["semantic_v2_fallback_reason"] = fallback_reason
    return status, fallback_reason


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
    clip_statuses = Counter()
    fallback_reasons = Counter()
    for clip in source_clips:
        clip_status, fallback_reason = _annotate_clip(
            clip,
            requested=requested,
            provider_available=provider_available,
        )
        clip_statuses[clip_status] += 1
        if fallback_reason:
            fallback_reasons[fallback_reason] += 1

    semantic_records = [
        (clip.get("meta") or {}).get("semantic_v2")
        for clip in source_clips
        if isinstance((clip.get("meta") or {}).get("semantic_v2"), dict)
    ]
    applied = sum(bool(record.get("applied")) for record in semantic_records)
    abstained = sum(bool(record.get("abstain")) for record in semantic_records)
    unapplied = len(semantic_records) - applied

    if not requested:
        status = "not_requested"
    elif not source_clips:
        status = "no_clauses"
        fallback_reasons["no_clauses"] += 1
    elif clip_statuses.get("applied", 0) == len(source_clips):
        status = "applied"
    elif clip_statuses.get("applied", 0):
        status = "partially_applied"
    elif clip_statuses.get("provider_unavailable", 0) == len(source_clips):
        status = "provider_unavailable"
    elif clip_statuses.get("classifier_no_result", 0) == len(source_clips):
        status = "classifier_no_result"
    else:
        status = "fallback_only"

    return {
        "source_index": source_index,
        "status": status,
        "clip_count": len(source_clips),
        "semantic_results": len(semantic_records),
        "applied": applied,
        "abstained": abstained,
        "unapplied": unapplied,
        "clip_status_counts": dict(clip_statuses),
        "fallback_reasons": dict(fallback_reasons),
    }


def attach_semantic_execution(
    result: Dict[str, Any],
    *,
    requested: bool,
    provider_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Attach per-clip and per-source Semantic V2 execution/fallback metadata."""
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
    clip_status_counts = Counter()
    fallback_reasons = Counter()
    for source in sources:
        clip_status_counts.update(source["clip_status_counts"])
        fallback_reasons.update(source["fallback_reasons"])

    result["semantic_execution"] = {
        "requested": bool(requested),
        "provider_available": available,
        "sources": sources,
        "status_counts": dict(status_counts),
        "clip_status_counts": dict(clip_status_counts),
        "fallback_reasons": dict(fallback_reasons),
    }
    return result
