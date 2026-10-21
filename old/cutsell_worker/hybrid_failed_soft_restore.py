"""Restore weak ``failed``-only cross-group deletions that lack destructive authority.

Hybrid can call a clean retry ``failed`` with medium confidence even when local Watch +
Listen finds no failure.  Cross-group lexical coverage must not turn that weak semantic
opinion into destructive authority.  This late wrapper restores only candidates deleted
by the cross-group integrity pass when the recorded cross-group decision is ``failed``
with confidence below 0.90.  Alternates and high-confidence failures remain untouched.
"""
from __future__ import annotations


def restore_weak_failed_cross_group_deletions(result, source_takes):
    """Core transform, extracted for direct use by composite_resolver.py.

    Identical logic to what used to live only inside this module's
    install-time monkeypatch closure -- see D-023. ``install_hybrid_
    failed_soft_restore`` below now delegates here so its own
    (monkeypatch-based) tests keep working unchanged.
    """
    restore_ids: set[str] = set()
    audits = []
    for row in tuple(result.diagnostics or ()):
        items = row.get("hybrid_cross_group_retry_integrity") if isinstance(row, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("semantic_label") or "") != "failed":
                continue
            confidence = float(item.get("semantic_confidence") or 0.0)
            if confidence >= 0.90:
                continue
            clip_id = str(item.get("clip_id") or "")
            if not clip_id:
                continue
            restore_ids.add(clip_id)
            audits.append({
                "clip_id": clip_id,
                "reason": "weak_failed_semantics_without_destructive_authority",
                "semantic_confidence": round(confidence, 4),
            })
    if not restore_ids:
        return result

    source_by_id = {take.clip_id: take for take in source_takes}
    restored = tuple(source_by_id[cid] for cid in restore_ids if cid in source_by_id)
    if not restored:
        return result
    deleted = tuple(take for take in result.deleted if take.clip_id not in restore_ids)
    kept_map = {take.clip_id: take for take in result.kept}
    for take in restored:
        kept_map[take.clip_id] = take
    kept = tuple(sorted(kept_map.values(), key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_failed_soft_restore": audits,
        "restored_ids": sorted(restore_ids),
    },)
    return type(result)(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics,
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_failed_soft_restore() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_failed_soft_restore", False):
        return

    def apply_with_soft_restore(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)
        return restore_weak_failed_cross_group_deletions(result, source_takes)

    apply_with_soft_restore._cutsell_hybrid_failed_soft_restore = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_soft_restore
