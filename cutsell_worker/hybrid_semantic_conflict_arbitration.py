"""Resolve contradictory Hybrid labels without changing retry/composite structure.

Hybrid evaluates overlapping windows, so one complete delivery can be called a strong
winner in one window and failed in another. This authority restores only complete deleted
deliveries whose strongest winner/keep evidence is at least as strong as conflicting
failed/bts evidence. It does not decide retry-vs-complementary relationships; that belongs
to final selection after Composite and post-selection splitting.
"""
from __future__ import annotations

import sys


def _strengths(rows):
    out = {}
    for clip_id, label, confidence in rows:
        bucket = out.setdefault(str(clip_id), {})
        label = str(label)
        bucket[label] = max(float(confidence), bucket.get(label, 0.0))
    return out


def _explicit_replacements(diagnostics):
    replaced = set()

    def visit(value):
        if isinstance(value, dict):
            clip_id = str(value.get("clip_id") or value.get("removed_clip_id") or "")
            replacement = value.get("later_retry_replacement_id") or value.get("winner_clip_id")
            reason = str(value.get("reason") or value.get("reason_code") or "").casefold()
            if clip_id and replacement and ("retry" in reason or "replacement" in reason or "yields" in reason):
                replaced.add(clip_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(diagnostics)
    return replaced


def conflict_restore_ids(source_takes, deleted_ids, semantic_rows, diagnostics=(), *, winner_min=0.90, failed_min=0.80, margin=0.02):
    deleted_ids = {str(x) for x in deleted_ids}
    strengths = _strengths(semantic_rows)
    replaced = _explicit_replacements(diagnostics)
    restore = set()
    for take in source_takes:
        clip_id = str(take.clip_id)
        if clip_id not in deleted_ids or clip_id in replaced or not bool(take.complete_idea):
            continue
        labels = strengths.get(clip_id) or {}
        winner = max(labels.get("winner", 0.0), labels.get("keep", 0.0))
        failed = max(labels.get("failed", 0.0), labels.get("bts", 0.0))
        if winner < winner_min or failed < failed_min:
            continue
        if winner + 1e-9 < failed + margin:
            continue
        restore.add(clip_id)
    return restore


def reconcile(result, source_takes):
    restore_ids = conflict_restore_ids(
        source_takes,
        {take.clip_id for take in result.deleted},
        result.semantic_decisions,
        result.diagnostics,
    )
    if not restore_ids:
        return result
    kept_ids = {take.clip_id for take in result.kept} | restore_ids
    kept = tuple(take for take in source_takes if take.clip_id in kept_ids)
    deleted = tuple(take for take in source_takes if take.clip_id not in kept_ids)
    strengths = _strengths(result.semantic_decisions)
    audit = []
    for clip_id in sorted(restore_ids):
        labels = strengths.get(clip_id) or {}
        audit.append({
            "clip_id": clip_id,
            "reason": "conflicting_hybrid_semantics_stronger_winner_fails_open",
            "winner_confidence": round(max(labels.get("winner", 0.0), labels.get("keep", 0.0)), 4),
            "failed_confidence": round(max(labels.get("failed", 0.0), labels.get("bts", 0.0)), 4),
        })
    diagnostics = tuple(result.diagnostics) + ({"hybrid_semantic_conflict_arbitration": audit, "restored_ids": sorted(restore_ids)},)
    return type(result)(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics,
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_semantic_conflict_arbitration() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_semantic_conflict_arbitration", False):
        return

    def apply_with_conflict_arbitration(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)
        return reconcile(result, source_takes)

    apply_with_conflict_arbitration._cutsell_hybrid_semantic_conflict_arbitration = True
    apply_with_conflict_arbitration._cutsell_hybrid_composite_best_take = bool(getattr(original, "_cutsell_hybrid_composite_best_take", False))
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_conflict_arbitration
    pipeline_module = sys.modules.get(f"{__package__}.pipeline")
    if pipeline_module is not None:
        pipeline_module.apply_hybrid_session_cleanup = apply_with_conflict_arbitration
