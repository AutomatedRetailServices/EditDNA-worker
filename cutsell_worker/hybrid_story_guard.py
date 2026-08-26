"""Post-Hybrid protection for unique coherent story coverage.

A deterministic story guard may restore a long audience-facing paragraph because it is
unique, information-dense, and has no competing retry or strong recording-failure
evidence. Hybrid cleanup must not immediately delete that same paragraph from weak
semantic evidence alone.

Only a Hybrid deletion corroborated by local performance evidence is irrevocable. A
semantic-only "failed" label, even at high confidence, may still pass through the unique
story-coverage guard because model confidence is not physical proof that the delivery is
a failed take. Extremely high semantic confidence is still protected separately by the
hard semantic floor in ``restore_hybrid_story_coverage``.
"""
from __future__ import annotations

from typing import Iterable

from .contracts import CandidateTake
from .story_coverage_guard import restore_unique_story_coverage
from .whole_video_analysis import WholeVideoContext

_AUTHORITATIVE_DELETE_BASES = frozenset({
    "semantic_failed_plus_local_performance",
})


def _authoritative_applied_delete_ids(result) -> set[str]:
    """Extract Hybrid deletions that later fail-open guards are not allowed to undo."""
    authoritative: set[str] = set()
    for diagnostic in tuple(getattr(result, "diagnostics", ()) or ()):
        if not isinstance(diagnostic, dict):
            continue
        decisions = diagnostic.get("decisions")
        if not isinstance(decisions, (list, tuple)):
            continue
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            if decision.get("applied_delete") is not True:
                continue
            basis = str(decision.get("delete_basis") or "")
            clip_id = str(decision.get("clip_id") or "")
            if clip_id and basis in _AUTHORITATIVE_DELETE_BASES:
                authoritative.add(clip_id)
    return authoritative


def restore_hybrid_story_coverage(
    source_takes: Iterable[CandidateTake],
    result,
    context: WholeVideoContext | None,
    *,
    hard_semantic_confidence: float = 0.985,
):
    """Restore unique story paragraphs deleted only on non-authoritative Hybrid evidence."""
    source_tuple = tuple(source_takes)
    if not source_tuple or not result.deleted:
        return result

    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in result.semantic_decisions
    }
    authoritative_delete_ids = _authoritative_applied_delete_ids(result)

    eligible_deleted = []
    for take in result.deleted:
        if take.clip_id in authoritative_delete_ids:
            continue
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        if label in {"failed", "bts"} and confidence >= hard_semantic_confidence:
            continue
        eligible_deleted.append(take)

    if not eligible_deleted:
        return result

    kept, _, diagnostics = restore_unique_story_coverage(
        result.kept,
        tuple(eligible_deleted),
        source_tuple,
        context,
    )
    if not diagnostics:
        return result

    restored_ids = {str(item["clip_id"]) for item in diagnostics}
    deleted = tuple(take for take in result.deleted if take.clip_id not in restored_ids)
    guard_diagnostics = tuple(result.diagnostics) + ({
        "hybrid_story_coverage_guard": [
            {
                **dict(item),
                "hybrid_label": semantic.get(str(item["clip_id"]), ("", 0.0))[0],
                "hybrid_confidence": round(
                    semantic.get(str(item["clip_id"]), ("", 0.0))[1], 4
                ),
            }
            for item in diagnostics
        ],
        "restored_ids": sorted(restored_ids),
        "authoritative_delete_ids": sorted(authoritative_delete_ids),
    },)

    return type(result)(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=guard_diagnostics,
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_story_coverage_guard() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_story_guard", False):
        return

    def apply_with_hybrid_story_guard(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            call_args = (source_takes, *args[1:])
            context = args[1] if len(args) > 1 else kwargs.get("context")
            result = original(*call_args, **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            context = call_kwargs.get("context")
            result = original(**call_kwargs)
        return restore_hybrid_story_coverage(source_takes, result, context)

    apply_with_hybrid_story_guard._cutsell_hybrid_story_guard = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_hybrid_story_guard
