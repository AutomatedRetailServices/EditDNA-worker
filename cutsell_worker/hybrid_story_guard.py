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

import re
import unicodedata
from typing import Iterable

from .contracts import CandidateTake
from .story_coverage_guard import restore_unique_story_coverage
from .whole_video_analysis import WholeVideoContext

_AUTHORITATIVE_DELETE_BASES = frozenset({
    "semantic_failed_plus_local_performance",
})
_NEGATIONS = frozenset({"no", "not", "never", "nunca", "nadie", "ningun", "ninguna", "sin"})


def _normalized_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return {
        token for token in re.findall(r"[a-z0-9%]+", normalized)
        if len(token) > 2 or token.isdigit() or "%" in token
    }


def _critical_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = set(re.findall(r"[a-z0-9%]+", normalized))
    critical = {token for token in tokens if any(ch.isdigit() for ch in token)}
    critical.update(token for token in tokens if token in _NEGATIONS)
    return critical


def _covered_by_kept_delivery(take: CandidateTake, kept: Iterable[CandidateTake]) -> bool:
    """Return True when an incomplete failed retry adds no critical information.

    Story coverage is allowed to rescue genuinely unique material. It must not resurrect
    an incomplete retry merely because its wording differs when an already-kept delivery
    covers most of the same idea and preserves all numeric/negation facts.
    """
    take_tokens = _normalized_tokens(take.text)
    if not take_tokens:
        return False
    critical = _critical_tokens(take.text)
    for peer in kept:
        if float(getattr(peer, "start", 0.0)) >= float(getattr(take, "start", 0.0)):
            continue
        peer_tokens = _normalized_tokens(peer.text)
        coverage = len(take_tokens & peer_tokens) / max(1, len(take_tokens))
        if coverage < 0.50:
            continue
        if not critical.issubset(_critical_tokens(peer.text) | peer_tokens):
            continue
        return True
    return False


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
    suppressed_covered_retries: list[dict] = []
    for take in result.deleted:
        if take.clip_id in authoritative_delete_ids:
            continue
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        if label in {"failed", "bts"} and confidence >= hard_semantic_confidence:
            continue
        if (
            label == "failed"
            and confidence >= 0.90
            and not bool(getattr(take, "complete_idea", False))
            and _covered_by_kept_delivery(take, result.kept)
        ):
            suppressed_covered_retries.append({
                "clip_id": take.clip_id,
                "reason": "incomplete_failed_retry_covered_by_kept_delivery",
                "hybrid_confidence": round(confidence, 4),
            })
            continue
        eligible_deleted.append(take)

    if not eligible_deleted:
        if not suppressed_covered_retries:
            return result
        return type(result)(
            kept=result.kept,
            deleted=result.deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=tuple(result.diagnostics) + ({
                "hybrid_story_coverage_suppressed_retries": suppressed_covered_retries,
            },),
            semantic_decisions=result.semantic_decisions,
        )

    kept, _, diagnostics = restore_unique_story_coverage(
        result.kept,
        tuple(eligible_deleted),
        source_tuple,
        context,
    )
    if not diagnostics:
        if not suppressed_covered_retries:
            return result
        return type(result)(
            kept=result.kept,
            deleted=result.deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=tuple(result.diagnostics) + ({
                "hybrid_story_coverage_suppressed_retries": suppressed_covered_retries,
            },),
            semantic_decisions=result.semantic_decisions,
        )

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
        "suppressed_covered_retries": suppressed_covered_retries,
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
