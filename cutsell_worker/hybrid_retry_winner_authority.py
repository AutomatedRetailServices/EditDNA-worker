"""Final authority for a proven failed attempt followed by its clean retry winner.

Human Gold for Video 00 exposed a conservative gap in Hybrid cleanup: Gemini can label
an attempt ``failed`` at 0.80 with a strong local ``retry_setup`` event while a nearby
later delivery is a high-confidence ``winner`` of the same idea. The generic delete gate
uses a higher threshold, so both deliveries survive and composition renders the fumble
plus the retake.

This pass is intentionally narrow. It runs after Hybrid guards and removes the earlier
attempt only when all of the following are true:
- semantic label is ``failed`` with confidence >= 0.80;
- local whole-video evidence contains an authoritative retry_setup >= 0.84;
- a later high-confidence winner is nearby in the same source;
- lexical coverage proves the winner is the same communication attempt;
- numeric facts are compatible.

It never deletes a winner and fails open when the peer relationship is ambiguous.

D-109/D-110 (authority collision fix, see docs/CUTSELL_DECISIONS.md): this
module's own ``_same_retry_attempt`` is a second, independently-computed
"is this the same retry attempt" judgment -- looser than, and blind to,
``complete_retry_identity_guard.py``'s stricter ``sequence_identity`` check
that ``hybrid_session_cleanup.py`` already consulted for the SAME failed
candidate this run. D-109's forensic proved a real pimples-family run
where that stricter guard explicitly rejected a candidate as a valid
replacement (``SEQUENCE_IDENTITY_BELOW_THRESHOLD``) while this module's
own looser test would independently say "same retry attempt" for the
identical pair -- letting a later, weaker authority override an earlier,
stricter one's explicit rejection. Fix: before removing ``failed`` in
favor of a proposed ``winner``, check whether THIS SAME RUN's own
``complete_retry_identity_guard`` evidence (threaded in via
``session_diagnostics``) already recorded a rejection naming that EXACT
(failed, winner) pair; if so, decline the removal and record
``prior_replacement_rejection_respected`` instead. This reuses existing
evidence verbatim (no new heuristic, no threshold recomputed) and is
directional: a rejection recorded for (X, Y) never blocks Y from
competing with, or replacing, any OTHER candidate.

D-113 (shared verdict-consumption contract): D-112's forensic sweep proved
this exact collision shape recurring in at least one other chain hook
(``hybrid_cross_group_retry_integrity``). The rejection-lookup helper this
module introduced under D-110 is now owned by ``complete_retry_identity_
guard.py`` itself (``prior_replacement_rejections``) so every destructive
authority in the chain consumes the identical, single implementation
instead of each hook copy-pasting its own. This module's own behavior,
tests, and diagnostics are unchanged -- this is a call-site refactor only.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from .complete_retry_identity_guard import (
    SEQUENCE_IDENTITY_BELOW_THRESHOLD,
    prior_replacement_rejections,
)
from .contracts import CandidateTake
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _numbers(text: str) -> set[str]:
    return {token for token in _tokens(text) if any(ch.isdigit() for ch in token)}


def _retry_setup_confidence(
    take: CandidateTake,
    context: WholeVideoContext | None,
) -> float:
    if context is None:
        return 0.0
    for source in context.sources:
        if source.source_asset_id != take.source_asset_id:
            continue
        confidence = 0.0
        for event in source.events:
            if str(event.kind).strip().lower().replace("-", "_").replace(" ", "_") != "retry_setup":
                continue
            if event.end < take.start - 0.25 or event.start > take.end + 0.75:
                continue
            confidence = max(confidence, float(event.confidence))
        return confidence
    return 0.0


def _same_retry_attempt(failed: CandidateTake, winner: CandidateTake) -> tuple[bool, dict]:
    left = _content(failed.text)
    right = _content(winner.text)
    if len(left) < 3 or len(right) < 3:
        return False, {}
    shared = left & right
    failed_cov = len(shared) / max(1, len(left))
    winner_cov = len(shared) / max(1, len(right))
    numbers_left = _numbers(failed.text)
    numbers_right = _numbers(winner.text)
    numbers_ok = not numbers_left or not numbers_right or numbers_left == numbers_right

    incomplete_prefix_covered = bool(
        not failed.complete_idea
        and len(shared) >= 3
        and winner_cov >= 0.70
    )
    enough = bool(
        numbers_ok
        and (
            (len(shared) >= 4 and max(failed_cov, winner_cov) >= 0.45)
            or (len(shared) >= 3 and min(failed_cov, winner_cov) >= 0.55)
            or incomplete_prefix_covered
        )
    )
    return enough, {
        "shared_content_tokens": sorted(shared),
        "shared_count": len(shared),
        "failed_coverage": round(failed_cov, 4),
        "winner_coverage": round(winner_cov, 4),
        "incomplete_prefix_covered": incomplete_prefix_covered,
        "numbers_ok": numbers_ok,
    }


def enforce_proven_retry_winners(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    context: WholeVideoContext | None,
    *,
    session_diagnostics: Iterable[dict] = (),
    failed_confidence: float = 0.80,
    winner_confidence: float = 0.90,
    retry_setup_confidence: float = 0.84,
    maximum_gap_sec: float = 20.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    prior_rejections = prior_replacement_rejections(session_diagnostics)
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for failed in kept_tuple:
        label, confidence = semantic.get(failed.clip_id, ("", 0.0))
        if label != "failed" or confidence < failed_confidence:
            continue
        retry_conf = _retry_setup_confidence(failed, context)
        if retry_conf < retry_setup_confidence:
            continue

        candidates = []
        for winner in kept_tuple:
            if winner.clip_id == failed.clip_id or winner.source_asset_id != failed.source_asset_id:
                continue
            if winner.start < failed.end:
                continue
            gap = float(winner.start - failed.end)
            if gap > maximum_gap_sec:
                continue
            winner_label, winner_conf = semantic.get(winner.clip_id, ("", 0.0))
            if winner_label != "winner" or winner_conf < winner_confidence:
                continue
            same, evidence = _same_retry_attempt(failed, winner)
            if not same:
                continue
            candidates.append((gap, -winner_conf, winner.start, winner, winner_conf, evidence))

        if not candidates:
            continue
        gap, _, _, winner, winner_conf, evidence = min(candidates, key=lambda item: item[:3])

        # D-109/D-110: this run's own complete_retry_identity_guard evidence
        # already rejected THIS EXACT (failed, winner) pair as a valid
        # replacement -- an earlier, stricter authority's explicit finding.
        # Never override it with this module's own looser test. Directional
        # only: a rejection recorded for (failed, winner) never affects any
        # OTHER pair, so `winner` remains free to compete with/replace any
        # other candidate on its own separate evidence.
        rejected_replacement_id = prior_rejections.get(failed.clip_id)
        if rejected_replacement_id is not None and rejected_replacement_id == winner.clip_id:
            diagnostics.append({
                "clip_id": failed.clip_id,
                "reason": "prior_replacement_rejection_respected",
                "final_reason": "prior_replacement_rejection_respected",
                "proposed_winner_clip_id": winner.clip_id,
                "prior_replacement_rejection_found": True,
                "prior_replacement_rejection_reason": SEQUENCE_IDENTITY_BELOW_THRESHOLD,
                "retry_winner_deletion_applied": False,
                "failed_confidence": round(confidence, 4),
                "retry_setup_confidence": round(retry_conf, 4),
                "winner_clip_id": winner.clip_id,
                "winner_confidence": round(winner_conf, 4),
                "gap_sec": round(gap, 3),
                **evidence,
                "failed_text": failed.text,
                "winner_text": winner.text,
            })
            continue

        removed_ids.add(failed.clip_id)
        diagnostics.append({
            "clip_id": failed.clip_id,
            "reason": "failed_attempt_yields_to_proven_later_retry_winner",
            "final_reason": "failed_attempt_yields_to_proven_later_retry_winner",
            "proposed_winner_clip_id": winner.clip_id,
            "prior_replacement_rejection_found": False,
            "prior_replacement_rejection_reason": None,
            "retry_winner_deletion_applied": True,
            "failed_confidence": round(confidence, 4),
            "retry_setup_confidence": round(retry_conf, 4),
            "winner_clip_id": winner.clip_id,
            "winner_confidence": round(winner_conf, 4),
            "gap_sec": round(gap, 3),
            **evidence,
            "failed_text": failed.text,
            "winner_text": winner.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_hybrid_retry_winner_authority() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_retry_winner_authority", False):
        return

    def apply_with_retry_winner_authority(*args, **kwargs):
        context = kwargs.get("context")
        if context is None and len(args) >= 2:
            context = args[1]
        result = original(*args, **kwargs)
        if not result.kept or not result.semantic_decisions:
            return result
        kept, extra_deleted, authority_diagnostics = enforce_proven_retry_winners(
            result.kept,
            result.semantic_decisions,
            context,
            session_diagnostics=result.diagnostics,
        )
        if not authority_diagnostics:
            return result

        input_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in extra_deleted)
        deleted = tuple(take for take in input_takes if take.clip_id in deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_retry_winner_authority": list(authority_diagnostics),
            # D-109/D-110: only ACTUALLY-applied removals belong here -- a
            # declined ("prior_replacement_rejection_respected") entry is
            # observability only, its candidate was never deleted.
            "deleted_ids": [
                item["clip_id"] for item in authority_diagnostics
                if item.get("retry_winner_deletion_applied", True)
            ],
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_retry_winner_authority._cutsell_hybrid_retry_winner_authority = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_retry_winner_authority
