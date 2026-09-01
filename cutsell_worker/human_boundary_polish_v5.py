"""Boundary-only Watch+Listen micro-polish v5.

This pass is intentionally a pure Boundary authority. It may split an already-selected
spoken delivery only at a source gap between aligned words when stored multimodal reset
evidence is strong enough. It must never promote/demote takes, restore alternates, remove
spoken retry material, or otherwise change the ordered spoken token stream.

Older human_boundary_polish v1-v4 mixed Selection and Boundary responsibilities. V5 no
longer chains those passes. Their selection-changing behavior belongs upstream in the
Selection phase and is deliberately unreachable from this Boundary authority.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Mapping

from .contracts import DraftClip, ProcessingResult, Word
from .human_boundary_polish_v2 import _timeline_proxies, _reset_score

BOUNDARY_REASON_MICRO_VISUAL_RESET_GAP = "remove_micro_visual_reset_word_gap"


def _words_before(words: tuple[Word, ...], end: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.end) <= end + 1e-6)


def _words_after(words: tuple[Word, ...], start: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.start) >= start - 1e-6)


def _fragment_id(clip: DraftClip, *, start: float, end: float) -> str:
    """A physical identity unique to this exact piece -- D-036. Derived from
    the clip's OWN clip_id (its nearest parent, semantic or already a
    fragment) plus its exact new boundaries, so re-splitting an existing
    fragment again always mints a fresh id rather than colliding with the
    piece it was split from (see post_selection_interior_gap_trim.py's
    `_child_id`, the same pattern this mirrors)."""
    digest = hashlib.sha256(
        f"{clip.clip_id}|human_boundary_polish_v5|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{clip.clip_id}__hbp5{digest}"


def _clip_from_words(
    clip: DraftClip, start: float, end: float, words: tuple[Word, ...],
    *, parent_semantic_clip_id: str,
) -> DraftClip | None:
    if end - start < 0.18 or not words:
        return None
    text = " ".join(str(word.text).strip() for word in words).strip()
    if not text:
        return None
    return replace(
        clip, start=start, end=end, words=words, text=text, caption_text=text,
        # D-036: clip_id (semantic identity) is intentionally left unchanged
        # -- CanonicalEditPlan/FinalEditReviewer/Selection Freeze must keep
        # referring to the same semantic clip. Only the physical identity is
        # fresh per piece; parent_semantic_clip_id always points at the true
        # ROOT semantic clip even when splitting an already-split fragment
        # (i.e. it is never itself a fragment id), so every physical sibling
        # of one frozen delivery is discoverable by one shared key regardless
        # of how many Boundary passes touched it.
        render_fragment_id=_fragment_id(clip, start=start, end=end),
        parent_semantic_clip_id=parent_semantic_clip_id,
        boundary_reason=BOUNDARY_REASON_MICRO_VISUAL_RESET_GAP,
    )


def _micro_reset_evidence(timeline, start: float, end: float) -> tuple[float, int]:
    """Return aggregate reset score and number of strong visual reset events."""
    if timeline is None:
        return 0.0, 0
    score = _reset_score(timeline, start, end, pad=0.28)
    strong = 0
    for event in getattr(timeline, "events", ()):
        if event.end < start - 0.28 or event.start > end + 0.28:
            continue
        if event.kind not in {
            "body_reset_candidate",
            "hand_motion_reset_candidate",
            "camera_disengagement_candidate",
            "facial_expression_shift_candidate",
        }:
            continue
        if float(event.confidence) >= 0.82:
            strong += 1
    return score, strong


def _remove_micro_visual_reset_word_gaps(
    clip: DraftClip,
    timeline,
) -> tuple[tuple[DraftClip, ...], list[dict]]:
    words = tuple(sorted(tuple(clip.words), key=lambda word: (word.start, word.end)))
    if len(words) < 2 or timeline is None:
        return (clip,), []

    candidates: list[tuple[float, float, float, int]] = []
    for left, right in zip(words, words[1:]):
        gap_start = float(left.end)
        gap_end = float(right.start)
        gap = gap_end - gap_start

        if gap < 0.22 or gap >= 0.55:
            continue

        score, strong = _micro_reset_evidence(timeline, gap_start, gap_end)
        if gap < 0.34:
            eligible = score >= 1.55 or (score >= 1.30 and strong >= 2)
        else:
            eligible = score >= 1.20 and strong >= 1
        if not eligible:
            continue

        candidates.append((gap_start, gap_end, score, strong))

    if not candidates:
        return (clip,), []

    # D-036: the ROOT semantic clip every physical sibling this call produces
    # reconstructs together -- never a fragment id, even if `clip` itself is
    # already a fragment from an earlier Boundary pass (e.g.
    # post_selection_interior_gap_trim), so all pieces stay discoverable
    # under one shared key regardless of how many splitting passes touched
    # this delivery.
    root_parent = clip.parent_semantic_clip_id or clip.clip_id

    pieces: list[DraftClip] = [clip]
    diagnostics: list[dict] = []
    for gap_start, gap_end, score, strong in candidates:
        rebuilt: list[DraftClip] = []
        changed = False
        for piece in pieces:
            if not (float(piece.start) < gap_start and float(piece.end) > gap_end):
                rebuilt.append(piece)
                continue
            left_words = _words_before(tuple(piece.words), gap_start)
            right_words = _words_after(tuple(piece.words), gap_end)
            left_piece = _clip_from_words(
                piece, float(piece.start), gap_start, left_words,
                parent_semantic_clip_id=root_parent,
            )
            right_piece = _clip_from_words(
                piece, gap_end, float(piece.end), right_words,
                parent_semantic_clip_id=root_parent,
            )
            if left_piece is None or right_piece is None:
                rebuilt.append(piece)
                continue
            rebuilt.extend((left_piece, right_piece))
            changed = True
        if changed:
            pieces = rebuilt
            diagnostics.append({
                "authority": "human_boundary_polish_v5",
                "decision": "split",
                "action": "remove_micro_visual_reset_word_gap",
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration_sec": round(gap_end - gap_start, 3),
                "reset_score": round(score, 3),
                "strong_reset_events": strong,
                "semantic_membership_changed": False,
            })

    if len(pieces) > 1:
        # fragment_index/fragment_count are only meaningful once every gap
        # candidate has been applied and the final piece count is known --
        # stamped here as one last pass rather than threaded through the
        # splitting loop above.
        total = len(pieces)
        pieces = [replace(piece, fragment_index=index, fragment_count=total) for index, piece in enumerate(pieces)]

    return tuple(pieces), diagnostics


def polish_human_boundaries_v5(result: ProcessingResult, local_paths: Mapping[str, str]) -> ProcessingResult:
    """Apply Boundary-only micro-gap splits without invoking legacy Selection polish."""
    if not hasattr(result.draft, "selected"):
        return result

    timelines = _timeline_proxies(result)
    polished: list[DraftClip] = []
    diagnostics: list[dict] = []

    for clip in result.draft.selected:
        pieces, rows = _remove_micro_visual_reset_word_gaps(clip, timelines.get(clip.source_asset_id))
        polished.extend(pieces)
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in rows)

    if not diagnostics:
        return result

    existing = list((result.draft.diagnostics or {}).get("human_boundary_polish") or ())
    diag = dict(result.draft.diagnostics or {})
    diag["human_boundary_polish"] = [*existing, *diagnostics][:600]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=diag)
    return replace(result, draft=draft)
