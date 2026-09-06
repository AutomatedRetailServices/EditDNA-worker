"""Coalesce over-segmented selected clips when source continuity is proven.

Best-Take and semantic stages may split one clean creator delivery into several
DraftClips. Rendering those fragments independently can manufacture jump cuts even
when the underlying source between them contains only a tiny natural breath/gesture.

This final draft pass merges adjacent selected clips only when they come from the same
source, the omitted source gap is very small, and whole-video evidence does not show a
retry/fumble/reset in that gap. It never deletes spoken words; it only restores source
continuity that earlier segmentation unnecessarily broke. Ambiguity fails open.

Boundary-authorized microcuts are explicit final-timeline decisions and must never be
re-coalesced here. Selection chooses what survives; Boundary owns the exact physical
cut once that cut has been proven speech-safe.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import DraftClip, SemanticRole

_FAILURE_KINDS = frozenset({
    "retry_setup",
    "false_start",
    "wrong_take",
    "searching_for_words",
    "breaking_character",
    "unintentional_dead_air",
})
_RESET_KINDS = frozenset({
    "body_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
    "hand_motion_reset_candidate",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _events_for_source(diagnostics: dict, source_asset_id: str) -> tuple[dict, ...]:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(event for event in (source.get("events") or ()) if isinstance(event, dict))
    return ()


def _has_blocking_event(events: tuple[dict, ...], gap_start: float, gap_end: float) -> bool:
    # Small padding catches a reset that starts just before/after the omitted gap.
    window_start = gap_start - 0.12
    window_end = gap_end + 0.12
    for event in events:
        start = float(event.get("start") or 0.0)
        end = float(event.get("end") or start)
        if end < window_start or start > window_end:
            continue
        kind = _kind(event.get("kind"))
        confidence = float(event.get("confidence") or 0.0)
        if kind in _FAILURE_KINDS and confidence >= 0.78:
            return True
        if kind == "body_reset_candidate" and confidence >= 0.90:
            return True
        if kind in {"camera_disengagement_candidate", "hand_motion_reset_candidate"} and confidence >= 0.82:
            return True
        if kind == "facial_expression_shift_candidate" and confidence >= 0.86:
            return True
    return False


def _is_boundary_authorized_gap(diagnostics: dict, gap_start: float, gap_end: float) -> bool:
    """Return True when a prior Boundary pass explicitly authorized this microcut."""
    tolerance = 0.035
    for item in diagnostics.get("post_selection_interior_gap_trim") or ():
        if not isinstance(item, dict):
            continue
        if str(item.get("decision") or "split") != "split":
            continue
        start = item.get("removed_gap_start")
        end = item.get("removed_gap_end")
        if start is None or end is None:
            continue
        if abs(float(start) - float(gap_start)) <= tolerance and abs(float(end) - float(gap_end)) <= tolerance:
            return True
    return False


def _join_text(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}".strip()


def coalesce_selected_source_continuity(
    selected: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_source_gap_sec: float = 0.45,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    clips = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    if not clips:
        return (), ()

    output: list[DraftClip] = []
    audit: list[dict] = []
    current = clips[0]

    for nxt in clips[1:]:
        same_source = (
            current.source_asset_id == nxt.source_asset_id
            and current.source_order == nxt.source_order
        )
        gap = float(nxt.start) - float(current.end)
        if not same_source or gap < -0.03 or gap > maximum_source_gap_sec:
            output.append(current)
            current = nxt
            continue

        if _is_boundary_authorized_gap(diagnostics, float(current.end), float(nxt.start)):
            output.append(current)
            current = nxt
            continue

        events = _events_for_source(diagnostics, current.source_asset_id)
        if _has_blocking_event(events, float(current.end), float(nxt.start)):
            output.append(current)
            current = nxt
            continue

        # Keep role only when both fragments agree. Role disagreement is metadata, not
        # a reason to manufacture a visible cut; OTHER is the safe neutral merge label.
        role = current.semantic_role if current.semantic_role == nxt.semantic_role else SemanticRole.OTHER
        merged_words = tuple(sorted(tuple(current.words) + tuple(nxt.words), key=lambda w: (float(w.start), float(w.end))))
        parent_ids = [current.clip_id, nxt.clip_id]
        merged = replace(
            current,
            clip_id=f"{current.clip_id}__continuity__{nxt.clip_id}",
            end=float(nxt.end),
            text=_join_text(current.text, nxt.text),
            caption_text=_join_text(current.caption_text, nxt.caption_text),
            words=merged_words,
            semantic_role=role,
            take_group_id=(current.take_group_id if current.take_group_id == nxt.take_group_id else None),
        )
        audit.append({
            "authority": "post_selection_continuity_coalescer",
            "left_clip_id": current.clip_id,
            "right_clip_id": nxt.clip_id,
            "source_gap_start": round(float(current.end), 3),
            "source_gap_end": round(float(nxt.start), 3),
            "source_gap_sec": round(max(0.0, gap), 3),
            "reason": "same_source_micro_gap_without_retry_or_reset_evidence",
            "merged_parent_ids": parent_ids,
        })
        current = merged

    output.append(current)
    return tuple(output), tuple(audit)


def install_post_selection_continuity_coalescer() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_continuity_coalescer", False):
        return

    def build_with_continuity_coalescer(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, audit = coalesce_selected_source_continuity(draft.selected, diagnostics)
        if not audit:
            return result
        diagnostics["post_selection_continuity_coalescer"] = list(audit)
        repaired = replace(draft, selected=selected, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_continuity_coalescer._cutsell_post_selection_continuity_coalescer = True
    pipeline.build_flow_b_draft = build_with_continuity_coalescer
