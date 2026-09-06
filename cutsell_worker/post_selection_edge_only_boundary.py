"""Edge-only boundary trim after Selection is frozen.

This authority may change only ``start`` and ``end`` of already-selected DraftClips.
It never changes clip identity, text, words, semantic role, selected membership, or
relative ordering. The goal is to remove proven non-speech setup/post-roll while making
it impossible for Boundary work to mutate Best Take / Selection.

A trim is allowed only inside the existing clip envelope, never through a spoken Word.
Ambiguity fails open.
"""
from __future__ import annotations

from dataclasses import replace

_AUTHORITATIVE = frozenset({
    "unintentional_dead_air",
    "retry_setup",
    "searching_for_words",
    "false_start",
    "wrong_take",
    "breaking_character",
    "camera_adjustment",
})
_RESET = frozenset({
    "body_reset_candidate",
    "hand_motion_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _events_for_source(diagnostics: dict, source_asset_id: str) -> tuple[dict, ...]:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(event for event in (source.get("events") or ()) if isinstance(event, dict))
    return ()


def _edge_evidence(events, start: float, end: float) -> tuple[bool, list[str]]:
    nearby = []
    for event in events:
        event_start = float(event.get("start") or 0.0)
        event_end = float(event.get("end") or event_start)
        if event_end < start - 0.16 or event_start > end + 0.16:
            continue
        nearby.append(event)

    authoritative = [
        event for event in nearby
        if _kind(event.get("kind")) in _AUTHORITATIVE
        and float(event.get("confidence") or 0.0) >= 0.78
    ]
    if authoritative:
        strongest = max(authoritative, key=lambda item: float(item.get("confidence") or 0.0))
        return True, [
            f"event:{_kind(strongest.get('kind'))}:{float(strongest.get('confidence') or 0.0):.2f}"
        ]

    resets = [
        event for event in nearby
        if _kind(event.get("kind")) in _RESET
        and float(event.get("confidence") or 0.0) >= 0.88
    ]
    kinds = {_kind(event.get("kind")) for event in resets}
    # One generic motion event is not enough. Require corroboration across modalities,
    # or a dense cluster of at least three strong reset events.
    if len(kinds) >= 2 or len(resets) >= 3:
        return True, [
            f"reset:{_kind(event.get('kind'))}:{float(event.get('confidence') or 0.0):.2f}"
            for event in resets[:4]
        ]
    return False, []


def trim_locked_selection_edges(
    selected,
    diagnostics: dict,
    *,
    minimum_leading_slack_sec: float = 0.30,
    minimum_trailing_slack_sec: float = 0.12,
    maximum_slack_sec: float = 3.0,
):
    output = []
    audit = []

    for clip in selected:
        words = tuple(sorted(tuple(clip.words), key=lambda w: (float(w.start), float(w.end))))
        if not words:
            output.append(clip)
            continue

        original_start = float(clip.start)
        original_end = float(clip.end)
        first_start = float(words[0].start)
        last_end = float(words[-1].end)
        new_start = original_start
        new_end = original_end
        actions = []
        events = _events_for_source(diagnostics, clip.source_asset_id)

        leading = first_start - original_start
        if minimum_leading_slack_sec <= leading <= maximum_slack_sec:
            confirmed, evidence = _edge_evidence(events, original_start, first_start)
            if confirmed:
                new_start = first_start
                actions.append({
                    "action": "trim_locked_leading_non_speech_edge",
                    "duration_sec": round(leading, 3),
                    "evidence": evidence,
                })

        trailing = original_end - last_end
        if minimum_trailing_slack_sec <= trailing <= maximum_slack_sec:
            confirmed, evidence = _edge_evidence(events, last_end, original_end)
            if confirmed:
                new_end = last_end
                actions.append({
                    "action": "trim_locked_trailing_non_speech_edge",
                    "duration_sec": round(trailing, 3),
                    "evidence": evidence,
                })

        if not actions or new_end - new_start < 0.25:
            output.append(clip)
            continue

        updated = replace(clip, start=new_start, end=new_end)
        output.append(updated)
        audit.append({
            "clip_id": clip.clip_id,
            "original_start": round(original_start, 3),
            "original_end": round(original_end, 3),
            "result_start": round(new_start, 3),
            "result_end": round(new_end, 3),
            "actions": actions,
            "selection_identity_preserved": True,
        })

    return tuple(output), tuple(audit)


def install_post_selection_edge_only_boundary() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_edge_only_boundary", False):
        return

    def build_with_locked_edge_boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, audit = trim_locked_selection_edges(draft.selected, diagnostics)
        if not audit:
            return result
        diagnostics["post_selection_edge_only_boundary"] = list(audit)
        repaired = replace(draft, selected=selected, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_locked_edge_boundary._cutsell_post_selection_edge_only_boundary = True
    pipeline.build_flow_b_draft = build_with_locked_edge_boundary
