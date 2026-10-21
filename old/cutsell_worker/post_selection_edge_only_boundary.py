"""Edge-only boundary trim after Selection is frozen.

This authority may change only ``start`` and ``end`` of already-selected DraftClips.
It never changes clip identity, text, words, semantic role, selected membership, or
relative ordering. The goal is to remove proven non-speech setup/post-roll while making
it impossible for Boundary work to mutate Best Take / Selection.

A trim is allowed only inside the existing clip envelope, never through a spoken Word.
Ambiguity fails open.

## D-242 -- edge audit completeness (no policy/threshold/behavior change)

D-241's forensic proved this module's own audit row was emitted ONLY when a
trim actually applied -- so a clip whose edge was evaluated and found to need
no trim was indistinguishable, downstream in `pacing_v2_source_audio_handle.
py`, from a clip whose edge was never looked at at all (both collapsed to
`NO_BOUNDARY_PROVENANCE_RECORDED`). This module now emits exactly one audit
row for EVERY selected clip, always, carrying an explicit `leading_edge_
status`/`trailing_edge_status` (see `EDGE_STATUS_*` below) alongside the
existing timing/action fields. No trim decision changes: `result_start`/
`result_end` equal `original_start`/`original_end` whenever no trim applies,
exactly as before this gate -- only the audit row's PRESENCE and its new
status fields are additive.
"""
from __future__ import annotations

from dataclasses import replace

# --- D-242 edge-audit-completeness status vocabulary (shared with
# boundary_engine_pass.py's own audio-edge audit -- both authorities report
# the same five values so a downstream consumer never has to know which
# authority produced a given row). Reused verbatim, never duplicated with
# different spellings. ---------------------------------------------------
EDGE_STATUS_TRIM_APPLIED = "TRIM_APPLIED"
EDGE_STATUS_EVALUATED_NO_TRIM = "EVALUATED_NO_TRIM"
EDGE_STATUS_NO_ELIGIBLE_EVIDENCE = "NO_ELIGIBLE_EVIDENCE"
EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE = "NO_SOURCE_ROOM_DETERMINABLE"
EDGE_STATUS_BLOCKED_BY_SAFETY = "BLOCKED_BY_SAFETY"
EDGE_STATUSES = frozenset({
    EDGE_STATUS_TRIM_APPLIED, EDGE_STATUS_EVALUATED_NO_TRIM,
    EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
    EDGE_STATUS_BLOCKED_BY_SAFETY,
})

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


def _edge_evidence(events, start: float, end: float) -> tuple[bool, list[str], bool]:
    """Returns `(confirmed, evidence, nearby_seen)`. D-242 adds `nearby_seen`
    (any event at all fell within the edge window, confirmed or not) so a
    caller can distinguish "an event was there but did not corroborate
    enough to trim" from "nothing was near this edge at all" -- both were
    previously reported identically as `False, []`."""
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
        ], True

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
        ], True
    return False, [], bool(nearby)


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
        original_start = float(clip.start)
        original_end = float(clip.end)

        if not words:
            # D-242: no word alignment at all means this authority has no
            # basis to even determine whether leading/trailing slack exists
            # -- genuinely unknown, not "no evidence found within a known
            # window". Row is still emitted (previously: silently skipped).
            output.append(clip)
            audit.append({
                "clip_id": clip.clip_id,
                "original_start": round(original_start, 3),
                "original_end": round(original_end, 3),
                "result_start": round(original_start, 3),
                "result_end": round(original_end, 3),
                "actions": [],
                "selection_identity_preserved": True,
                "edge_evaluated": True,
                "leading_edge_status": EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
                "leading_edge_reason": "no_word_alignment_available",
                "trailing_edge_status": EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE,
                "trailing_edge_reason": "no_word_alignment_available",
            })
            continue

        first_start = float(words[0].start)
        last_end = float(words[-1].end)
        new_start = original_start
        new_end = original_end
        actions = []
        events = _events_for_source(diagnostics, clip.source_asset_id)
        source_has_any_events = bool(events)

        leading = first_start - original_start
        leading_candidate = False
        leading_nearby_seen = False
        if minimum_leading_slack_sec <= leading <= maximum_slack_sec:
            confirmed, evidence, leading_nearby_seen = _edge_evidence(events, original_start, first_start)
            if confirmed:
                leading_candidate = True
                new_start = first_start
                actions.append({
                    "action": "trim_locked_leading_non_speech_edge",
                    "duration_sec": round(leading, 3),
                    "evidence": evidence,
                })

        trailing = original_end - last_end
        trailing_candidate = False
        trailing_nearby_seen = False
        if minimum_trailing_slack_sec <= trailing <= maximum_slack_sec:
            confirmed, evidence, trailing_nearby_seen = _edge_evidence(events, last_end, original_end)
            if confirmed:
                trailing_candidate = True
                new_end = last_end
                actions.append({
                    "action": "trim_locked_trailing_non_speech_edge",
                    "duration_sec": round(trailing, 3),
                    "evidence": evidence,
                })

        # D-242: the existing per-clip minimum-remaining-duration floor is
        # unchanged -- still 0.25s, still applied identically. What changes
        # is that a rejection here is now reported as BLOCKED_BY_SAFETY for
        # whichever edge(s) had a candidate, rather than silently reverting
        # with no trace at all.
        floor_blocked = bool(actions) and (new_end - new_start < 0.25)
        if floor_blocked or not actions:
            final_start, final_end, final_actions = original_start, original_end, []
        else:
            final_start, final_end, final_actions = new_start, new_end, actions
        applied_leading = leading_candidate and not floor_blocked
        applied_trailing = trailing_candidate and not floor_blocked

        leading_in_range = minimum_leading_slack_sec <= leading <= maximum_slack_sec
        trailing_in_range = minimum_trailing_slack_sec <= trailing <= maximum_slack_sec

        if applied_leading:
            leading_status, leading_reason = EDGE_STATUS_TRIM_APPLIED, "leading_non_speech_edge_trimmed"
        elif leading_candidate and floor_blocked:
            leading_status, leading_reason = EDGE_STATUS_BLOCKED_BY_SAFETY, "minimum_remaining_duration_floor"
        elif not leading_in_range:
            # A definitive geometric fact (word alignment vs. clip edge is
            # already known) -- not "we don't know", but "there is no
            # material slack here worth evaluating for evidence".
            leading_status, leading_reason = EDGE_STATUS_EVALUATED_NO_TRIM, "leading_slack_outside_eligible_range"
        elif leading_nearby_seen:
            # A slack window existed and an event was found inside it, but
            # it did not corroborate enough to confirm a trim (e.g. a
            # single uncorroborated reset candidate).
            leading_status, leading_reason = EDGE_STATUS_EVALUATED_NO_TRIM, "nearby_evidence_insufficient_to_confirm"
        elif not source_has_any_events:
            leading_status, leading_reason = EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE, "no_recorded_events_for_source"
        else:
            leading_status, leading_reason = EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, "no_event_near_leading_slack_window"

        if applied_trailing:
            trailing_status, trailing_reason = EDGE_STATUS_TRIM_APPLIED, "trailing_non_speech_edge_trimmed"
        elif trailing_candidate and floor_blocked:
            trailing_status, trailing_reason = EDGE_STATUS_BLOCKED_BY_SAFETY, "minimum_remaining_duration_floor"
        elif not trailing_in_range:
            trailing_status, trailing_reason = EDGE_STATUS_EVALUATED_NO_TRIM, "trailing_slack_outside_eligible_range"
        elif trailing_nearby_seen:
            trailing_status, trailing_reason = EDGE_STATUS_EVALUATED_NO_TRIM, "nearby_evidence_insufficient_to_confirm"
        elif not source_has_any_events:
            trailing_status, trailing_reason = EDGE_STATUS_NO_SOURCE_ROOM_DETERMINABLE, "no_recorded_events_for_source"
        else:
            trailing_status, trailing_reason = EDGE_STATUS_NO_ELIGIBLE_EVIDENCE, "no_event_near_trailing_slack_window"

        if final_actions:
            output.append(replace(clip, start=final_start, end=final_end))
        else:
            output.append(clip)

        audit.append({
            "clip_id": clip.clip_id,
            "original_start": round(original_start, 3),
            "original_end": round(original_end, 3),
            "result_start": round(final_start, 3),
            "result_end": round(final_end, 3),
            "actions": final_actions,
            "selection_identity_preserved": True,
            "edge_evaluated": True,
            "leading_edge_status": leading_status,
            "leading_edge_reason": leading_reason,
            "trailing_edge_status": trailing_status,
            "trailing_edge_reason": trailing_reason,
        })

    return tuple(output), tuple(audit)


def install_post_selection_edge_only_boundary() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_edge_only_boundary", False):
        return

    def build_with_locked_edge_boundary(*args, **kwargs):
        result = original(*args, **kwargs)
        # D-097.C/E: owned by the post-Freeze BoundaryEngine pass on the
        # universal path (see boundary_engine_pass.py).
        if str(kwargs.get("boundary_owner") or "pre_freeze") == "post_freeze":
            return result
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
