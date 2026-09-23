"""D-224 -- PACING V2 SOURCE AUDIO HANDLE LIVE EVIDENCE INTEGRATION.
DIAGNOSTIC ONLY. NO LIVE J_CUT/L_CUT/MICRO_AUDIO_OVERLAP AUTHORITY, NO
AUDIO JOIN TREATMENT AUTHORITY, NO RENDERER LIVE AUDIO-HANDLE USE.

Answers the question D-223's own forensic left open: "does widening J/L
candidate-availability evidence with a real `SourceAudioHandle` (D-223)
actually change what D-215's decision engine would recommend, on the
SAME already-computed real evidence this pipeline already has?" -- a
DIAGNOSTIC comparison only. D-142's own live `apply_dialogue_pacing_
transition_pass` output (`HARD_CUT`/`TIGHT_CUT` only) is never touched,
never overwritten, never read back from this module's own output.

## Combined-availability model (this task's own "core model")

For the RIGHT clip of a J-cut candidate pair: `combined_j_available_
window` unions (never blindly sums) two SOURCE-COORDINATE intervals in
the SAME source file --
  (a) the existing in-window silent-head interval `pacing_v2_evidence_
      adapter.available_silent_head_sec(right)` already measures,
      `[right.start, right.start + in_window_head)`;
  (b) the ELIGIBLE portion of `right`'s own PRE_ROLL `SourceAudioHandle`
      (D-223), `[handle.handle_source_start, handle.handle_source_end)`,
      counted ONLY when `handle.handle_status == SAFE_NON_SPEECH_HANDLE`
      -- this task's own explicit gate: word-present/blocked/unknown
      handles contribute ZERO, never partial credit.
Symmetric for the LEFT clip's L-cut tail and its POST_ROLL handle.

`_merged_interval_length` performs a real interval union (sort, merge
touching/overlapping spans, sum merged lengths) rather than `a + b` --
these two intervals are adjacent (the handle's own `handle_source_end`/
`handle_source_start` is, by D-223's own derivation, exactly the clip's
current `.start`/`.end`), so a blind sum happens to equal the union's
length in every real case this module produces today, but the union is
computed properly so a future provenance source that produced an
overlapping interval could never be silently double-counted.

## D-215 reuse, never a second classifier

`pacing_transition_decision.decide_transition` (D-215, unmodified) is
called TWICE per pair with identical relationship/Prosodic evidence and
ONLY the candidate lead/tail argument varied -- once with today's live
in-window-only evidence (`old_d215_mode`, reproducing exactly what
`pacing_v2_evidence_adapter.py`'s existing live call already computes),
once with the wider combined evidence (`new_handle_aware_diagnostic_
mode`). Neither call is ever written back to `draft.selected`, D-142's
own `pacing_stage`, or any `RenderSegment`.

## D-220 reuse, never a heuristic change

`pacing_v2_timing_policy.decide_jcut_timing`/`decide_lcut_timing` (D-220,
unmodified, same anchor-word-duration formula) is evaluated, purely as a
diagnostic curiosity, ONLY on a pair the wider evidence itself found
J_CUT/L_CUT-eligible (`new_handle_aware_diagnostic_mode`) -- to see
whether a wider `max_safe_window` input would let D-220's own existing
formula choose a non-zero amount. This never changes D-220's own anchor
choice or formula and is never consulted by anything live.

## Firewalls restated, not reopened

Video-window immutability (`clip.start`/`clip.end` are read-only
everywhere in this module -- no `dataclasses.replace` of a `DraftClip`
occurs here); handle eligibility gating (only `SAFE_NON_SPEECH_HANDLE`
extends availability -- D-223's own closed vocabulary, not broadened
here); D-223's own discarded/neighbor/retry/meaning firewalls are never
re-implemented, only consumed via `pacing_v2_source_audio_handle.build_
source_audio_handles` (D-223, unmodified).

## Live input source

Every input is already-computed, already-serialized real pipeline state,
reachable at the ONE real call site (`universal_clean_cut.py`, inside
the SAME `pacing_v2_diagnostics_enabled()` block D-216/D-217 already
gate on): `draft.diagnostics["boundary_engine_pass"]["audio_edge_rows"]`,
`draft.diagnostics["post_selection_edge_only_boundary"]`, `draft.
discarded`, and `request.sources[*].duration_sec` (the one place a real
full-source-file duration is reachable without a new probe, per D-222
item 8). `broader_word_timings` stays honestly empty at the live seam
(D-222 item 11's own confirmed gap: no per-source transcript survives to
`DraftTimeline`) -- accepted as an optional parameter for the same
future-extensibility/offline-fixture reasons D-223 already documented,
never populated by any live caller today.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from .contracts import DraftClip, Word
from .dialogue_pacing_transition import J_CUT, L_CUT
from .pacing_transition_decision import decide_transition
from .pacing_v2_evidence_adapter import (
    available_silent_head_sec,
    available_silent_tail_sec,
    build_pacing_v2_real_evidence,
)
from .pacing_v2_source_audio_handle import (
    DIRECTION_POST_ROLL,
    DIRECTION_PRE_ROLL,
    HANDLE_STATUS_SAFE_NON_SPEECH,
    HANDLE_STATUS_UNKNOWN_WORD_COVERAGE,
    SourceAudioHandle,
    build_source_audio_handles,
    source_audio_handle_diagnostics,
)
from .pacing_v2_timing_policy import decide_jcut_timing, decide_lcut_timing

SCHEMA_VERSION = "cutsell.pacing_v2_handle_aware_evidence.v1"

# --- combined-availability source vocabulary --------------------------------
COMBINED_SOURCE_IN_WINDOW_ONLY = "IN_WINDOW_ONLY"
COMBINED_SOURCE_HANDLE_ONLY = "HANDLE_ONLY"
COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE = "IN_WINDOW_PLUS_HANDLE"
COMBINED_SOURCE_NONE = "NONE"

# Only this D-223 status extends candidate availability -- this task's own
# explicit gate, never broadened.
_ELIGIBLE_HANDLE_STATUSES = frozenset({HANDLE_STATUS_SAFE_NON_SPEECH})

_EMPTY: Mapping = {}


def _merged_interval_length(intervals: Sequence[Tuple[float, float]]) -> float:
    """Real source-coordinate interval union -- never a blind duration
    sum. Sorts, merges touching (`start <= previous_end`) or overlapping
    spans, then sums the merged lengths. Proven directly (test 19) with a
    synthetic overlapping pair, independent of whether today's real
    derivation ever actually produces an overlap."""
    cleaned = sorted((float(s), float(e)) for (s, e) in intervals if e > s)
    if not cleaned:
        return 0.0
    merged = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged)


def _combined_source(in_window: float, handle: float) -> str:
    has_in_window = in_window > 0.0
    has_handle = handle > 0.0
    if has_in_window and has_handle:
        return COMBINED_SOURCE_IN_WINDOW_PLUS_HANDLE
    if has_handle:
        return COMBINED_SOURCE_HANDLE_ONLY
    if has_in_window:
        return COMBINED_SOURCE_IN_WINDOW_ONLY
    return COMBINED_SOURCE_NONE


def _eligible_handle_duration(handle: Optional[SourceAudioHandle]) -> float:
    """Zero for every handle status except `SAFE_NON_SPEECH_HANDLE` --
    this task's own explicit gate. A handle's own geometric `available_
    duration` is still reported in full in the diagnostics row regardless
    (so a reviewer can see WHY a blocked/unknown handle contributed
    nothing), but never counted toward `combined_*_available_window`."""
    if handle is None or handle.handle_status not in _ELIGIBLE_HANDLE_STATUSES:
        return 0.0
    return float(handle.available_duration)


def _combined_j_availability(
    old_in_window_head: Optional[float], right_start: float, pre_handle: Optional[SourceAudioHandle],
) -> Tuple[Optional[float], str]:
    handle_duration = _eligible_handle_duration(pre_handle)
    if old_in_window_head is None and handle_duration <= 0.0:
        return None, _combined_source(0.0, 0.0)
    intervals = []
    if old_in_window_head is not None and old_in_window_head > 0.0:
        intervals.append((right_start, right_start + old_in_window_head))
    if handle_duration > 0.0:
        intervals.append((pre_handle.handle_source_start, pre_handle.handle_source_end))
    combined = _merged_interval_length(intervals)
    source = _combined_source(old_in_window_head or 0.0, handle_duration)
    return combined, source


def _combined_l_availability(
    old_in_window_tail: Optional[float], left_end: float, post_handle: Optional[SourceAudioHandle],
) -> Tuple[Optional[float], str]:
    handle_duration = _eligible_handle_duration(post_handle)
    if old_in_window_tail is None and handle_duration <= 0.0:
        return None, _combined_source(0.0, 0.0)
    intervals = []
    if old_in_window_tail is not None and old_in_window_tail > 0.0:
        intervals.append((left_end - old_in_window_tail, left_end))
    if handle_duration > 0.0:
        intervals.append((post_handle.handle_source_start, post_handle.handle_source_end))
    combined = _merged_interval_length(intervals)
    source = _combined_source(old_in_window_tail or 0.0, handle_duration)
    return combined, source


def _handle_lookup(handles: Sequence[SourceAudioHandle]) -> Mapping[Tuple[str, str], SourceAudioHandle]:
    return {(h.owner_clip_id, h.direction): h for h in handles}


def build_handle_aware_pacing_v2_diagnostics(
    selected: Sequence[DraftClip],
    *,
    dialogue_overlap_enabled: bool,
    boundary_diagnostics: Optional[Mapping] = None,
    discarded: Sequence[DraftClip] = (),
    boundary_engine_pass_audit: Sequence[Mapping] = (),
    post_selection_edge_only_boundary_audit: Sequence[Mapping] = (),
    source_duration_by_asset: Optional[Mapping[str, float]] = None,
    broader_word_timings: Optional[Mapping[str, Sequence[Word]]] = None,
    editorial_moment_sequence_diagnostics: Optional[Mapping] = None,
    take_judge_groups: Sequence[Mapping] = (),
) -> dict:
    """The one D-224 entry point. Pure function of already-computed real
    pipeline state; builds no new evidence source of its own beyond
    D-223's own handle foundation and D-217's own relationship/Prosodic
    evidence (both reused verbatim, never duplicated)."""
    clips = tuple(selected)
    boundary_diagnostics = boundary_diagnostics or {}
    source_duration_by_asset = source_duration_by_asset or _EMPTY
    broader_word_timings = broader_word_timings or _EMPTY

    if len(clips) < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "transition_count": 0,
            "transitions": (),
            "handle_diagnostics": (),
            "run_summary": _run_summary((), ()),
        }

    handles = build_source_audio_handles(
        clips, discarded=discarded,
        boundary_engine_pass_audit=boundary_engine_pass_audit,
        post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
        source_duration_by_asset=source_duration_by_asset,
        broader_word_timings=broader_word_timings,
    )
    handle_by_owner_direction = _handle_lookup(handles)

    evidence = build_pacing_v2_real_evidence(
        clips,
        editorial_moment_sequence_diagnostics=editorial_moment_sequence_diagnostics,
        take_judge_groups=take_judge_groups,
    )
    relationship_hint_by_pair = evidence["relationship_hint_by_pair"]
    prosody_by_clip_id = evidence["prosody_by_clip_id"]

    rows = []
    for index in range(len(clips) - 1):
        left, right = clips[index], clips[index + 1]
        key = (left.clip_id, right.clip_id)
        relationship_hint = relationship_hint_by_pair.get(key)
        left_prosody = prosody_by_clip_id.get(left.clip_id)
        right_prosody = prosody_by_clip_id.get(right.clip_id)

        old_j_head = available_silent_head_sec(right)
        old_l_tail = available_silent_tail_sec(left)

        right_pre_handle = handle_by_owner_direction.get((right.clip_id, DIRECTION_PRE_ROLL))
        left_post_handle = handle_by_owner_direction.get((left.clip_id, DIRECTION_POST_ROLL))

        combined_j, combined_j_source = _combined_j_availability(old_j_head, float(right.start), right_pre_handle)
        combined_l, combined_l_source = _combined_l_availability(old_l_tail, float(left.end), left_post_handle)

        old_plan = decide_transition(
            left, right, dialogue_overlap_enabled=dialogue_overlap_enabled,
            boundary_diagnostics=boundary_diagnostics,
            left_prosody=left_prosody, right_prosody=right_prosody,
            candidate_audio_lead_sec=old_j_head, candidate_audio_tail_sec=old_l_tail,
            relationship_hint=relationship_hint, transition_index=index,
        )
        new_plan = decide_transition(
            left, right, dialogue_overlap_enabled=dialogue_overlap_enabled,
            boundary_diagnostics=boundary_diagnostics,
            left_prosody=left_prosody, right_prosody=right_prosody,
            candidate_audio_lead_sec=combined_j, candidate_audio_tail_sec=combined_l,
            relationship_hint=relationship_hint, transition_index=index,
        )

        j_timing_evaluated = new_plan.mode == J_CUT
        l_timing_evaluated = new_plan.mode == L_CUT
        j_timing = (
            decide_jcut_timing(left, right, max_safe_lead=combined_j, transition_index=index,
                                relationship_hint=relationship_hint, right_prosody=right_prosody)
            if j_timing_evaluated else None
        )
        l_timing = (
            decide_lcut_timing(left, right, max_safe_tail=combined_l, transition_index=index,
                                relationship_hint=relationship_hint, left_prosody=left_prosody)
            if l_timing_evaluated else None
        )

        rows.append({
            "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
            "old_in_window_j_head": old_j_head, "old_in_window_l_tail": old_l_tail,
            "right_pre_handle_id": right_pre_handle.handle_id if right_pre_handle else None,
            "right_pre_handle_status": right_pre_handle.handle_status if right_pre_handle else None,
            "right_pre_handle_start": right_pre_handle.handle_source_start if right_pre_handle else None,
            "right_pre_handle_end": right_pre_handle.handle_source_end if right_pre_handle else None,
            "right_pre_handle_duration": right_pre_handle.available_duration if right_pre_handle else None,
            "left_post_handle_id": left_post_handle.handle_id if left_post_handle else None,
            "left_post_handle_status": left_post_handle.handle_status if left_post_handle else None,
            "left_post_handle_start": left_post_handle.handle_source_start if left_post_handle else None,
            "left_post_handle_end": left_post_handle.handle_source_end if left_post_handle else None,
            "left_post_handle_duration": left_post_handle.available_duration if left_post_handle else None,
            "combined_j_available_window": combined_j, "combined_l_available_window": combined_l,
            "combined_j_source": combined_j_source, "combined_l_source": combined_l_source,
            "old_d215_mode": old_plan.mode, "new_handle_aware_diagnostic_mode": new_plan.mode,
            "d220_j_max_safe_window_evaluated": j_timing_evaluated,
            "d220_j_chosen_duration": j_timing.chosen_duration if j_timing is not None else None,
            "d220_l_max_safe_window_evaluated": l_timing_evaluated,
            "d220_l_chosen_duration": l_timing.chosen_duration if l_timing is not None else None,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(rows),
        "transitions": rows,
        "handle_diagnostics": tuple(source_audio_handle_diagnostics(h) for h in handles),
        "run_summary": _run_summary(rows, handles),
    }


def _run_summary(rows: Sequence[Mapping], handles: Sequence[SourceAudioHandle]) -> dict:
    """Counts only -- NO master/global score (this track's own binding
    D-220/D-223 convention)."""
    pre_handles = tuple(h for h in handles if h.direction == DIRECTION_PRE_ROLL)
    post_handles = tuple(h for h in handles if h.direction == DIRECTION_POST_ROLL)
    old_j = sum(1 for row in rows if row["old_d215_mode"] == J_CUT)
    old_l = sum(1 for row in rows if row["old_d215_mode"] == L_CUT)
    new_j = sum(1 for row in rows if row["new_handle_aware_diagnostic_mode"] == J_CUT)
    new_l = sum(1 for row in rows if row["new_handle_aware_diagnostic_mode"] == L_CUT)
    unlocked_j = sum(
        1 for row in rows
        if row["old_d215_mode"] != J_CUT and row["new_handle_aware_diagnostic_mode"] == J_CUT
    )
    unlocked_l = sum(
        1 for row in rows
        if row["old_d215_mode"] != L_CUT and row["new_handle_aware_diagnostic_mode"] == L_CUT
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(rows),
        "pre_handle_available_count": sum(1 for h in pre_handles if h.available_duration > 0.0),
        "post_handle_available_count": sum(1 for h in post_handles if h.available_duration > 0.0),
        "safe_pre_handle_count": sum(1 for h in pre_handles if h.handle_status == HANDLE_STATUS_SAFE_NON_SPEECH),
        "safe_post_handle_count": sum(1 for h in post_handles if h.handle_status == HANDLE_STATUS_SAFE_NON_SPEECH),
        "blocked_pre_handle_count": sum(1 for h in pre_handles if h.handle_status.startswith("BLOCKED_")),
        "blocked_post_handle_count": sum(1 for h in post_handles if h.handle_status.startswith("BLOCKED_")),
        "old_j_candidate_count": old_j,
        "old_l_candidate_count": old_l,
        "handle_aware_j_candidate_count": new_j,
        "handle_aware_l_candidate_count": new_l,
        "j_candidates_unlocked_by_handle_count": unlocked_j,
        "l_candidates_unlocked_by_handle_count": unlocked_l,
        "unknown_handle_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_UNKNOWN_WORD_COVERAGE),
    }
