"""D-233 -- Pacing V2 Audio Join Treatment Renderer/Timing Contract.
TIMING PLANNING HALF. OFFLINE ONLY.

Converts a D-232 `AudioJoinTreatmentDecision` into a concrete, bounded,
deterministic `AudioJoinTreatmentTimingPlan` -- the exact source/output
audio windows a (test-only, not-live-wired) renderer would need to
execute the recommended treatment. This module NEVER re-selects a
treatment (D-232's own closed authority, read verbatim from `decision.
treatment`) and NEVER executes anything itself -- see `render.py`'s own
new, additive `render_audio_join_treatment_preview` (this task's other
half) for the actual ffmpeg execution, which consumes exactly the plan
this module produces.

## Canonical position (restated, not renumbered)

    ... -> AUDIO JOIN TREATMENT (D-232, decision only)
        v
    TIMING PLANNING                <- this module
        v
    RENDERER EXECUTION              (render.py's new, additive,
                                     NOT-live-wired execution function)

## What this module explicitly does NOT do

Does not choose a different treatment than `decision.treatment` (D-232's
own closed authority) -- it may only report a timing FAILURE (`timing_
status` other than `SUPPORTED`) when the geometry cannot be safely
realized; the CALLER then falls back to `CLICK_FADE`/`NONE` exactly as
D-232's own "safe fallback" contract already describes (this module
never silently substitutes an advanced treatment of its own choosing).
Does not correct loudness, does not implement a room-tone classifier,
does not touch Boundary/Ordering/Family/BestTake, does not authorize
`MICRO_AUDIO_OVERLAP` (impossible by construction -- it is not a member
of `TREATMENT_VALUES`, D-232's own closed vocabulary, unmodified here).
Has no live wiring, no feature flag.

## Timing policy (D-098 Section 16.9's own "a future, SEPARATE
audio-treatment timing policy is named, never assumed to already
exist" -- this module IS that policy, for exactly the four advanced
treatments D-232 can recommend)

Mirrors D-220's own STRUCTURE (`chosen_duration = min(max_safe_window,
anchor)`, never an arbitrary duration alone) without reusing D-220's
own SPECIFIC anchor (a lexical word's own duration) -- Audio Join
Treatment durations apply only to NON-LEXICAL material by construction
(D-232's own firewalls already guarantee this), so no word-duration
anchor is available or appropriate here. `chosen_duration = min(all
available real safe windows, ONE new bounded, explicitly labeled
heuristic cap)`:

    AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC = 0.25

-- a single, isolated, directly tested constant (never a silent
default baked into a formula), named per this task's own required
`AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED`
convention. 250ms is a common editorial short-crossfade ballpark
(explicitly NOT claimed to be real-media tuned) applied uniformly to
every advanced treatment's own cap, so there is exactly one heuristic
constant in this whole module, never a second one per treatment.

- `SHORT_CROSSFADE`: `chosen_duration = min(left_safe_audio_window_sec,
  right_safe_audio_window_sec, CAP)`. Both safe windows are CALLER-
  supplied (the SAME already-computed `available_silent_tail_sec`/
  `available_silent_head_sec`-shaped quantities `pacing_v2_evidence_
  adapter.py` (D-217) already derives from each clip's own word
  geometry -- reused, never re-derived here) -- this satisfies "No
  SourceAudioHandle required by definition" (this task's own
  restatement of D-229's own finding) while still bounding the
  treatment to REAL, ALREADY-SAFE room inside each clip's own already-
  selected span; never extends past either clip's own `[start, end)`.
- `AMBIENCE_CARRY_LEFT`/`AMBIENCE_CARRY_RIGHT`: `chosen_duration =
  min(handle.available_duration, CAP)` -- the D-223 `SourceAudioHandle`
  (D-231/D-232's own already-validated safe-non-speech evidence) is the
  natural bound; the SAME heuristic cap is still applied (never a
  second, treatment-specific cap) so an unusually long safe handle
  never produces an unbounded ambience carry.
- `AMBIENCE_BRIDGE`: `chosen_duration = min(left_handle.available_
  duration, right_handle.available_duration, CAP)` -- both sides'
  material must be available, matching D-231's own `ambience_bridge_
  evidence_status` precondition (both sides `TREATMENT_EVIDENCE_READY`).
- `NONE`/`CLICK_FADE`: `chosen_duration = None`, `timing_status =
  TIMING_NOT_APPLICABLE` -- these introduce no new audio extension at
  all (this task's own explicit "NONE must introduce no new overlap...
  Preserve existing 12ms technical fade behavior exactly" instruction);
  there is nothing for a timing policy to plan.

## Source/output bounds (fail-closed, never silently clamped)

Every window this module returns satisfies `0 <= source_start < source_
end <= source_duration` for its own source asset (checked directly,
independently per side -- never a cross-source timestamp comparison,
per this task's own "do not compare raw timestamps across files"
instruction) and every OUTPUT-timeline position is `>= 0`. A violation
is reported via `timing_status` (`OUT_OF_BOUNDS`/`INSUFFICIENT_WINDOW`/
`ZERO_DURATION`), never silently repaired.

## One treatment per join, video-cut immutability

This module receives exactly one `decision.treatment` and returns
exactly one plan. `visual_join_time` is recorded for diagnostic/
verification purposes only -- this module never computes a video
position itself and has no parameter through which it could move one;
video-cut immutability is structural (no video-geometry mutation
parameter exists anywhere on this module's own public functions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .pacing_v2_audio_join_treatment_decision import (
    AudioJoinTreatmentDecision,
    TREATMENT_AMBIENCE_BRIDGE,
    TREATMENT_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT,
    TREATMENT_CLICK_FADE,
    TREATMENT_NONE,
    TREATMENT_SHORT_CROSSFADE,
    TREATMENT_STATUS_SUPPORTED,
)
from .pacing_v2_source_audio_handle import HANDLE_STATUS_SAFE_NON_SPEECH, SourceAudioHandle

SCHEMA_VERSION = "cutsell.pacing_v2_audio_join_treatment_timing.v1"

# ---------------------------------------------------------------------------
# Timing-status vocabulary (this task's own named seven).
# ---------------------------------------------------------------------------
TIMING_SUPPORTED = "SUPPORTED"
TIMING_ZERO_DURATION = "ZERO_DURATION"
TIMING_INSUFFICIENT_WINDOW = "INSUFFICIENT_WINDOW"
TIMING_OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
TIMING_INCOMPATIBLE_GEOMETRY = "INCOMPATIBLE_GEOMETRY"
TIMING_CONFLICTED = "CONFLICTED"
TIMING_UNKNOWN = "UNKNOWN"
TIMING_NOT_APPLICABLE = "NOT_APPLICABLE"  # NONE/CLICK_FADE -- no timing to plan.

# ---------------------------------------------------------------------------
# The ONE bounded, isolated, explicitly-labeled heuristic this module adds.
# ---------------------------------------------------------------------------
AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC = 0.25

# ---------------------------------------------------------------------------
# Renderer-capability status (D-233's own offline proof, distinct from
# D-232's own historical D-229 classification field -- see module
# docstring; D-232's own `renderer_capability_status()` is preserved
# unmodified as the historical record).
# ---------------------------------------------------------------------------
RENDERER_RESULT_SUPPORTED = "SUPPORTED"
RENDERER_RESULT_SUPPORTED_OFFLINE = "SUPPORTED_OFFLINE"
RENDERER_RESULT_BLOCKED = "BLOCKED"

# ---------------------------------------------------------------------------
# Conflict-flag vocabulary.
# ---------------------------------------------------------------------------
CONFLICT_NEGATIVE_OR_ZERO_SAFE_WINDOW = "negative_or_zero_safe_window"
CONFLICT_MISSING_SAFE_WINDOW_EVIDENCE = "missing_safe_window_evidence"
CONFLICT_HANDLE_NOT_SAFE_NON_SPEECH = "handle_not_safe_non_speech"
CONFLICT_HANDLE_MISSING = "handle_missing"
CONFLICT_HANDLE_ZERO_AVAILABLE_DURATION = "handle_zero_available_duration"
CONFLICT_SOURCE_BOUNDS_VIOLATION = "source_bounds_violation"
CONFLICT_CLIP_GEOMETRY_INVALID = "clip_geometry_invalid"
CONFLICT_TREATMENT_NOT_TIMED = "treatment_has_no_timing_plan"

PROVENANCE_DECISION_REUSED = "audio_join_treatment_decision_reused"
PROVENANCE_HANDLE_REUSED = "source_audio_handle_reused"
PROVENANCE_SAFE_WINDOW_CALLER_SUPPLIED = "caller_supplied_safe_audio_window"


@dataclass(frozen=True)
class AudioJoinTreatmentTimingPlan:
    """One adjacent join's concrete timing geometry for its OWN already-
    decided (D-232) treatment. Evidence/geometry only -- see module
    docstring for what this never decides or executes. No transcript
    dump, no master score."""

    schema_version: str
    transition_index: int
    left_clip_id: str
    right_clip_id: str
    left_source_asset_id: str
    right_source_asset_id: str

    treatment: str

    chosen_duration: Optional[float]

    left_source_audio_start: Optional[float]
    left_source_audio_end: Optional[float]
    right_source_audio_start: Optional[float]
    right_source_audio_end: Optional[float]

    left_output_audio_start: Optional[float]
    left_output_audio_end: Optional[float]
    right_output_audio_start: Optional[float]
    right_output_audio_end: Optional[float]

    visual_join_time: Optional[float]

    timing_status: str
    renderer_capability_status: str

    fallback_reason: Optional[str]

    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _validate_source_bounds(start: float, end: float, duration: float) -> bool:
    return 0.0 <= start < end <= duration + 1e-6


def _not_applicable_plan(decision: AudioJoinTreatmentDecision, **ids) -> AudioJoinTreatmentTimingPlan:
    return AudioJoinTreatmentTimingPlan(
        schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
        left_clip_id=decision.left_clip_id, right_clip_id=decision.right_clip_id,
        left_source_asset_id=ids.get("left_source_asset_id", ""), right_source_asset_id=ids.get("right_source_asset_id", ""),
        treatment=decision.treatment, chosen_duration=None,
        left_source_audio_start=None, left_source_audio_end=None,
        right_source_audio_start=None, right_source_audio_end=None,
        left_output_audio_start=None, left_output_audio_end=None,
        right_output_audio_start=None, right_output_audio_end=None,
        visual_join_time=ids.get("visual_join_time"),
        timing_status=TIMING_NOT_APPLICABLE, renderer_capability_status=RENDERER_RESULT_SUPPORTED,
        fallback_reason=None, conflict_flags=(), provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED),
    )


def _crossfade_plan(
    decision: AudioJoinTreatmentDecision, *,
    left_clip_id: str, right_clip_id: str, left_source_asset_id: str, right_source_asset_id: str,
    left_video_start: float, left_video_end: float, left_source_duration: float,
    right_video_start: float, right_video_end: float, right_source_duration: float,
    left_safe_audio_window_sec: Optional[float], right_safe_audio_window_sec: Optional[float],
) -> AudioJoinTreatmentTimingPlan:
    conflict_flags: list = []
    if left_video_end <= left_video_start or right_video_end <= right_video_start:
        return _conflicted_plan(decision, left_source_asset_id, right_source_asset_id, CONFLICT_CLIP_GEOMETRY_INVALID)
    if left_safe_audio_window_sec is None or right_safe_audio_window_sec is None:
        conflict_flags.append(CONFLICT_MISSING_SAFE_WINDOW_EVIDENCE)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_UNKNOWN, conflict_flags=conflict_flags, fallback_reason="safe_window_evidence_missing",
        )
    if left_safe_audio_window_sec <= 0.0 or right_safe_audio_window_sec <= 0.0:
        conflict_flags.append(CONFLICT_NEGATIVE_OR_ZERO_SAFE_WINDOW)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_INSUFFICIENT_WINDOW, conflict_flags=conflict_flags, fallback_reason="no_safe_crossfade_window",
        )

    cap = AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC
    chosen = min(left_safe_audio_window_sec, right_safe_audio_window_sec, cap)
    if chosen <= 0.0:
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_ZERO_DURATION, conflict_flags=conflict_flags, fallback_reason="zero_duration_after_bounding",
        )

    left_src_start = left_video_end - chosen
    left_src_end = left_video_end
    right_src_start = right_video_start
    right_src_end = right_video_start + chosen

    if not _validate_source_bounds(left_src_start, left_src_end, left_source_duration) or \
            not _validate_source_bounds(right_src_start, right_src_end, right_source_duration):
        conflict_flags.append(CONFLICT_SOURCE_BOUNDS_VIOLATION)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_OUT_OF_BOUNDS, conflict_flags=conflict_flags, fallback_reason="source_window_out_of_bounds",
        )

    return AudioJoinTreatmentTimingPlan(
        schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
        left_clip_id=left_clip_id, right_clip_id=right_clip_id,
        left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        treatment=TREATMENT_SHORT_CROSSFADE, chosen_duration=chosen,
        left_source_audio_start=left_src_start, left_source_audio_end=left_src_end,
        right_source_audio_start=right_src_start, right_source_audio_end=right_src_end,
        # Symmetric crossfade window: both slices occupy the SAME output
        # window [0, chosen], overlapping fully -- the visual join sits at
        # the window's own midpoint (see module docstring / render.py's
        # own execution counterpart).
        left_output_audio_start=0.0, left_output_audio_end=chosen,
        right_output_audio_start=0.0, right_output_audio_end=chosen,
        visual_join_time=chosen / 2.0,
        timing_status=TIMING_SUPPORTED, renderer_capability_status=RENDERER_RESULT_SUPPORTED_OFFLINE,
        fallback_reason=None, conflict_flags=tuple(conflict_flags),
        provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED, PROVENANCE_SAFE_WINDOW_CALLER_SUPPLIED),
    )


def _ambience_carry_plan(
    decision: AudioJoinTreatmentDecision, *, side: str,
    left_clip_id: str, right_clip_id: str, left_source_asset_id: str, right_source_asset_id: str,
    left_video_start: float, left_video_end: float,
    right_video_start: float, right_video_end: float,
    handle: Optional[SourceAudioHandle], source_duration: float,
) -> AudioJoinTreatmentTimingPlan:
    conflict_flags: list = []
    if handle is None:
        conflict_flags.append(CONFLICT_HANDLE_MISSING)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_UNKNOWN, conflict_flags=conflict_flags, fallback_reason="handle_missing",
        )
    if handle.handle_status != HANDLE_STATUS_SAFE_NON_SPEECH:
        conflict_flags.append(CONFLICT_HANDLE_NOT_SAFE_NON_SPEECH)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_CONFLICTED, conflict_flags=conflict_flags, fallback_reason="handle_not_safe_non_speech",
        )
    if handle.available_duration <= 0.0:
        conflict_flags.append(CONFLICT_HANDLE_ZERO_AVAILABLE_DURATION)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_INSUFFICIENT_WINDOW, conflict_flags=conflict_flags, fallback_reason="handle_zero_duration",
        )

    cap = AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC
    chosen = min(float(handle.available_duration), cap)
    if chosen <= 0.0:
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_ZERO_DURATION, conflict_flags=conflict_flags, fallback_reason="zero_duration_after_bounding",
        )

    src_start = float(handle.handle_source_start)
    src_end = min(float(handle.handle_source_end), src_start + chosen)
    if not _validate_source_bounds(src_start, src_end, source_duration):
        conflict_flags.append(CONFLICT_SOURCE_BOUNDS_VIOLATION)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_OUT_OF_BOUNDS, conflict_flags=conflict_flags, fallback_reason="handle_window_out_of_bounds",
        )

    if side == "LEFT":
        # Carries LEFT's own trailing safe material past the join --
        # occupies output [0, chosen] (join-local frame, join at t=0),
        # overlapping RIGHT's own unmodified opening audio.
        return AudioJoinTreatmentTimingPlan(
            schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
            left_clip_id=left_clip_id, right_clip_id=right_clip_id,
            left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
            treatment=TREATMENT_AMBIENCE_CARRY_LEFT, chosen_duration=chosen,
            left_source_audio_start=src_start, left_source_audio_end=src_end,
            right_source_audio_start=None, right_source_audio_end=None,
            left_output_audio_start=0.0, left_output_audio_end=chosen,
            right_output_audio_start=None, right_output_audio_end=None,
            visual_join_time=0.0,
            timing_status=TIMING_SUPPORTED, renderer_capability_status=RENDERER_RESULT_SUPPORTED_OFFLINE,
            fallback_reason=None, conflict_flags=tuple(conflict_flags),
            provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED, PROVENANCE_HANDLE_REUSED),
        )
    # RIGHT: carries RIGHT's own leading safe material before the join --
    # occupies output [-chosen, 0] relative to the join; reported here in
    # a zero-based local frame as [0, chosen] with the join at t=chosen.
    return AudioJoinTreatmentTimingPlan(
        schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
        left_clip_id=left_clip_id, right_clip_id=right_clip_id,
        left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        treatment=TREATMENT_AMBIENCE_CARRY_RIGHT, chosen_duration=chosen,
        left_source_audio_start=None, left_source_audio_end=None,
        right_source_audio_start=src_start, right_source_audio_end=src_end,
        left_output_audio_start=None, left_output_audio_end=None,
        right_output_audio_start=0.0, right_output_audio_end=chosen,
        visual_join_time=chosen,
        timing_status=TIMING_SUPPORTED, renderer_capability_status=RENDERER_RESULT_SUPPORTED_OFFLINE,
        fallback_reason=None, conflict_flags=tuple(conflict_flags),
        provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED, PROVENANCE_HANDLE_REUSED),
    )


def _ambience_bridge_plan(
    decision: AudioJoinTreatmentDecision, *,
    left_clip_id: str, right_clip_id: str, left_source_asset_id: str, right_source_asset_id: str,
    left_handle: Optional[SourceAudioHandle], right_handle: Optional[SourceAudioHandle],
    left_source_duration: float, right_source_duration: float,
) -> AudioJoinTreatmentTimingPlan:
    conflict_flags: list = []
    for handle in (left_handle, right_handle):
        if handle is None:
            conflict_flags.append(CONFLICT_HANDLE_MISSING)
        elif handle.handle_status != HANDLE_STATUS_SAFE_NON_SPEECH:
            conflict_flags.append(CONFLICT_HANDLE_NOT_SAFE_NON_SPEECH)
    if conflict_flags:
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_CONFLICTED, conflict_flags=conflict_flags, fallback_reason="bridge_handle_evidence_incomplete",
        )
    if left_handle.available_duration <= 0.0 or right_handle.available_duration <= 0.0:
        conflict_flags.append(CONFLICT_HANDLE_ZERO_AVAILABLE_DURATION)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_INSUFFICIENT_WINDOW, conflict_flags=conflict_flags, fallback_reason="bridge_handle_zero_duration",
        )

    cap = AUDIO_JOIN_TREATMENT_TIMING_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_MAX_DURATION_SEC
    chosen = min(float(left_handle.available_duration), float(right_handle.available_duration), cap)
    if chosen <= 0.0:
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_ZERO_DURATION, conflict_flags=conflict_flags, fallback_reason="zero_duration_after_bounding",
        )

    left_src_start = float(left_handle.handle_source_start)
    left_src_end = min(float(left_handle.handle_source_end), left_src_start + chosen)
    right_src_start = float(right_handle.handle_source_start)
    right_src_end = min(float(right_handle.handle_source_end), right_src_start + chosen)
    if not _validate_source_bounds(left_src_start, left_src_end, left_source_duration) or \
            not _validate_source_bounds(right_src_start, right_src_end, right_source_duration):
        conflict_flags.append(CONFLICT_SOURCE_BOUNDS_VIOLATION)
        return _bounded_plan(
            decision, left_clip_id, right_clip_id, left_source_asset_id, right_source_asset_id,
            timing_status=TIMING_OUT_OF_BOUNDS, conflict_flags=conflict_flags, fallback_reason="bridge_window_out_of_bounds",
        )

    # Both sides overlap the SAME output window (both diverging around
    # the join, the shared geometric shape MICRO_AUDIO_OVERLAP would also
    # use -- CONTENT ROLE, not geometry, is what makes this AMBIENCE_
    # BRIDGE, per D-098/D-232's own restated distinction).
    return AudioJoinTreatmentTimingPlan(
        schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
        left_clip_id=left_clip_id, right_clip_id=right_clip_id,
        left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        treatment=TREATMENT_AMBIENCE_BRIDGE, chosen_duration=chosen,
        left_source_audio_start=left_src_start, left_source_audio_end=left_src_end,
        right_source_audio_start=right_src_start, right_source_audio_end=right_src_end,
        left_output_audio_start=0.0, left_output_audio_end=chosen,
        right_output_audio_start=0.0, right_output_audio_end=chosen,
        visual_join_time=chosen / 2.0,
        timing_status=TIMING_SUPPORTED, renderer_capability_status=RENDERER_RESULT_SUPPORTED_OFFLINE,
        fallback_reason=None, conflict_flags=tuple(conflict_flags),
        provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED, PROVENANCE_HANDLE_REUSED),
    )


def _bounded_plan(
    decision: AudioJoinTreatmentDecision, left_clip_id: str, right_clip_id: str,
    left_source_asset_id: str, right_source_asset_id: str, *,
    timing_status: str, conflict_flags: Sequence[str], fallback_reason: Optional[str],
) -> AudioJoinTreatmentTimingPlan:
    return AudioJoinTreatmentTimingPlan(
        schema_version=SCHEMA_VERSION, transition_index=decision.transition_index,
        left_clip_id=left_clip_id, right_clip_id=right_clip_id,
        left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        treatment=decision.treatment, chosen_duration=None,
        left_source_audio_start=None, left_source_audio_end=None,
        right_source_audio_start=None, right_source_audio_end=None,
        left_output_audio_start=None, left_output_audio_end=None,
        right_output_audio_start=None, right_output_audio_end=None,
        visual_join_time=None,
        timing_status=timing_status, renderer_capability_status=RENDERER_RESULT_BLOCKED,
        fallback_reason=fallback_reason, conflict_flags=tuple(conflict_flags),
        provenance=(SCHEMA_VERSION, PROVENANCE_DECISION_REUSED),
    )


def _conflicted_plan(decision, left_source_asset_id, right_source_asset_id, reason) -> AudioJoinTreatmentTimingPlan:
    return _bounded_plan(
        decision, decision.left_clip_id, decision.right_clip_id, left_source_asset_id, right_source_asset_id,
        timing_status=TIMING_INCOMPATIBLE_GEOMETRY, conflict_flags=(reason,), fallback_reason=reason,
    )


def build_audio_join_treatment_timing_plan(
    decision: AudioJoinTreatmentDecision,
    *,
    left_source_asset_id: str,
    right_source_asset_id: str,
    left_video_start: float = 0.0,
    left_video_end: float = 0.0,
    left_source_duration: float = 0.0,
    right_video_start: float = 0.0,
    right_video_end: float = 0.0,
    right_source_duration: float = 0.0,
    left_safe_audio_window_sec: Optional[float] = None,
    right_safe_audio_window_sec: Optional[float] = None,
    left_post_roll_handle: Optional[SourceAudioHandle] = None,
    right_pre_roll_handle: Optional[SourceAudioHandle] = None,
) -> AudioJoinTreatmentTimingPlan:
    """The one D-233 timing-planning entry point. `decision.treatment` is
    read-only (D-232's own closed authority) -- this function NEVER
    reassigns it to a different advanced treatment; it may only report a
    timing failure (see module docstring)."""
    if decision.treatment in (TREATMENT_NONE, TREATMENT_CLICK_FADE):
        return _not_applicable_plan(
            decision, left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        )
    if decision.treatment == TREATMENT_SHORT_CROSSFADE:
        return _crossfade_plan(
            decision, left_clip_id=decision.left_clip_id, right_clip_id=decision.right_clip_id,
            left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
            left_video_start=left_video_start, left_video_end=left_video_end, left_source_duration=left_source_duration,
            right_video_start=right_video_start, right_video_end=right_video_end, right_source_duration=right_source_duration,
            left_safe_audio_window_sec=left_safe_audio_window_sec, right_safe_audio_window_sec=right_safe_audio_window_sec,
        )
    if decision.treatment == TREATMENT_AMBIENCE_CARRY_LEFT:
        return _ambience_carry_plan(
            decision, side="LEFT", left_clip_id=decision.left_clip_id, right_clip_id=decision.right_clip_id,
            left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
            left_video_start=left_video_start, left_video_end=left_video_end,
            right_video_start=right_video_start, right_video_end=right_video_end,
            handle=left_post_roll_handle, source_duration=left_source_duration,
        )
    if decision.treatment == TREATMENT_AMBIENCE_CARRY_RIGHT:
        return _ambience_carry_plan(
            decision, side="RIGHT", left_clip_id=decision.left_clip_id, right_clip_id=decision.right_clip_id,
            left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
            left_video_start=left_video_start, left_video_end=left_video_end,
            right_video_start=right_video_start, right_video_end=right_video_end,
            handle=right_pre_roll_handle, source_duration=right_source_duration,
        )
    if decision.treatment == TREATMENT_AMBIENCE_BRIDGE:
        return _ambience_bridge_plan(
            decision, left_clip_id=decision.left_clip_id, right_clip_id=decision.right_clip_id,
            left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
            left_handle=left_post_roll_handle, right_handle=right_pre_roll_handle,
            left_source_duration=left_source_duration, right_source_duration=right_source_duration,
        )
    # decision.treatment is not a recognized member of TREATMENT_VALUES
    # (defensive only -- D-232's own vocabulary is closed).
    return _bounded_plan(
        decision, decision.left_clip_id, decision.right_clip_id, left_source_asset_id, right_source_asset_id,
        timing_status=TIMING_UNKNOWN, conflict_flags=(CONFLICT_TREATMENT_NOT_TIMED,), fallback_reason="unrecognized_treatment",
    )


def audio_join_treatment_timing_diagnostics(plan: AudioJoinTreatmentTimingPlan) -> dict:
    """Compact, JSON-safe diagnostics row -- no transcript dump."""
    return {
        "schema_version": plan.schema_version,
        "transition_index": plan.transition_index,
        "left_clip_id": plan.left_clip_id, "right_clip_id": plan.right_clip_id,
        "left_source_asset_id": plan.left_source_asset_id, "right_source_asset_id": plan.right_source_asset_id,
        "treatment": plan.treatment, "chosen_duration": plan.chosen_duration,
        "left_source_audio_start": plan.left_source_audio_start, "left_source_audio_end": plan.left_source_audio_end,
        "right_source_audio_start": plan.right_source_audio_start, "right_source_audio_end": plan.right_source_audio_end,
        "left_output_audio_start": plan.left_output_audio_start, "left_output_audio_end": plan.left_output_audio_end,
        "right_output_audio_start": plan.right_output_audio_start, "right_output_audio_end": plan.right_output_audio_end,
        "visual_join_time": plan.visual_join_time,
        "timing_status": plan.timing_status, "renderer_capability_status": plan.renderer_capability_status,
        "fallback_reason": plan.fallback_reason,
        "conflict_flags": list(plan.conflict_flags), "provenance": list(plan.provenance),
    }


def audio_join_treatment_timing_run_summary(rows: Sequence[AudioJoinTreatmentTimingPlan]) -> dict:
    """No master score -- plain counts only."""
    def _count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    return {
        "schema_version": SCHEMA_VERSION,
        "plan_count": len(rows),
        "supported_count": _count(lambda r: r.timing_status == TIMING_SUPPORTED),
        "not_applicable_count": _count(lambda r: r.timing_status == TIMING_NOT_APPLICABLE),
        "zero_duration_count": _count(lambda r: r.timing_status == TIMING_ZERO_DURATION),
        "insufficient_window_count": _count(lambda r: r.timing_status == TIMING_INSUFFICIENT_WINDOW),
        "out_of_bounds_count": _count(lambda r: r.timing_status == TIMING_OUT_OF_BOUNDS),
        "incompatible_geometry_count": _count(lambda r: r.timing_status == TIMING_INCOMPATIBLE_GEOMETRY),
        "conflicted_count": _count(lambda r: r.timing_status == TIMING_CONFLICTED),
        "unknown_count": _count(lambda r: r.timing_status == TIMING_UNKNOWN),
    }
