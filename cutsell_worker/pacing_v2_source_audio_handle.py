"""D-223 -- PACING V2 SOURCE AUDIO HANDLE FOUNDATION. OFFLINE ONLY.

Closes the gap D-222's own forensic confirmed by direct code inspection:
`pacing_v2_evidence_adapter.py`'s `available_silent_head_sec`/
`available_silent_tail_sec` (D-217, unmodified by this module) measure J/L
candidate room STRICTLY INSIDE a clip's own already-Boundary-finalized
`[start, end)` window -- even though at least two Boundary-adjacent
authorities already prove, via their OWN already-computed audit trail, that
real safe (non-speech) source audio exists immediately OUTSIDE that window,
in the same physical source file, without any new ASR/provider call:

  * `boundary_engine_pass.tighten_selected_audio_edges` (D-097.C/E) --
    trims a clip's own audio-silence-event-confirmed leading/trailing edge,
    recording `original_start`/`original_end` (pre-trim) alongside
    `result_start`/`result_end` (post-trim, == the clip's own current
    `.start`/`.end`) in `draft.diagnostics["boundary_engine_pass"]`.
  * `post_selection_edge_only_boundary.trim_locked_selection_edges` --
    the alternate (`boundary_owner != "post_freeze"`) edge-only trim,
    recording the identical `original_start`/`original_end` vs
    `result_start`/`result_end` shape, gated on named recording-process/
    retry event evidence (see `_RETRY_OR_BTS_EVIDENCE_KINDS` below).

This module reads ONLY those two already-computed, already-serialized audit
trails (never recomputes a trim, never reopens Boundary, never mutates
`clip.start`/`clip.end`) and derives a `SourceAudioHandle` per clip per
direction: the room ALREADY PROVEN, by an accepted Boundary authority's own
prior decision, to be either (a) real, measured audio silence, or (b)
material that authority already excised as recording-process/retry debris
-- never a new, speculative reach into unprobed source media.

## Scope decision required by this task: (A) bound to original candidate
## span vs (B) extend into the full physical source file

**Chosen: (A), bound strictly to each clip's own already-recorded
"original" Boundary-audit span. Justified, not guessed, from current
architecture:**

D-222 (item 6/13-14) proved the RENDERER (`render.py`'s
`validate_audio_window`) and the DECISION layer (`pacing_transition_
decision.decide_transition`'s unused `left_words`/`right_words`
parameter) already support option (B) in principle -- reaching anywhere
within the full physical source file. But D-222 (item 11) also proved,
by direct inspection, that the WORD-TIMING EVIDENCE option (B) would
need to safely justify such a reach ("broader word evidence [that]
safely supports it") DOES NOT EXIST reachable from `DraftTimeline`
today: the full per-source-asset transcript is dropped between `take_
segmentation.py` slicing and `DraftTimeline` construction, and this
task is explicitly scoped to "minimum identity/provenance helpers" --
not a new full-transcript retention seam, not a new acoustic silence/
room-tone detector (D-222 item 18: none exists), and not a pipeline
reordering. Option (B)'s own precondition is therefore unsatisfiable
within this task's authorized scope. Option (A) uses ONLY evidence two
existing, accepted authorities have ALREADY computed and already
proved safe by their own accepted decision -- no new detector, no new
provider call, no reach into media nothing has ever measured. This is
the safer canonical rule current architecture actually supports today.

## Word/meaning safety -- reused, never duplicated

Every handle this module derives from `boundary_engine_pass`'s own
audio-silence trim is, BY CONSTRUCTION of that authority's own code (its
`candidate = max(candidate, last_word_end)` / `min(candidate, first_
word_start)` clamps), proven never to cross a real spoken word -- a
structural guarantee, not a belief. `post_selection_edge_only_boundary`'s
own trim carries the identical word-boundary clamp AND an explicit,
already-computed EVENT-KIND justification for why the trimmed room was
never part of the delivery -- this module reads that already-computed
kind (never a new detector) to distinguish genuinely inert dead air
(`unintentional_dead_air`) from confirmed recording-process/retry
debris (`retry_setup`, `false_start`, `wrong_take`, `breaking_
character`, `camera_adjustment`, `searching_for_words`, and the four
`*_reset_candidate` kinds) -- the latter is BLOCKED outright, restating
CLAUDE.md's own binding "Remove real failed/retry/BTS material" rule,
never reused as a J/L/ambience source merely because it lacks an aligned
word.

For the (today unreachable, but architecturally supported for future
extensibility and offline testing) case where a caller supplies broader
per-source-asset word timings (`broader_word_timings`, empty by default,
POPULATED BY NO LIVE CALLER TODAY -- D-222 item 11's own confirmed gap),
any word found inside the handle window is run through `semantic_claims.
classify_claim` (D-038, reused verbatim, never duplicated) -- CRITICAL
content BLOCKS the handle outright.

## Why an 8th handle-status value beyond this task's own named 7

This task's own directive names `SAFE_SPEECH_HANDLE` as a value a handle
MAY reach, while also requiring, in the same breath: "presence of words
alone must NOT authorize reuse; if architecture cannot prove safe speech
reuse, keep the handle non-authoritative." No mechanism in this
codebase proves speech-reuse safety at the SOURCE-AUDIO level -- that
proof (`_word_safety`, double-speech overlap safety) is `pacing_
transition_decision.decide_transition`'s (D-215's) own authority,
exercised on a WHOLE TRANSITION's candidate window, never duplicated
here. So this module defines `SAFE_SPEECH_HANDLE` in its own vocabulary
(for a FUTURE gate that wires a real proof through this foundation) but
never itself assigns it -- a handle with proven word presence and
non-critical meaning is `SPEECH_PRESENT_NOT_AUTHORITATIVE`: honestly
representable, never treated as safe-to-use. `test_no_builder_ever_
assigns_safe_speech_handle` proves this by construction.

## What this module explicitly does NOT do

Does not choose J_CUT/L_CUT/timing amount (`pacing_v2_timing_policy.py`,
D-220, untouched); does not wire anything into `RenderSegment`/`render.
py`/`render_plan.py` (both untouched, cited only in this docstring as
already-sufficient per D-222); does not reorder any pipeline stage;
does not change Boundary/Ordering/Family/BestTake behavior (both audit
trails are READ, never re-executed or altered); does not implement any
Audio Join Treatment (`SHORT_CROSSFADE`/`AMBIENCE_*`, D-220C Section
16.5, still `NOT_IMPLEMENTED`/`NO_AUTHORITY`); does not invent a new
acoustic speech/room-tone detector (D-222 item 18: none exists in this
codebase; none is added here). Zero live authority, zero RAW, zero
feature flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from .contracts import DraftClip, Word
from .semantic_claims import CRITICAL, classify_claim

SCHEMA_VERSION = "cutsell.pacing_v2_source_audio_handle.v1"

# --- direction vocabulary ---------------------------------------------------
DIRECTION_PRE_ROLL = "PRE_ROLL"
DIRECTION_POST_ROLL = "POST_ROLL"

# --- provenance-authority vocabulary (names of already-computed, already-
# accepted audit sources this module reads -- it recomputes none of them) ---
PROVENANCE_BOUNDARY_ENGINE_PASS_AUDIO_EDGE = "boundary_engine_pass.tighten_selected_audio_edges"
PROVENANCE_POST_SELECTION_EDGE_ONLY_BOUNDARY = "post_selection_edge_only_boundary.trim_locked_selection_edges"
PROVENANCE_NO_TRIM_RECORDED = "NO_BOUNDARY_PROVENANCE_RECORDED"
PROVENANCE_STALE_MISMATCH = "BOUNDARY_PROVENANCE_STALE_RESULT_MISMATCH"

# --- speech-presence vocabulary (minimum 3-value, per this task's own rule --
# no new acoustic speech detector is invented; NO_WORDS_PRESENT here is a
# STRUCTURAL PROOF from the owning Boundary authority's own word-boundary
# clamp, not merely "we did not look") ---------------------------------------
SPEECH_PRESENCE_NO_WORDS = "NO_WORDS_PRESENT"
SPEECH_PRESENCE_WORDS_PRESENT = "WORDS_PRESENT"
SPEECH_PRESENCE_UNKNOWN = "WORD_COVERAGE_UNKNOWN"

# --- discarded/neighbor-overlap vocabulary -----------------------------------
OVERLAP_NONE = "NO_OVERLAP"
OVERLAP_DISCARDED = "OVERLAPS_DISCARDED_MATERIAL"
OVERLAP_NEIGHBOR_SELECTED = "OVERLAPS_NEIGHBOR_SELECTED_CLIP"
OVERLAP_NOT_APPLICABLE = "NOT_APPLICABLE_NO_HANDLE_WINDOW"

# --- meaning-safety vocabulary (D-038's classify_claim, reused verbatim) ----
MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS = "NOT_APPLICABLE_NO_WORDS"
MEANING_SAFETY_SAFE = "SAFE"
MEANING_SAFETY_BLOCKED_CRITICAL = "BLOCKED_MEANING_CRITICAL"
MEANING_SAFETY_UNKNOWN = "UNKNOWN"

# --- closed, READ-ONLY classification of `post_selection_edge_only_boundary`'s
# own already-named event kinds (copied verbatim from that module's private
# `_AUTHORITATIVE`/`_RESET` sets -- this module invents no new kind, it only
# separates the one purely-inert kind from every recording-process/retry kind
# already named there) --------------------------------------------------------
_SAFE_NON_SPEECH_EVIDENCE_KINDS = frozenset({"unintentional_dead_air"})
_RETRY_OR_BTS_EVIDENCE_KINDS = frozenset({
    "retry_setup", "searching_for_words", "false_start", "wrong_take",
    "breaking_character", "camera_adjustment",
    "body_reset_candidate", "hand_motion_reset_candidate",
    "camera_disengagement_candidate", "facial_expression_shift_candidate",
})

# --- handle-status vocabulary (this task's own 7 + one justified addition,
# see module docstring "Why an 8th handle-status value") --------------------
HANDLE_STATUS_SAFE_NON_SPEECH = "SAFE_NON_SPEECH_HANDLE"
HANDLE_STATUS_SAFE_SPEECH = "SAFE_SPEECH_HANDLE"
HANDLE_STATUS_BLOCKED_DISCARDED = "BLOCKED_DISCARDED_MATERIAL"
HANDLE_STATUS_BLOCKED_MEANING_CRITICAL = "BLOCKED_MEANING_CRITICAL"
HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION = "BLOCKED_RETRY_OR_CORRECTION"
HANDLE_STATUS_UNKNOWN_WORD_COVERAGE = "UNKNOWN_WORD_COVERAGE"
HANDLE_STATUS_UNAVAILABLE = "UNAVAILABLE"
HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE = "SPEECH_PRESENT_NOT_AUTHORITATIVE"
HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP = "BLOCKED_NEIGHBOR_SELECTED_CLIP"

CLOSED_HANDLE_STATUSES = frozenset({
    HANDLE_STATUS_SAFE_NON_SPEECH, HANDLE_STATUS_SAFE_SPEECH,
    HANDLE_STATUS_BLOCKED_DISCARDED, HANDLE_STATUS_BLOCKED_MEANING_CRITICAL,
    HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION, HANDLE_STATUS_UNKNOWN_WORD_COVERAGE,
    HANDLE_STATUS_UNAVAILABLE, HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE,
    HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP,
})

# --- conflict-flag vocabulary ------------------------------------------------
CONFLICT_INVALID_GEOMETRY = "invalid_geometry"
CONFLICT_ZERO_OR_NEGATIVE_DURATION = "zero_or_negative_duration"
CONFLICT_EXCEEDS_SOURCE_DURATION = "exceeds_source_duration"
CONFLICT_OVERLAPS_DISCARDED = "overlaps_discarded_material"
CONFLICT_OVERLAPS_NEIGHBOR_SELECTED = "overlaps_neighbor_selected_clip"
CONFLICT_RETRY_OR_BTS_EVIDENCE = "retry_or_bts_evidence_at_edge"
CONFLICT_MEANING_CRITICAL = "meaning_critical_content_present"
CONFLICT_WORD_COVERAGE_UNKNOWN = "word_coverage_unknown"
CONFLICT_NO_PROVENANCE = "no_boundary_provenance_recorded"
CONFLICT_STALE_PROVENANCE = "boundary_provenance_stale_result_mismatch"

_RESULT_MATCH_EPSILON_SEC = 1e-3
_EMPTY_MAPPING: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class SourceAudioHandle:
    """One directional (PRE_ROLL/POST_ROLL) candidate audio-handle window
    for one clip, entirely offline -- never wired into `RenderSegment`,
    never authorizing J/L timing or Audio Join Treatment on its own. See
    module docstring for the full derivation/safety contract."""

    schema_version: str
    handle_id: str
    source_asset_id: str
    owner_clip_id: str
    owner_realization_id: Optional[str]
    direction: str
    video_start: float
    video_end: float
    handle_source_start: float
    handle_source_end: float
    available_duration: float
    word_intervals_present: Tuple[Tuple[float, float, str], ...]
    speech_presence_status: str
    discarded_overlap_status: str
    meaning_safety_status: str
    handle_status: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _handle_id(source_asset_id: str, owner_clip_id: str, direction: str, start: float, end: float) -> str:
    """Stable, deterministic -- derived purely from identity/geometry
    inputs, never a random UUID (required by this task)."""
    return f"handle:{source_asset_id}:{owner_clip_id}:{direction}:{round(float(start), 3)}:{round(float(end), 3)}"


def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _audit_rows_for_clip(audit_rows: Sequence[Mapping], clip_id: str) -> Tuple[Mapping, ...]:
    return tuple(row for row in (audit_rows or ()) if row.get("clip_id") == clip_id)


def _evidence_kinds_from_actions(actions: Sequence[Mapping]) -> Tuple[str, ...]:
    """Parses the already-computed `"event:<kind>:<confidence>"` /
    `"reset:<kind>:<confidence>"` evidence strings `post_selection_edge_
    only_boundary`'s own audit already carries (see that module's `_edge_
    evidence`) -- never invents a new kind, only reads the one already
    recorded."""
    kinds: list = []
    for action in actions or ():
        for item in action.get("evidence") or ():
            parts = str(item).split(":")
            if len(parts) >= 2:
                kinds.append(parts[1])
    return tuple(kinds)


def _widen_bound(
    clip: DraftClip,
    direction: str,
    *,
    boundary_engine_pass_audit: Sequence[Mapping],
    post_selection_edge_only_boundary_audit: Sequence[Mapping],
) -> dict:
    """Looks up whichever already-computed Boundary-adjacent audit trail
    recorded a trim for this clip, and returns the widened bound it
    already proved safe (or `available=False` when no such record
    exists -- never a guessed/default window)."""
    rows_a = _audit_rows_for_clip(boundary_engine_pass_audit, clip.clip_id)
    rows_b = _audit_rows_for_clip(post_selection_edge_only_boundary_audit, clip.clip_id)

    best: Optional[dict] = None
    for rows, provenance, is_authoritative_kind_gated in (
        (rows_a, PROVENANCE_BOUNDARY_ENGINE_PASS_AUDIO_EDGE, False),
        (rows_b, PROVENANCE_POST_SELECTION_EDGE_ONLY_BOUNDARY, True),
    ):
        for row in rows:
            original_start = float(row.get("original_start", clip.start))
            original_end = float(row.get("original_end", clip.end))
            result_start = float(row.get("result_start", clip.start))
            result_end = float(row.get("result_end", clip.end))
            if direction == DIRECTION_PRE_ROLL:
                stale = abs(result_start - float(clip.start)) > _RESULT_MATCH_EPSILON_SEC
                widened = original_start < result_start
                candidate_start, candidate_end = original_start, result_start
            else:
                stale = abs(result_end - float(clip.end)) > _RESULT_MATCH_EPSILON_SEC
                widened = original_end > result_end
                candidate_start, candidate_end = result_end, original_end
            if stale:
                if best is None:
                    best = {"available": False, "stale": True, "provenance": provenance}
                continue
            if not widened:
                continue
            evidence_kinds = _evidence_kinds_from_actions(row.get("actions") or ())
            candidate = {
                "available": True, "stale": False, "provenance": provenance,
                "handle_source_start": candidate_start, "handle_source_end": candidate_end,
                "evidence_kinds": evidence_kinds, "kind_gated": is_authoritative_kind_gated,
            }
            if best is None or not best.get("available") or (
                candidate["handle_source_end"] - candidate["handle_source_start"]
                > best["handle_source_end"] - best["handle_source_start"]
            ):
                best = candidate
    if best is None:
        return {"available": False, "stale": False, "provenance": PROVENANCE_NO_TRIM_RECORDED}
    return best


def _classify_speech_presence(
    source_asset_id: str,
    handle_start: float,
    handle_end: float,
    *,
    broader_word_timings: Mapping[str, Sequence[Word]],
) -> Tuple[str, Tuple[Tuple[float, float, str], ...]]:
    """`NO_WORDS_PRESENT` is the default for every handle this module
    derives (a structural proof from the owning Boundary authority's own
    word-boundary clamp -- see module docstring). A caller-supplied
    `broader_word_timings` (empty for every live caller today, D-222 item
    11's own confirmed gap) can surface words this module's own derivation
    could never see on its own -- used for future extensibility and
    offline fixture proof only."""
    words = broader_word_timings.get(source_asset_id) or ()
    found = tuple(
        (float(w.start), float(w.end), w.text)
        for w in words
        if _intervals_overlap(float(w.start), float(w.end), handle_start, handle_end)
    )
    if found:
        return SPEECH_PRESENCE_WORDS_PRESENT, found
    return SPEECH_PRESENCE_NO_WORDS, ()


def _classify_meaning_safety(word_intervals: Sequence[Tuple[float, float, str]]) -> str:
    if not word_intervals:
        return MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS
    text = " ".join(text for (_, _, text) in sorted(word_intervals) if text).strip()
    if not text:
        return MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS
    _, importance, _ = classify_claim(text)
    return MEANING_SAFETY_BLOCKED_CRITICAL if importance == CRITICAL else MEANING_SAFETY_SAFE


def _overlap_status(
    source_asset_id: str, handle_start: float, handle_end: float, owner_clip_id: str,
    *, discarded: Sequence[DraftClip], other_selected: Sequence[DraftClip],
) -> str:
    for clip in discarded or ():
        if clip.source_asset_id != source_asset_id:
            continue
        if _intervals_overlap(handle_start, handle_end, float(clip.start), float(clip.end)):
            return OVERLAP_DISCARDED
    for clip in other_selected or ():
        if clip.clip_id == owner_clip_id or clip.source_asset_id != source_asset_id:
            continue
        if _intervals_overlap(handle_start, handle_end, float(clip.start), float(clip.end)):
            return OVERLAP_NEIGHBOR_SELECTED
    return OVERLAP_NONE


def _decide_handle_status(
    *, overlap_status: str, retry_evidence_kinds: Sequence[str],
    meaning_safety_status: str, speech_presence_status: str,
) -> str:
    """The one, exhaustive, priority-ordered decision table. Extracted as a
    pure function so every combination -- including `SPEECH_PRESENCE_
    UNKNOWN`, which the two current provenance sources never actually
    produce (both offer a positive structural proof, see module docstring)
    but which a FUTURE provenance source lacking that proof would -- is
    directly unit-testable without fabricating a live-reachable fixture for
    a combination today's real inputs cannot produce."""
    if overlap_status == OVERLAP_DISCARDED:
        return HANDLE_STATUS_BLOCKED_DISCARDED
    if overlap_status == OVERLAP_NEIGHBOR_SELECTED:
        return HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP
    if retry_evidence_kinds:
        return HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION
    if meaning_safety_status == MEANING_SAFETY_BLOCKED_CRITICAL:
        return HANDLE_STATUS_BLOCKED_MEANING_CRITICAL
    if speech_presence_status == SPEECH_PRESENCE_UNKNOWN:
        return HANDLE_STATUS_UNKNOWN_WORD_COVERAGE
    if speech_presence_status == SPEECH_PRESENCE_WORDS_PRESENT:
        # Conservative by this task's own explicit rule: word presence alone
        # never authorizes reuse -- see module docstring "Why an 8th
        # handle-status value". Never `SAFE_SPEECH_HANDLE` from this table.
        return HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE
    return HANDLE_STATUS_SAFE_NON_SPEECH


def _build_handle(
    clip: DraftClip,
    direction: str,
    *,
    boundary_engine_pass_audit: Sequence[Mapping],
    post_selection_edge_only_boundary_audit: Sequence[Mapping],
    discarded: Sequence[DraftClip],
    other_selected: Sequence[DraftClip],
    source_duration_sec: Optional[float],
    broader_word_timings: Mapping[str, Sequence[Word]],
) -> SourceAudioHandle:
    conflict_flags: list = []
    provenance: list = []
    video_start, video_end = float(clip.start), float(clip.end)

    widened = _widen_bound(
        clip, direction,
        boundary_engine_pass_audit=boundary_engine_pass_audit,
        post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
    )
    provenance.append(widened["provenance"])
    if widened.get("stale"):
        conflict_flags.append(CONFLICT_STALE_PROVENANCE)
    if not widened.get("available"):
        if widened["provenance"] == PROVENANCE_NO_TRIM_RECORDED:
            conflict_flags.append(CONFLICT_NO_PROVENANCE)
        return SourceAudioHandle(
            schema_version=SCHEMA_VERSION,
            handle_id=_handle_id(clip.source_asset_id, clip.clip_id, direction, video_start, video_end),
            source_asset_id=clip.source_asset_id, owner_clip_id=clip.clip_id,
            owner_realization_id=clip.realization_id, direction=direction,
            video_start=video_start, video_end=video_end,
            handle_source_start=video_start if direction == DIRECTION_PRE_ROLL else video_end,
            handle_source_end=video_start if direction == DIRECTION_PRE_ROLL else video_end,
            available_duration=0.0, word_intervals_present=(),
            speech_presence_status=SPEECH_PRESENCE_NO_WORDS,
            discarded_overlap_status=OVERLAP_NOT_APPLICABLE,
            meaning_safety_status=MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS,
            handle_status=HANDLE_STATUS_UNAVAILABLE,
            conflict_flags=tuple(conflict_flags), provenance=tuple(provenance),
        )

    handle_start = widened["handle_source_start"]
    handle_end = widened["handle_source_end"]

    # Source-duration invariant (this task's own explicit contract, mirrors
    # -- without importing -- render.py's own `validate_audio_window` shape):
    # invalid geometry is UNAVAILABLE/blocked outright, never silently clamped.
    geometry_invalid = handle_start < 0.0 or not (handle_start < handle_end)
    exceeds_source = (
        source_duration_sec is not None and handle_end > float(source_duration_sec) + _RESULT_MATCH_EPSILON_SEC
    )
    if geometry_invalid:
        conflict_flags.append(CONFLICT_INVALID_GEOMETRY)
        conflict_flags.append(CONFLICT_ZERO_OR_NEGATIVE_DURATION)
    if exceeds_source:
        conflict_flags.append(CONFLICT_EXCEEDS_SOURCE_DURATION)
    if geometry_invalid or exceeds_source:
        return SourceAudioHandle(
            schema_version=SCHEMA_VERSION,
            handle_id=_handle_id(clip.source_asset_id, clip.clip_id, direction, handle_start, handle_end),
            source_asset_id=clip.source_asset_id, owner_clip_id=clip.clip_id,
            owner_realization_id=clip.realization_id, direction=direction,
            video_start=video_start, video_end=video_end,
            handle_source_start=handle_start, handle_source_end=handle_end,
            available_duration=0.0, word_intervals_present=(),
            speech_presence_status=SPEECH_PRESENCE_NO_WORDS,
            discarded_overlap_status=OVERLAP_NOT_APPLICABLE,
            meaning_safety_status=MEANING_SAFETY_NOT_APPLICABLE_NO_WORDS,
            handle_status=HANDLE_STATUS_UNAVAILABLE,
            conflict_flags=tuple(conflict_flags), provenance=tuple(provenance),
        )

    available_duration = handle_end - handle_start

    overlap_status = _overlap_status(
        clip.source_asset_id, handle_start, handle_end, clip.clip_id,
        discarded=discarded, other_selected=other_selected,
    )
    if overlap_status == OVERLAP_DISCARDED:
        conflict_flags.append(CONFLICT_OVERLAPS_DISCARDED)
    elif overlap_status == OVERLAP_NEIGHBOR_SELECTED:
        conflict_flags.append(CONFLICT_OVERLAPS_NEIGHBOR_SELECTED)

    retry_evidence_kinds = tuple(
        kind for kind in widened.get("evidence_kinds", ()) if kind in _RETRY_OR_BTS_EVIDENCE_KINDS
    )
    if retry_evidence_kinds:
        conflict_flags.append(CONFLICT_RETRY_OR_BTS_EVIDENCE)
        provenance.extend(f"evidence_kind:{kind}" for kind in retry_evidence_kinds)

    speech_presence_status, word_intervals = _classify_speech_presence(
        clip.source_asset_id, handle_start, handle_end, broader_word_timings=broader_word_timings,
    )
    meaning_safety_status = _classify_meaning_safety(word_intervals)
    if meaning_safety_status == MEANING_SAFETY_BLOCKED_CRITICAL:
        conflict_flags.append(CONFLICT_MEANING_CRITICAL)
    if speech_presence_status == SPEECH_PRESENCE_UNKNOWN:
        conflict_flags.append(CONFLICT_WORD_COVERAGE_UNKNOWN)

    handle_status = _decide_handle_status(
        overlap_status=overlap_status, retry_evidence_kinds=retry_evidence_kinds,
        meaning_safety_status=meaning_safety_status, speech_presence_status=speech_presence_status,
    )

    return SourceAudioHandle(
        schema_version=SCHEMA_VERSION,
        handle_id=_handle_id(clip.source_asset_id, clip.clip_id, direction, handle_start, handle_end),
        source_asset_id=clip.source_asset_id, owner_clip_id=clip.clip_id,
        owner_realization_id=clip.realization_id, direction=direction,
        video_start=video_start, video_end=video_end,
        handle_source_start=handle_start, handle_source_end=handle_end,
        available_duration=available_duration, word_intervals_present=word_intervals,
        speech_presence_status=speech_presence_status,
        discarded_overlap_status=overlap_status,
        meaning_safety_status=meaning_safety_status,
        handle_status=handle_status,
        conflict_flags=tuple(conflict_flags), provenance=tuple(provenance),
    )


def build_pre_roll_audio_handle(
    clip: DraftClip,
    *,
    boundary_engine_pass_audit: Sequence[Mapping] = (),
    post_selection_edge_only_boundary_audit: Sequence[Mapping] = (),
    discarded: Sequence[DraftClip] = (),
    other_selected: Sequence[DraftClip] = (),
    source_duration_sec: Optional[float] = None,
    broader_word_timings: Mapping[str, Sequence[Word]] = _EMPTY_MAPPING,
) -> SourceAudioHandle:
    """The room immediately BEFORE `clip.start`, already proven safe (or
    already proven recording-process/retry debris) by an already-accepted
    Boundary-adjacent authority's own audit trail. Used, in a future,
    separately-authorized gate (D-224), as the candidate PRE-ROLL source for
    this clip acting as the RIGHT member of a J-cut. `clip.start`/`clip.end`
    are never read as mutable and never changed."""
    return _build_handle(
        clip, DIRECTION_PRE_ROLL,
        boundary_engine_pass_audit=boundary_engine_pass_audit,
        post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
        discarded=discarded, other_selected=other_selected,
        source_duration_sec=source_duration_sec, broader_word_timings=broader_word_timings,
    )


def build_post_roll_audio_handle(
    clip: DraftClip,
    *,
    boundary_engine_pass_audit: Sequence[Mapping] = (),
    post_selection_edge_only_boundary_audit: Sequence[Mapping] = (),
    discarded: Sequence[DraftClip] = (),
    other_selected: Sequence[DraftClip] = (),
    source_duration_sec: Optional[float] = None,
    broader_word_timings: Mapping[str, Sequence[Word]] = _EMPTY_MAPPING,
) -> SourceAudioHandle:
    """Symmetric to `build_pre_roll_audio_handle` -- the room immediately
    AFTER `clip.end`, for this clip acting as the LEFT member of an L-cut."""
    return _build_handle(
        clip, DIRECTION_POST_ROLL,
        boundary_engine_pass_audit=boundary_engine_pass_audit,
        post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
        discarded=discarded, other_selected=other_selected,
        source_duration_sec=source_duration_sec, broader_word_timings=broader_word_timings,
    )


def build_source_audio_handles(
    selected: Sequence[DraftClip],
    *,
    discarded: Sequence[DraftClip] = (),
    boundary_engine_pass_audit: Sequence[Mapping] = (),
    post_selection_edge_only_boundary_audit: Sequence[Mapping] = (),
    source_duration_by_asset: Mapping[str, float] = _EMPTY_MAPPING,
    broader_word_timings: Mapping[str, Sequence[Word]] = _EMPTY_MAPPING,
) -> Tuple[SourceAudioHandle, ...]:
    """Builds BOTH a PRE_ROLL and a POST_ROLL handle for every clip in
    `selected` -- pairing a handle to a specific J/L transition is NOT this
    module's job (matches D-217's own per-clip, not per-pair, `available_
    silent_head_sec`/`available_silent_tail_sec` shape, reused here at the
    wider source-audio-handle scope). No global state, no pipeline
    reordering -- pure function of its own inputs."""
    clips = tuple(selected)
    handles: list = []
    for clip in clips:
        others = tuple(c for c in clips if c.clip_id != clip.clip_id)
        duration = source_duration_by_asset.get(clip.source_asset_id)
        handles.append(build_pre_roll_audio_handle(
            clip, boundary_engine_pass_audit=boundary_engine_pass_audit,
            post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
            discarded=discarded, other_selected=others,
            source_duration_sec=duration, broader_word_timings=broader_word_timings,
        ))
        handles.append(build_post_roll_audio_handle(
            clip, boundary_engine_pass_audit=boundary_engine_pass_audit,
            post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
            discarded=discarded, other_selected=others,
            source_duration_sec=duration, broader_word_timings=broader_word_timings,
        ))
    return tuple(handles)


def source_audio_handle_diagnostics(handle: SourceAudioHandle) -> dict:
    """The exact per-handle diagnostics row this task's own directive
    requires -- one dict, JSON-safe, no additional derivation."""
    return {
        "handle_id": handle.handle_id,
        "direction": handle.direction,
        "source_asset_id": handle.source_asset_id,
        "owner_clip_id": handle.owner_clip_id,
        "video_start": handle.video_start,
        "video_end": handle.video_end,
        "handle_start": handle.handle_source_start,
        "handle_end": handle.handle_source_end,
        "available_duration": handle.available_duration,
        "word_coverage_status": handle.speech_presence_status,
        "words_present_count": len(handle.word_intervals_present),
        "discarded_overlap_status": handle.discarded_overlap_status,
        "meaning_safety_status": handle.meaning_safety_status,
        "handle_status": handle.handle_status,
        "conflict_flags": list(handle.conflict_flags),
        "provenance": list(handle.provenance),
    }


def source_audio_handle_run_summary(handles: Sequence[SourceAudioHandle]) -> dict:
    """Counts only -- NO master/global score of any kind (this track's own
    binding convention, see D-220's own `timing_policy_run_summary`)."""
    handles = tuple(handles)
    pre_roll = tuple(h for h in handles if h.direction == DIRECTION_PRE_ROLL)
    post_roll = tuple(h for h in handles if h.direction == DIRECTION_POST_ROLL)
    return {
        "schema_version": SCHEMA_VERSION,
        "pre_roll_handle_count": len(pre_roll),
        "post_roll_handle_count": len(post_roll),
        "safe_non_speech_handle_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_SAFE_NON_SPEECH),
        "speech_present_handle_count": sum(1 for h in handles if h.speech_presence_status == SPEECH_PRESENCE_WORDS_PRESENT),
        "speech_present_not_authoritative_count": sum(
            1 for h in handles if h.handle_status == HANDLE_STATUS_SPEECH_PRESENT_NOT_AUTHORITATIVE
        ),
        "blocked_discarded_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_BLOCKED_DISCARDED),
        "blocked_neighbor_selected_clip_count": sum(
            1 for h in handles if h.handle_status == HANDLE_STATUS_BLOCKED_NEIGHBOR_SELECTED_CLIP
        ),
        "blocked_retry_or_correction_count": sum(
            1 for h in handles if h.handle_status == HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION
        ),
        "blocked_meaning_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_BLOCKED_MEANING_CRITICAL),
        "unknown_word_coverage_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_UNKNOWN_WORD_COVERAGE),
        "unavailable_count": sum(1 for h in handles if h.handle_status == HANDLE_STATUS_UNAVAILABLE),
        "total_safe_handle_duration": sum(
            h.available_duration for h in handles if h.handle_status == HANDLE_STATUS_SAFE_NON_SPEECH
        ),
    }
