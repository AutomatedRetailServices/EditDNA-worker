"""D-234: Pacing V2 Audio Join Treatment -- Live Diagnostic Integration,
NO AUTHORITY. Wires the closed, offline-proven D-230 -> D-231 -> D-232 ->
D-233 chain (AcousticEdgeEvidence -> AudioJoinUnderstanding ->
AudioJoinTreatmentDecision -> AudioJoinTreatmentTimingPlan) into the
existing live Clean Cut diagnostic path, at the SAME seam D-216/D-217/
D-224 already use in `universal_clean_cut.py`.

Mirrors D-216/D-224's own precedent exactly: default-OFF env flag, pure
adapter over already-computed real pipeline state, zero new evidence
source, zero live authority.

## What this module is NOT (binding, restated from this task's own scope)

- No live audio-treatment execution. `render.py`'s D-233
  `render_audio_join_treatment_preview` executor is never imported here.
- No renderer-plan mutation. This module never imports `render_plan`/
  `render`, and its output is attached ONLY to `draft.diagnostics` as a
  new, additive key -- `draft.selected`, `RenderSegment`, Boundary,
  Ordering, Family/BestTake, and the D-142 primary transition mode are
  never read back into or reassigned by anything in this module.
- No new provider call, no ASR rerun, no semantic rerun. Handles are
  built via D-223's own `build_source_audio_handles` over ALREADY-
  SELECTED clips' ALREADY-KNOWN word timings/geometry; relationship
  hints/Prosodic evidence are D-217's own `build_pacing_v2_real_evidence`
  over diagnostics dicts already threaded through `draft.diagnostics`.
  `audio=None` is passed to every D-230 `build_acoustic_edge_evidence`
  call (this module never decodes or reads a raw audio sample) -- edge
  evidence here is built purely from word-timing/handle-status
  structure, per D-230's own documented `audio=None` convention (fails
  soft to the SAFE/UNKNOWN shapes, never crashes, never fabricates
  energy/silence measurements it does not have).
- No live J_CUT/L_CUT/MICRO_AUDIO_OVERLAP authority and no MICRO
  treatment value: `primary_transition_mode` is read-only, taken from
  D-142's own already-executed live mode
  (`draft.diagnostics["dialogue_pacing_transition"]["transitions"][i]
  ["mode"]`) -- never recomputed, never overwritten.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from .contracts import DraftClip
from .pacing_v2_acoustic_edge_evidence import (
    AcousticEdgeEvidence,
    EDGE_LEFT_END,
    EDGE_POST_HANDLE,
    EDGE_PRE_HANDLE,
    EDGE_RIGHT_START,
    EVIDENCE_STATUS_CONFLICTED,
    EVIDENCE_STATUS_INSUFFICIENT,
    EVIDENCE_STATUS_SAFE_FALLBACK,
    EVIDENCE_STATUS_SUPPORTED,
    build_acoustic_edge_evidence,
    compare_acoustic_edges,
    derive_retained_edge_window,
)
from .pacing_v2_audio_join_treatment_decision import (
    TREATMENT_AMBIENCE_BRIDGE,
    TREATMENT_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT,
    TREATMENT_CLICK_FADE,
    TREATMENT_NONE,
    TREATMENT_SHORT_CROSSFADE,
    audio_join_treatment_decision_diagnostics,
    build_audio_join_treatment_decision,
)
from .pacing_v2_audio_join_treatment_timing import (
    TIMING_SUPPORTED,
    audio_join_treatment_timing_diagnostics,
    build_audio_join_treatment_timing_plan,
)
from .pacing_v2_audio_join_understanding import (
    UNDERSTANDING_AVAILABLE,
    UNDERSTANDING_CONFLICTED,
    UNDERSTANDING_PARTIAL,
    audio_join_understanding_diagnostics,
    build_audio_join_understanding,
)
from .pacing_v2_evidence_adapter import build_pacing_v2_real_evidence
from .pacing_v2_source_audio_handle import (
    DIRECTION_POST_ROLL,
    DIRECTION_PRE_ROLL,
    build_source_audio_handles,
)

SCHEMA_VERSION = "cutsell.pacing_v2_audio_join_treatment_live_diagnostics.v1"
_DIAGNOSTICS_ENV = "CUTSELL_AUDIO_JOIN_TREATMENT_DIAGNOSTICS_ENABLED"

STATUS_AVAILABLE = "AVAILABLE"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_CONFLICTED = "CONFLICTED"

_ADVANCED_TREATMENTS = (
    TREATMENT_SHORT_CROSSFADE,
    TREATMENT_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT,
    TREATMENT_AMBIENCE_BRIDGE,
)
_EMPTY: Mapping = {}


def _env_true_default_false(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def audio_join_treatment_diagnostics_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Default OFF. When OFF, `universal_clean_cut.py` never calls
    `build_audio_join_treatment_live_diagnostics` -- zero D-230-D-233
    compute, D-142/D-216/D-224's own live output stays byte-identical."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


def _handle_lookup(handles) -> dict:
    return {(h.owner_clip_id, h.direction): h for h in handles}


def _words_as_tuples(clip: DraftClip):
    return tuple((float(w.start), float(w.end), w.text) for w in (clip.words or ()))


def _retained_edge_evidence(clip: DraftClip, edge: str, source_asset_id: str) -> Optional[AcousticEdgeEvidence]:
    """Bounded retained-edge evidence for one clip. `audio=None` always --
    this module never decodes raw audio; word-derived geometry only (D-230's
    own documented convention, fails soft to INSUFFICIENT/UNKNOWN)."""
    words = _words_as_tuples(clip)
    window = derive_retained_edge_window(words, edge)
    if window is None:
        return None
    window_start, window_end = window
    return build_acoustic_edge_evidence(
        source_asset_id=source_asset_id, owner_clip_id=clip.clip_id, edge=edge,
        window_start=window_start, window_end=window_end,
        audio=None, silence_intervals=None,
        word_intervals=words, word_coverage_known=True,
    )


def _handle_edge_evidence(handle) -> Optional[AcousticEdgeEvidence]:
    """Bounded handle-edge evidence built purely from the D-223 handle's
    own already-computed fields (word coverage / discarded / retry /
    meaning-safety) -- `audio=None`, no invented handle, no widened
    contract (this task's own explicit D-223 firewall)."""
    if handle is None:
        return None
    edge = EDGE_PRE_HANDLE if handle.direction == DIRECTION_PRE_ROLL else EDGE_POST_HANDLE
    return build_acoustic_edge_evidence(
        source_asset_id=handle.source_asset_id, owner_clip_id=handle.owner_clip_id, edge=edge,
        window_start=handle.handle_source_start, window_end=handle.handle_source_end,
        audio=None, silence_intervals=None, source_handle=handle,
    )


def _evidence_availability(*evidences: Optional[AcousticEdgeEvidence]) -> str:
    present = [e for e in evidences if e is not None]
    if not present:
        return STATUS_UNAVAILABLE
    if any(e.evidence_status == EVIDENCE_STATUS_CONFLICTED for e in present):
        return STATUS_CONFLICTED
    if len(present) < len(evidences) or any(
        e.evidence_status in (EVIDENCE_STATUS_INSUFFICIENT,) for e in present
    ):
        return STATUS_PARTIAL
    if all(e.evidence_status in (EVIDENCE_STATUS_SUPPORTED, EVIDENCE_STATUS_SAFE_FALLBACK) for e in present):
        return STATUS_AVAILABLE
    return STATUS_PARTIAL


def build_audio_join_treatment_live_diagnostics(
    selected: Sequence[DraftClip],
    *,
    dialogue_overlap_enabled: bool,
    boundary_diagnostics: Optional[Mapping] = None,
    discarded: Sequence[DraftClip] = (),
    boundary_engine_pass_audit: Sequence[Mapping] = (),
    post_selection_edge_only_boundary_audit: Sequence[Mapping] = (),
    source_duration_by_asset: Optional[Mapping[str, float]] = None,
    editorial_moment_sequence_diagnostics: Optional[Mapping] = None,
    take_judge_groups: Sequence[Mapping] = (),
    live_transition_modes: Sequence[str] = (),
    _source_audio_cache: Optional[dict] = None,
) -> dict:
    """The one D-234 entry point. Pure function of already-computed real
    pipeline state -- builds no new evidence source of its own beyond
    D-223's own handle foundation and D-217's own relationship/Prosodic
    evidence (both reused verbatim, never duplicated), then runs the
    closed D-230->D-231->D-232->D-233 chain per adjacent join,
    DIAGNOSTIC ONLY. `_source_audio_cache` is an optional, bounded,
    in-run-only dict a caller may pass to reuse across multiple calls in
    the same request; this module never populates persistent storage."""
    clips = tuple(selected)
    boundary_diagnostics = boundary_diagnostics or {}
    source_duration_by_asset = source_duration_by_asset or _EMPTY
    cache = _source_audio_cache if _source_audio_cache is not None else {}

    if len(clips) < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_UNAVAILABLE,
            "join_count": 0,
            "per_join": (),
            "summary": _summary((), (), (), ()),
        }

    # D-223 (reused verbatim, not widened) + D-217 (reused verbatim).
    handles = build_source_audio_handles(
        clips, discarded=discarded,
        boundary_engine_pass_audit=boundary_engine_pass_audit,
        post_selection_edge_only_boundary_audit=post_selection_edge_only_boundary_audit,
        source_duration_by_asset=source_duration_by_asset,
    )
    handle_by_owner_direction = _handle_lookup(handles)
    evidence = build_pacing_v2_real_evidence(
        clips,
        editorial_moment_sequence_diagnostics=editorial_moment_sequence_diagnostics,
        take_judge_groups=take_judge_groups,
    )
    relationship_hint_by_pair = evidence["relationship_hint_by_pair"]
    prosody_by_clip_id = evidence["prosody_by_clip_id"]

    understandings = []
    decisions = []
    timing_plans = []
    per_join = []
    evidence_statuses = []

    for index in range(len(clips) - 1):
        left, right = clips[index], clips[index + 1]
        left_source = left.source_asset_id
        right_source = right.source_asset_id

        # Bounded, in-run cache keyed by (source_asset_id, clip_id, edge) --
        # a source's retained-edge evidence never changes across joins that
        # happen to reuse it, so this only ever avoids recomputation, never
        # widens what is computed. No persistent cache infrastructure.
        left_key = ("retained", left_source, left.clip_id, EDGE_LEFT_END)
        right_key = ("retained", right_source, right.clip_id, EDGE_RIGHT_START)
        if left_key not in cache:
            cache[left_key] = _retained_edge_evidence(left, EDGE_LEFT_END, left_source)
        if right_key not in cache:
            cache[right_key] = _retained_edge_evidence(right, EDGE_RIGHT_START, right_source)
        left_edge_evidence = cache[left_key]
        right_edge_evidence = cache[right_key]

        left_post_handle = handle_by_owner_direction.get((left.clip_id, DIRECTION_POST_ROLL))
        right_pre_handle = handle_by_owner_direction.get((right.clip_id, DIRECTION_PRE_ROLL))
        left_post_handle_key = ("handle", left_post_handle.handle_id) if left_post_handle else None
        right_pre_handle_key = ("handle", right_pre_handle.handle_id) if right_pre_handle else None
        if left_post_handle_key is not None and left_post_handle_key not in cache:
            cache[left_post_handle_key] = _handle_edge_evidence(left_post_handle)
        if right_pre_handle_key is not None and right_pre_handle_key not in cache:
            cache[right_pre_handle_key] = _handle_edge_evidence(right_pre_handle)
        left_post_handle_edge_evidence = cache.get(left_post_handle_key) if left_post_handle_key else None
        right_pre_handle_edge_evidence = cache.get(right_pre_handle_key) if right_pre_handle_key else None

        continuity_comparison = (
            compare_acoustic_edges(left_edge_evidence, right_edge_evidence)
            if left_edge_evidence is not None and right_edge_evidence is not None else None
        )

        key = (left.clip_id, right.clip_id)
        relationship_hint = relationship_hint_by_pair.get(key)
        left_prosody = prosody_by_clip_id.get(left.clip_id)
        right_prosody = prosody_by_clip_id.get(right.clip_id)

        understanding = build_audio_join_understanding(
            transition_index=index,
            left_clip_id=left.clip_id, right_clip_id=right.clip_id,
            left_source_asset_id=left_source, right_source_asset_id=right_source,
            left_edge_evidence=left_edge_evidence, right_edge_evidence=right_edge_evidence,
            continuity_comparison=continuity_comparison,
            left_post_roll_handle=left_post_handle, right_pre_roll_handle=right_pre_handle,
            left_post_roll_handle_edge_evidence=left_post_handle_edge_evidence,
            right_pre_roll_handle_edge_evidence=right_pre_handle_edge_evidence,
            relationship_hint=relationship_hint,
            left_prosody=left_prosody, right_prosody=right_prosody,
            left_edge_words=_words_as_tuples(left), right_edge_words=_words_as_tuples(right),
        )
        understandings.append(understanding)
        evidence_statuses.append(_evidence_availability(left_edge_evidence, right_edge_evidence))

        # D-142's own already-executed live mode -- read-only, never recomputed.
        primary_transition_mode = (
            live_transition_modes[index] if index < len(live_transition_modes) and live_transition_modes[index]
            else "UNKNOWN"
        )
        candidate_duration_sec = None
        if left_post_handle is not None and right_pre_handle is not None:
            candidate_duration_sec = min(left_post_handle.available_duration, right_pre_handle.available_duration)

        decision = build_audio_join_treatment_decision(
            understanding, primary_transition_mode=primary_transition_mode,
            left_post_roll_handle_status=left_post_handle.handle_status if left_post_handle else None,
            right_pre_roll_handle_status=right_pre_handle.handle_status if right_pre_handle else None,
            candidate_duration_sec=candidate_duration_sec,
        )
        decisions.append(decision)

        # D-233: diagnostic timing plan. Always built (the module itself
        # returns a NOT_APPLICABLE-shaped plan for NONE/CLICK_FADE) --
        # never executed, never substituting a different advanced treatment.
        timing_plan = build_audio_join_treatment_timing_plan(
            decision,
            left_source_asset_id=left_source, right_source_asset_id=right_source,
            left_video_start=float(left.start), left_video_end=float(left.end),
            left_source_duration=float(source_duration_by_asset.get(left_source, 0.0)),
            right_video_start=float(right.start), right_video_end=float(right.end),
            right_source_duration=float(source_duration_by_asset.get(right_source, 0.0)),
            left_safe_audio_window_sec=left_post_handle.available_duration if left_post_handle else None,
            right_safe_audio_window_sec=right_pre_handle.available_duration if right_pre_handle else None,
            left_post_roll_handle=left_post_handle, right_pre_roll_handle=right_pre_handle,
        )
        timing_plans.append(timing_plan)

        row = {
            "transition_index": index,
            "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
            "primary_transition_mode": primary_transition_mode,
            "acoustic_evidence_status": evidence_statuses[-1],
            "understanding": audio_join_understanding_diagnostics(understanding),
            "decision": audio_join_treatment_decision_diagnostics(decision),
            "timing": audio_join_treatment_timing_diagnostics(timing_plan),
            "recommended_audio_treatment": decision.treatment,
        }
        per_join.append(row)

    status = _overall_status(understandings, evidence_statuses)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "join_count": len(per_join),
        "per_join": per_join,
        "summary": _summary(evidence_statuses, understandings, decisions, timing_plans),
    }


def _overall_status(understandings, evidence_statuses) -> str:
    if not understandings:
        return STATUS_UNAVAILABLE
    if any(u.understanding_status == UNDERSTANDING_CONFLICTED for u in understandings) or \
            STATUS_CONFLICTED in evidence_statuses:
        return STATUS_CONFLICTED
    if all(
        u.understanding_status == UNDERSTANDING_AVAILABLE and s == STATUS_AVAILABLE
        for u, s in zip(understandings, evidence_statuses)
    ):
        return STATUS_AVAILABLE
    if any(
        u.understanding_status in (UNDERSTANDING_AVAILABLE, UNDERSTANDING_PARTIAL) or s in (STATUS_AVAILABLE, STATUS_PARTIAL)
        for u, s in zip(understandings, evidence_statuses)
    ):
        return STATUS_PARTIAL
    return STATUS_UNAVAILABLE


def _summary(evidence_statuses, understandings, decisions, timing_plans) -> dict:
    def _count(seq, pred) -> int:
        return sum(1 for item in seq if pred(item))

    advanced_recommended = _count(decisions, lambda d: d.treatment in _ADVANCED_TREATMENTS)
    advanced_timing_supported = _count(
        timing_plans, lambda p: p.treatment in _ADVANCED_TREATMENTS and p.timing_status == TIMING_SUPPORTED
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "join_count": len(decisions),

        "evidence_available_count": _count(evidence_statuses, lambda s: s == STATUS_AVAILABLE),
        "evidence_partial_count": _count(evidence_statuses, lambda s: s == STATUS_PARTIAL),
        "evidence_unknown_count": _count(evidence_statuses, lambda s: s in (STATUS_UNAVAILABLE, STATUS_CONFLICTED)),

        "understanding_available_count": _count(understandings, lambda u: u.understanding_status == UNDERSTANDING_AVAILABLE),
        "understanding_partial_count": _count(understandings, lambda u: u.understanding_status == UNDERSTANDING_PARTIAL),
        "understanding_conflicted_count": _count(understandings, lambda u: u.understanding_status == UNDERSTANDING_CONFLICTED),

        "none_count": _count(decisions, lambda d: d.treatment == TREATMENT_NONE),
        "click_fade_count": _count(decisions, lambda d: d.treatment == TREATMENT_CLICK_FADE),
        "short_crossfade_count": _count(decisions, lambda d: d.treatment == TREATMENT_SHORT_CROSSFADE),
        "ambience_left_count": _count(decisions, lambda d: d.treatment == TREATMENT_AMBIENCE_CARRY_LEFT),
        "ambience_right_count": _count(decisions, lambda d: d.treatment == TREATMENT_AMBIENCE_CARRY_RIGHT),
        "ambience_bridge_count": _count(decisions, lambda d: d.treatment == TREATMENT_AMBIENCE_BRIDGE),

        "timing_supported_count": _count(timing_plans, lambda p: p.timing_status == TIMING_SUPPORTED),
        "timing_blocked_count": _count(
            timing_plans, lambda p: p.treatment in _ADVANCED_TREATMENTS and p.timing_status != TIMING_SUPPORTED
        ),

        "advanced_treatment_recommended_count": advanced_recommended,
        "advanced_timing_supported_count": advanced_timing_supported,
        # D-234 NEVER executes an advanced treatment: this is always 0,
        # structurally -- this module never imports or calls
        # `render.py`'s `render_audio_join_treatment_preview`.
        "advanced_treatment_executed_count": 0,
        # D-234 NEVER mutates `RenderSegment`/render-plan audio windows:
        # this is always 0, structurally -- this module never imports
        # `render_plan`/`render` and only ever returns a plain dict.
        "live_audio_window_mutation_count": 0,

        "word_block_count": _count(decisions, lambda d: d.word_safety_status == "BLOCKED"),
        "meaning_block_count": _count(decisions, lambda d: d.meaning_safety_status == "BLOCKED"),
        "double_speech_block_count": _count(decisions, lambda d: d.double_speech_status == "BOTH_SIDES_LEXICAL"),

        "safe_left_handle_count": _count(decisions, lambda d: d.left_handle_status == "SAFE_NON_SPEECH"),
        "safe_right_handle_count": _count(decisions, lambda d: d.right_handle_status == "SAFE_NON_SPEECH"),

        "acoustically_similar_count": _count(understandings, lambda u: u.acoustic_continuity_status == "SIMILAR"),
        "acoustically_different_count": _count(understandings, lambda u: u.acoustic_continuity_status == "DIFFERENT"),
        "acoustic_insufficient_count": _count(
            understandings, lambda u: u.acoustic_continuity_status in ("INSUFFICIENT", "UNKNOWN")
        ),

        "loudness_polish_needed_count": _count(decisions, lambda d: d.loudness_polish_status == "LOUDNESS_POLISH_NEEDED"),
    }
