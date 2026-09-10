"""D-216: Pacing V2 Live Diagnostic Integration -- OFFLINE-FIRST / NO LIVE
J/L/MICRO-OVERLAP AUTHORITY. A DIAGNOSTIC SIDE-CHANNEL over the real
pipeline's already-computed, real, post-Boundary `draft.selected` clips --
answering "what would D-215's Pacing V2 decision foundation recommend for
each adjacent join?" without ever changing what the live renderer does.

Mirrors D-208's exact precedent (`ordering_live_diagnostics_integration.py`)
for the Ordering track: default-OFF env flag, zero live authority, pure
adapter over already-computed real objects, no new evidence construction
of its own.

## What this module is NOT (binding, restated from this task's own scope)

No live J/L/MICRO_AUDIO_OVERLAP authority: the live `mode` D-142's own
`apply_dialogue_pacing_transition_pass` already chose (HARD_CUT/TIGHT_CUT
only) is NEVER overwritten, and nothing here is ever written back onto
`draft.selected` or `RenderSegment`. No renderer live use: D-214's own
independent-audio-window contract (`RenderSegment.audio_start`/`.audio_end`)
is never populated by this module for a live segment -- structurally
proven (this module never imports `render_plan`/`render`). No re-
computation of ASR, P1, P2, Ordering, BestTake, or Prosodic evidence --
every optional evidence source (`prosody_by_clip_id`, `relationship_
hint_by_pair`, `candidate_timing_by_pair`) is accepted BY REFERENCE from
the caller; this module never builds one itself.

## Live evidence-availability finding (honest, not a limitation to fix here)

Direct inspection of `universal_clean_cut.py` and `pipeline.py` (this
task's own required forensic) found: (1) NO in-memory `ProsodicDelivery
Evidence` object exists mapped to a final, Boundary-finalized `DraftClip`
at the Pacing seam -- D-187/D-188's own Prosodic evidence is computed
earlier, per-CANDIDATE, during BestTake arbitration
(`bounded_finalist_arbiter.py`), and is never carried through to the final
selected clip; (2) no per-adjacent-pair relationship hint
(`continuation`/`correction`/`retry`) is currently extracted from P1's
`editorial_moment_sequence`/P2's `whole_video_editorial_reasoning`
diagnostics for the FINAL selected sequence, though those diagnostics
dicts do persist in `draft.diagnostics` by this point (an honest, bounded
finding for a future gate, never attempted here per this task's own "do
not rerun P1/P2" and "minimum code required" instructions); (3) no
existing bounded candidate lead/tail/overlap timing source exists at this
seam at all. The live wiring below therefore honestly passes `None` for
all three at the one real call site (`universal_clean_cut.py`), and every
real diagnostic decision this produces today resolves to `HARD_CUT`/
`TIGHT_CUT`/`KEEP_PAUSE`/`UNKNOWN` -- NEVER `J_CUT`/`L_CUT`/`MICRO_AUDIO_
OVERLAP` -- because D-215's own `decide_transition` never recommends an
advanced mode without a caller-supplied candidate, and none is offered.
This is the CORRECT, by-design behavior (this task's own "do not force
J/L/micro-overlap just to demonstrate the mode" instruction), not a gap.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from .contracts import DraftClip
from .dialogue_pacing_transition import (
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
)
from .pacing_transition_decision import (
    DECISION_CONFLICTED,
    DECISION_UNKNOWN,
    decide_transition,
    dialogue_pacing_transition_decision_run_summary,
    sequence_consistency_diagnostics,
)

SCHEMA_VERSION = "cutsell.pacing_v2_live_diagnostics_integration.v1"
_DIAGNOSTICS_ENV = "CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED"

CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"
CAPABILITY_DISABLED = "DISABLED"

MISSING_FEWER_THAN_TWO_SELECTED = "FEWER_THAN_TWO_SELECTED_CLIPS"

PROSODIC_STATUS_AVAILABLE = "AVAILABLE"
PROSODIC_STATUS_UNAVAILABLE = "UNAVAILABLE"

TIMING_PROPOSED = "TIMING_PROPOSED"
TIMING_NOT_PROPOSED = "TIMING_NOT_PROPOSED"

COMPARISON_AGREEMENT = "AGREEMENT"
COMPARISON_V2_MORE_CONSERVATIVE = "V2_MORE_CONSERVATIVE"
COMPARISON_V2_WOULD_USE_ADVANCED_MODE = "V2_WOULD_USE_ADVANCED_MODE"
COMPARISON_INCOMPARABLE = "INCOMPARABLE"

_ADVANCED_MODES = (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)


def _env_true_default_false(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def pacing_v2_diagnostics_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Default OFF. When OFF, nothing in this module is ever called by
    `universal_clean_cut.py` -- zero V2 decision compute, D-142's own live
    output stays byte-identical. There is no authority flag anywhere in
    this module or its caller."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


def _classify_comparison(live_mode: str, v2_mode: str, decision_status: str) -> str:
    if decision_status in (DECISION_UNKNOWN, DECISION_CONFLICTED):
        return COMPARISON_INCOMPARABLE
    if v2_mode == live_mode:
        return COMPARISON_AGREEMENT
    if v2_mode in _ADVANCED_MODES and live_mode not in _ADVANCED_MODES:
        return COMPARISON_V2_WOULD_USE_ADVANCED_MODE
    return COMPARISON_V2_MORE_CONSERVATIVE


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id)


def build_pacing_v2_live_diagnostics(
    selected: Sequence[DraftClip],
    *,
    dialogue_overlap_enabled: bool,
    boundary_diagnostics: Optional[Mapping] = None,
    live_transition_modes: Sequence[str] = (),
    prosody_by_clip_id: Optional[Mapping[str, object]] = None,
    relationship_hint_by_pair: Optional[Mapping[tuple[str, str], str]] = None,
    candidate_timing_by_pair: Optional[Mapping[tuple[str, str], Mapping[str, float]]] = None,
) -> dict:
    """Build one D-215 `decide_transition` result per REAL adjacent pair in
    `selected` (already-Boundary-finalized, already in final output order --
    never re-ordered here). `live_transition_modes[i]` (optional), when
    given, is D-142's own already-computed live mode for join `i` -- read
    from `draft.diagnostics["dialogue_pacing_transition"]["transitions"]`
    by the caller, NEVER recomputed here -- used only for the HARD/TIGHT
    comparison classification. `prosody_by_clip_id`/`relationship_hint_by_
    pair`/`candidate_timing_by_pair` are optional, caller-supplied, BY-
    REFERENCE evidence -- this module builds none of them (see module
    docstring's own "live evidence-availability finding")."""
    clips = tuple(selected)
    boundary_diagnostics = boundary_diagnostics or {}
    prosody_by_clip_id = prosody_by_clip_id or {}
    relationship_hint_by_pair = relationship_hint_by_pair or {}
    candidate_timing_by_pair = candidate_timing_by_pair or {}

    if len(clips) < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "capability_status": CAPABILITY_NOT_EVALUABLE,
            "missing_evidence": (MISSING_FEWER_THAN_TWO_SELECTED,),
            "transition_count": 0,
            "transitions": (),
            "run_summary": dialogue_pacing_transition_decision_run_summary(()),
            "sequence_consistency": sequence_consistency_diagnostics(()),
            "firewall_violation_count": 0,
        }

    plans = []
    rows = []
    any_prosody = False
    any_missing_prosody = False
    any_timing = False
    any_missing_timing = False

    for index in range(len(clips) - 1):
        left, right = clips[index], clips[index + 1]
        key = _pair_key(left.clip_id, right.clip_id)

        left_prosody = prosody_by_clip_id.get(left.clip_id)
        right_prosody = prosody_by_clip_id.get(right.clip_id)
        prosody_available = left_prosody is not None or right_prosody is not None
        any_prosody = any_prosody or prosody_available
        any_missing_prosody = any_missing_prosody or not prosody_available

        relationship_hint = relationship_hint_by_pair.get(key)

        timing = candidate_timing_by_pair.get(key) or {}
        candidate_lead = timing.get("lead")
        candidate_tail = timing.get("tail")
        timing_proposed = candidate_lead is not None or candidate_tail is not None
        any_timing = any_timing or timing_proposed
        any_missing_timing = any_missing_timing or not timing_proposed

        plan = decide_transition(
            left, right,
            dialogue_overlap_enabled=dialogue_overlap_enabled,
            boundary_diagnostics=boundary_diagnostics,
            left_prosody=left_prosody, right_prosody=right_prosody,
            candidate_audio_lead_sec=candidate_lead, candidate_audio_tail_sec=candidate_tail,
            relationship_hint=relationship_hint,
            transition_index=index,
        )
        plans.append(plan)

        live_mode = live_transition_modes[index] if index < len(live_transition_modes) else None
        comparison = (
            _classify_comparison(live_mode, plan.mode, plan.decision_status)
            if live_mode is not None else COMPARISON_INCOMPARABLE
        )
        overlap_violation = (
            plan.mode in _ADVANCED_MODES and (
                plan.meaning_safety_status == "BLOCKED"
                or plan.word_safety_status == "BLOCKED"
                or plan.double_speech_status in ("NO_OVERLAP_REQUIRED", "CONFLICTED")
            )
        )

        rows.append({
            "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
            "left_source_asset_id": left.source_asset_id, "right_source_asset_id": right.source_asset_id,
            "selected_mode": plan.mode,
            "pacing_gap_decision": plan.pacing_gap_decision,
            "speech_overlap_status": plan.speech_overlap_status,
            "meaning_safety_status": plan.meaning_safety_status,
            "word_safety_status": plan.word_safety_status,
            "double_speech_status": plan.double_speech_status,
            "candidate_audio_lead": candidate_lead,
            "candidate_audio_tail": candidate_tail,
            "candidate_overlap": plan.overlap_duration if plan.mode in _ADVANCED_MODES else None,
            "decision_status": plan.decision_status,
            "fallback_reason": plan.fallback_reason,
            "prosodic_status": PROSODIC_STATUS_AVAILABLE if prosody_available else PROSODIC_STATUS_UNAVAILABLE,
            "candidate_timing_status": TIMING_PROPOSED if timing_proposed else TIMING_NOT_PROPOSED,
            "relationship_hint": relationship_hint,
            "conflict_flags": plan.conflict_flags,
            "provenance": plan.provenance,
            "live_mode": live_mode,
            "live_vs_v2_comparison": comparison,
            "firewall_violation": overlap_violation,
        })

    # AVAILABLE whenever decisions were computed at all (always true once we
    # reach here -- D-215 always produces a safe decision even with zero
    # optional evidence). PARTIAL only when optional evidence (Prosody/
    # candidate timing) was present for SOME pairs but not others -- a
    # genuinely mixed batch (not observed at today's live seam, where both
    # are universally absent -- see module docstring -- but supported for
    # a future gate where partial evidence coverage is real).
    mixed_prosody = any_prosody and any_missing_prosody
    mixed_timing = any_timing and any_missing_timing
    capability_status = CAPABILITY_PARTIAL if (mixed_prosody or mixed_timing) else CAPABILITY_AVAILABLE

    return {
        "schema_version": SCHEMA_VERSION,
        "capability_status": capability_status,
        "missing_evidence": (),
        "transition_count": len(rows),
        "transitions": rows,
        "run_summary": dialogue_pacing_transition_decision_run_summary(tuple(plans)),
        "sequence_consistency": sequence_consistency_diagnostics(tuple(plans)),
        "firewall_violation_count": sum(1 for row in rows if row["firewall_violation"]),
        "prosodic_available_count": sum(1 for row in rows if row["prosodic_status"] == PROSODIC_STATUS_AVAILABLE),
        "prosodic_unavailable_count": sum(1 for row in rows if row["prosodic_status"] == PROSODIC_STATUS_UNAVAILABLE),
        "relationship_hint_count": sum(1 for row in rows if row["relationship_hint"]),
        "candidate_timing_available_count": sum(1 for row in rows if row["candidate_timing_status"] == TIMING_PROPOSED),
        "candidate_timing_missing_count": sum(1 for row in rows if row["candidate_timing_status"] == TIMING_NOT_PROPOSED),
        "comparison_agreement_count": sum(1 for row in rows if row["live_vs_v2_comparison"] == COMPARISON_AGREEMENT),
        "comparison_more_conservative_count": sum(1 for row in rows if row["live_vs_v2_comparison"] == COMPARISON_V2_MORE_CONSERVATIVE),
        "comparison_advanced_mode_count": sum(1 for row in rows if row["live_vs_v2_comparison"] == COMPARISON_V2_WOULD_USE_ADVANCED_MODE),
        "comparison_incomparable_count": sum(1 for row in rows if row["live_vs_v2_comparison"] == COMPARISON_INCOMPARABLE),
    }
