"""D-220 -- PACING V2 ADVANCED TRANSITION TIMING POLICY FOUNDATION.
OFFLINE ONLY.

D-219's own forensic finding: D-215/D-217's existing firewall chain
(word safety, meaning safety, double-speech safety, relationship-hint
vetoes, fail-closed-on-unknown) proves J_CUT/L_CUT are SAFE TO EXECUTE,
but `pacing_v2_evidence_adapter.candidate_timing_for_pair` deliberately
offers the FULL mechanically-available silent window as the candidate
lead/tail ("no fraction, no cap, no invented duration" by design) --
nothing decides how much of that window an EXECUTED transition should
actually use. AVAILABLE SAFE WINDOW != CHOSEN TRANSITION AMOUNT.

THIS MODULE answers that one question, for J_CUT/L_CUT only (MICRO_
AUDIO_OVERLAP remains diagnostics-only/deferred, per D-219 item 34 and
this task's own explicit instruction). It is a TIMING-AMOUNT layer, not
a mode-selection layer -- `pacing_transition_decision.decide_transition`
(D-215) already decided WHETHER a pair is J_CUT/L_CUT-eligible and WHY;
this module only ever runs on an ALREADY-ELIGIBLE pair and decides HOW
MUCH of the already-proven-safe window to request. It never reopens
eligibility, never re-derives word/meaning/double-speech safety, never
recomputes RETRY/CORRECTION vetoes (defensively re-asserted here, never
re-decided), and is never imported by `universal_clean_cut.py` or
`pipeline.py` -- no live authority exists anywhere in this module.

## No magic constant (binding design principle)

The chosen amount is bounded by TWO real, structurally-derived
quantities, never an invented millisecond constant:
  1. `max_safe_lead`/`max_safe_tail` -- the caller-supplied, already-
     proven-safe window (from `pacing_v2_evidence_adapter.available_
     silent_head_sec`/`available_silent_tail_sec` via D-217, or an
     equivalent caller-derived value; this module never recomputes it).
  2. An ANCHOR WORD's own measured duration -- for J_CUT, the LEFT
     clip's own FINAL spoken word (the departing line's last word sets
     the pace for how much silent lead-in follows); for L_CUT, the
     RIGHT clip's own FIRST spoken word (the incoming line's first word
     sets the pace for how much trailing silence from the previous line
     survives under it). Both are real, already-measured word timings,
     never invented durations.

`chosen_duration = min(max_safe_window, anchor_word_duration)`, refined
(never overridden) by optional Prosodic/relationship evidence (see
`decide_jcut_timing`/`decide_lcut_timing`). This overall POLICY
STRUCTURE (bound by real safe geometry AND a real anchor word, never an
arbitrary duration) is well-grounded; the SPECIFIC CHOICE of anchor word
is honestly a `QUALITY_HEURISTIC_NOT_YET_REAL_MEDIA_TUNED` design
decision (D-219's own required labeling) -- it has never been validated
against real perceptual/Watch+Listen evidence. `TIMING_BASIS_BOUNDED_
DEFAULT` and `TIMING_BASIS_PAUSE_GEOMETRY` are named in this module's
own closed vocabulary for completeness (this task's own directive names
them explicitly) but are NEVER produced by this version's logic -- no
separate numeric bounded-default fallback was judged necessary; when the
anchor word itself cannot be measured, this module fails closed
(`INSUFFICIENT_EVIDENCE`, `chosen_duration=None`) rather than inventing
one.

## No ML / provider

Deterministic only. No model call, no LLM, no learned timing predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from .contracts import DraftClip
from .dialogue_pacing_transition import J_CUT, L_CUT
from .pacing_transition_decision import (
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
)

try:  # D-187 -- optional, type-only; this module never calls analyze_prosodic_delivery.
    from .prosodic_audio_v2 import ProsodicDeliveryEvidence
except ImportError:  # pragma: no cover -- defensive only, D-187 always ships alongside this module.
    ProsodicDeliveryEvidence = None  # type: ignore[assignment]

SCHEMA_VERSION = "cutsell.pacing_v2_timing_policy.v1"

# --- Numeric safety floor -----------------------------------------------
# A pure floating-point safety epsilon, NOT an editorial constant -- kept
# numerically identical to `render.py`'s own `AUDIO_TIMELINE_EPSILON_SEC`
# (proven by `test_render_epsilon_matches_timing_policy_epsilon` in this
# task's own test file) without importing `render.py` itself, to avoid
# creating a new decision-layer -> renderer-layer dependency edge (this
# track's own existing layering: the renderer never imports the decision
# layer, and the decision layer never imports the renderer). A window at
# or below this floor is one D-214's own `_validate_independent_audio_
# window` would reject as `audio_end <= audio_start + epsilon` anyway --
# this module fails closed on it proactively rather than proposing a
# value the renderer would refuse.
RENDER_EPSILON_SEC = 1e-6

# --- Timing status vocabulary (closed, D-220) ---------------------------
TIMING_STATUS_SUPPORTED = "SUPPORTED"
TIMING_STATUS_SAFE_FALLBACK = "SAFE_FALLBACK"
TIMING_STATUS_INSUFFICIENT_WINDOW = "INSUFFICIENT_WINDOW"
TIMING_STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
TIMING_STATUS_CONFLICTED = "CONFLICTED"
TIMING_STATUS_UNKNOWN = "UNKNOWN"

# --- Timing basis vocabulary (closed, D-220) ----------------------------
TIMING_BASIS_WORD_GEOMETRY = "WORD_GEOMETRY"
TIMING_BASIS_PAUSE_GEOMETRY = "PAUSE_GEOMETRY"  # named for vocabulary completeness; never produced today.
TIMING_BASIS_PROSODIC_EDGE = "PROSODIC_EDGE"  # named for vocabulary completeness; never produced today.
TIMING_BASIS_WORD_PLUS_PROSODIC = "WORD_PLUS_PROSODIC"
TIMING_BASIS_BOUNDED_DEFAULT = "BOUNDED_DEFAULT"  # named for vocabulary completeness; never produced today.
TIMING_BASIS_NONE = "NONE"

_RETRY_OR_CORRECTION = (RELATIONSHIP_RETRY, RELATIONSHIP_CORRECTION)


@dataclass(frozen=True)
class AdvancedTransitionTimingDecision:
    """One pair's own D-220 timing-amount decision. Never a master score,
    never a global optimizer across pairs -- pairwise only, mirroring
    D-215's own `DialogueTransitionPlan` discipline."""

    schema_version: str = SCHEMA_VERSION
    transition_index: int = 0
    mode: str = ""
    max_safe_window: Optional[float] = None
    chosen_duration: Optional[float] = None
    timing_basis: str = TIMING_BASIS_NONE
    timing_status: str = TIMING_STATUS_UNKNOWN
    fallback_reason: Optional[str] = None
    # Tuple of (key, value) pairs, sorted by key -- deterministic, hashable,
    # frozen-dataclass-friendly (mirrors the tuple discipline `conflict_
    # flags`/`provenance` already use elsewhere in this Pacing V2 track).
    source_geometry: Tuple[Tuple[str, Optional[float]], ...] = ()
    prosodic_support: Optional[bool] = None
    provenance: Tuple[str, ...] = ()


def _geometry(**kwargs: Optional[float]) -> Tuple[Tuple[str, Optional[float]], ...]:
    return tuple(sorted(kwargs.items()))


def _last_word_duration(clip: DraftClip) -> Optional[float]:
    """The LAST spoken word's own measured duration -- `None` (UNKNOWN,
    never `0.0`) when the clip has no word timing at all. A genuinely NEW
    computation this module introduces (existing helpers compute room
    BETWEEN a clip edge and a word, never a single word's own span)."""
    if not clip.words:
        return None
    last = max(clip.words, key=lambda w: w.end)
    return max(0.0, float(last.end) - float(last.start))


def _first_word_duration(clip: DraftClip) -> Optional[float]:
    """The FIRST spoken word's own measured duration -- `None` (UNKNOWN,
    never `0.0`) when the clip has no word timing at all."""
    if not clip.words:
        return None
    first = min(clip.words, key=lambda w: w.start)
    return max(0.0, float(first.end) - float(first.start))


def _prosodic_continuity_state(prosody: Optional["ProsodicDeliveryEvidence"]) -> Optional[str]:
    if prosody is None:
        return None
    return getattr(prosody, "vocal_continuity_state", None)


def _decide(
    *,
    mode: str,
    transition_index: int,
    max_safe_window: Optional[float],
    anchor_word_duration_getter,
    geometry_extra: Mapping[str, Optional[float]],
    relationship_hint: Optional[str],
    prosody: Optional["ProsodicDeliveryEvidence"],
) -> AdvancedTransitionTimingDecision:
    geometry = _geometry(max_safe_window=max_safe_window, **geometry_extra)

    # --- Defensive re-assertion, never a re-decision (D-219: "D-220
    # should not reopen those decisions") -- RETRY/CORRECTION are already
    # vetoed by decide_transition upstream; a caller that (incorrectly)
    # still invokes this module on such a pair gets a clean refusal, not
    # a silently-computed number. ---
    if relationship_hint in _RETRY_OR_CORRECTION:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
            chosen_duration=None, timing_basis=TIMING_BASIS_NONE, timing_status=TIMING_STATUS_CONFLICTED,
            fallback_reason=f"{relationship_hint}_relationship_not_reopened",
            source_geometry=geometry, prosodic_support=None, provenance=("pacing_v2_timing_policy_v1",),
        )

    if max_safe_window is None:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=None,
            chosen_duration=None, timing_basis=TIMING_BASIS_NONE, timing_status=TIMING_STATUS_UNKNOWN,
            fallback_reason="no_candidate_timing_offered",
            source_geometry=geometry, prosodic_support=None, provenance=("pacing_v2_timing_policy_v1",),
        )

    if max_safe_window < 0:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
            chosen_duration=None, timing_basis=TIMING_BASIS_NONE, timing_status=TIMING_STATUS_INSUFFICIENT_WINDOW,
            fallback_reason="negative_window_rejected",
            source_geometry=geometry, prosodic_support=None, provenance=("pacing_v2_timing_policy_v1",),
        )

    if max_safe_window <= RENDER_EPSILON_SEC:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
            chosen_duration=None, timing_basis=TIMING_BASIS_NONE, timing_status=TIMING_STATUS_INSUFFICIENT_WINDOW,
            fallback_reason="window_below_render_epsilon",
            source_geometry=geometry, prosodic_support=None, provenance=("pacing_v2_timing_policy_v1",),
        )

    anchor = anchor_word_duration_getter()
    if anchor is None:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
            chosen_duration=None, timing_basis=TIMING_BASIS_NONE, timing_status=TIMING_STATUS_INSUFFICIENT_EVIDENCE,
            fallback_reason="anchor_word_timing_missing",
            source_geometry=geometry, prosodic_support=None, provenance=("pacing_v2_timing_policy_v1",),
        )

    structural_cap = min(max_safe_window, anchor)
    basis = TIMING_BASIS_WORD_GEOMETRY
    provenance = ["pacing_v2_timing_policy_v1"]
    prosodic_support: Optional[bool] = None

    continuity = _prosodic_continuity_state(prosody)
    if continuity is not None:
        provenance.append("prosodic_edge_evidence")
        if continuity == "CONTINUOUS":
            prosodic_support = True
            basis = TIMING_BASIS_WORD_PLUS_PROSODIC
            chosen = structural_cap
        else:
            # Prosodic evidence IS available but does not confirm clean
            # continuity -- real, present, negative-leaning evidence, so
            # this refines (shrinks) the amount conservatively; it never
            # overrides `max_safe_window`/word/meaning/double-speech
            # safety (those are already proven upstream, D-215).
            prosodic_support = False
            basis = TIMING_BASIS_WORD_PLUS_PROSODIC
            if relationship_hint == RELATIONSHIP_CONTINUATION:
                # CONTINUATION corroborates that this is a natural,
                # expected join -- offsets the Prosodic-uncertain shrink
                # rather than compounding a second, unrelated reduction.
                chosen = structural_cap
                provenance.append("continuation_relationship_support")
            else:
                chosen = structural_cap / 2.0
    else:
        # Prosodic evidence simply ABSENT -- per D-219's own "Prosodic
        # absence does not produce arbitrary duration" requirement, this
        # must resolve to the SAME deterministic word-geometry-only
        # answer as if Prosodic had never been asked about at all, never
        # a different (larger or smaller) number.
        chosen = structural_cap

    chosen = min(chosen, max_safe_window)  # re-assert the invariant explicitly, never trust arithmetic alone.
    if chosen <= RENDER_EPSILON_SEC:
        return AdvancedTransitionTimingDecision(
            transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
            chosen_duration=None, timing_basis=basis, timing_status=TIMING_STATUS_SAFE_FALLBACK,
            fallback_reason="structural_cap_below_render_epsilon",
            source_geometry=geometry, prosodic_support=prosodic_support, provenance=tuple(provenance),
        )

    return AdvancedTransitionTimingDecision(
        transition_index=transition_index, mode=mode, max_safe_window=max_safe_window,
        chosen_duration=round(chosen, 6), timing_basis=basis, timing_status=TIMING_STATUS_SUPPORTED,
        fallback_reason=None,
        source_geometry=geometry, prosodic_support=prosodic_support, provenance=tuple(provenance),
    )


def decide_jcut_timing(
    left: DraftClip,
    right: DraftClip,
    *,
    max_safe_lead: Optional[float],
    transition_index: int = 0,
    relationship_hint: Optional[str] = None,
    right_prosody: Optional["ProsodicDeliveryEvidence"] = None,
) -> AdvancedTransitionTimingDecision:
    """J_CUT ONLY, and only ever meaningfully called on a pair D-215's own
    `decide_transition` has ALREADY found eligible (`mode == J_CUT`,
    `decision_status == SUPPORTED`) -- this function never re-derives that
    eligibility, only the AMOUNT within `max_safe_lead` (the caller's own
    already-proven-safe window, e.g. from `pacing_v2_evidence_adapter.
    available_silent_head_sec(right)` -- never recomputed here).

    Anchor: `left`'s own FINAL spoken word duration (the departing line's
    own last word sets the pace for how much silent lead-in from `right`
    follows it) -- `QUALITY_HEURISTIC_NOT_YET_REAL_MEDIA_TUNED` (D-219's
    own required labeling; structurally derived, never an invented
    duration, but this specific anchor CHOICE is untuned)."""
    return _decide(
        mode=J_CUT,
        transition_index=transition_index,
        max_safe_window=max_safe_lead,
        anchor_word_duration_getter=lambda: _last_word_duration(left),
        geometry_extra={
            "left_last_word_end": (max(float(w.end) for w in left.words) if left.words else None),
            "right_first_word_start": (min(float(w.start) for w in right.words) if right.words else None),
            "left_clip_end": float(left.end),
            "right_clip_start": float(right.start),
        },
        relationship_hint=relationship_hint,
        prosody=right_prosody,
    )


def decide_lcut_timing(
    left: DraftClip,
    right: DraftClip,
    *,
    max_safe_tail: Optional[float],
    transition_index: int = 0,
    relationship_hint: Optional[str] = None,
    left_prosody: Optional["ProsodicDeliveryEvidence"] = None,
) -> AdvancedTransitionTimingDecision:
    """L_CUT ONLY, symmetric to `decide_jcut_timing`. Anchor: `right`'s own
    FIRST spoken word duration (the incoming line's own first word sets
    the pace for how much trailing silence from `left` survives under
    it) -- same `QUALITY_HEURISTIC_NOT_YET_REAL_MEDIA_TUNED` labeling."""
    return _decide(
        mode=L_CUT,
        transition_index=transition_index,
        max_safe_window=max_safe_tail,
        anchor_word_duration_getter=lambda: _first_word_duration(right),
        geometry_extra={
            "left_last_word_end": (max(float(w.end) for w in left.words) if left.words else None),
            "right_first_word_start": (min(float(w.start) for w in right.words) if right.words else None),
            "left_clip_end": float(left.end),
            "right_clip_start": float(right.start),
        },
        relationship_hint=relationship_hint,
        prosody=left_prosody,
    )


def timing_policy_run_summary(decisions: Sequence[AdvancedTransitionTimingDecision]) -> dict:
    """Fixture/batch-evaluation summary -- counts only, no master score,
    mirroring D-215's own `dialogue_pacing_transition_decision_run_
    summary` discipline."""
    rows = tuple(decisions)
    j_policy_count = sum(1 for r in rows if r.mode == J_CUT)
    l_policy_count = sum(1 for r in rows if r.mode == L_CUT)
    supported_timing_count = sum(1 for r in rows if r.timing_status == TIMING_STATUS_SUPPORTED)
    fallback_timing_count = sum(1 for r in rows if r.timing_status == TIMING_STATUS_SAFE_FALLBACK)
    insufficient_window_count = sum(1 for r in rows if r.timing_status == TIMING_STATUS_INSUFFICIENT_WINDOW)
    insufficient_evidence_count = sum(1 for r in rows if r.timing_status == TIMING_STATUS_INSUFFICIENT_EVIDENCE)
    prosodic_supported_count = sum(1 for r in rows if r.prosodic_support is True)
    geometry_only_count = sum(1 for r in rows if r.prosodic_support is None and r.timing_status == TIMING_STATUS_SUPPORTED)
    zero_duration_count = sum(1 for r in rows if r.chosen_duration is None)
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(rows),
        "j_policy_count": j_policy_count,
        "l_policy_count": l_policy_count,
        "supported_timing_count": supported_timing_count,
        "fallback_timing_count": fallback_timing_count,
        "insufficient_window_count": insufficient_window_count,
        "insufficient_evidence_count": insufficient_evidence_count,
        "prosodic_supported_count": prosodic_supported_count,
        "geometry_only_count": geometry_only_count,
        "zero_duration_count": zero_duration_count,
    }
