"""D-215 -- PACING V2 TRANSITION DECISION FOUNDATION. OFFLINE ONLY.

Canonical order (unchanged, D-098 Section 11 / D-129 / D-142 / D-213):

    Selection / BestTake -> Selection Freeze -> Boundary -> Dialogue/Pacing
    Transition (`dialogue_pacing_transition.py`, live, HARD_CUT/TIGHT_CUT
    only) -> Renderer (`render_plan.py`/`render.py`, D-214's own extended
    execution capability, not live-wired) -> export.

THIS MODULE answers the question D-213/D-214 named next: for two adjacent,
already Boundary-finalized segments, which transition MODE is safe and
appropriate -- `HARD_CUT`/`TIGHT_CUT`/`J_CUT`/`L_CUT`/`MICRO_AUDIO_OVERLAP`
-- using only evidence that already exists (word timings, Boundary's own
diagnostics, D-038's own claim-criticality classifier, D-187's own
Prosodic evidence). It is a DECISION layer, not an execution layer (D-214
already proved the renderer CAN realize whatever this module decides) and
not a live layer (nothing here is imported by `universal_clean_cut.py`,
`pipeline.py`, or the live `plan_dialogue_pacing_transitions` -- Phase 1's
own live modes remain exactly `HARD_CUT`/`TIGHT_CUT`, unchanged).

## Boundary firewall (never reopened)

`left.start`/`left.end`/`right.start`/`right.end` are read-only inputs
here -- this module never returns a new source trim, only a TIMELINE
PLACEMENT / MODE decision. It reuses the LIVE `plan_dialogue_pacing_
transitions` (D-142) itself, unmodified, to obtain the baseline HARD_CUT/
TIGHT_CUT determination (Boundary's own already-applied-trim attribution)
before layering J/L/overlap eligibility on top -- never a second,
divergent HARD_CUT/TIGHT_CUT implementation.

## No new detection, no new thresholds

Meaning-criticality reuses `semantic_claims.classify_claim` (D-038)
VERBATIM -- no new negation/number/correction detector. Prosodic evidence,
where supplied, is `prosodic_audio_v2.ProsodicDeliveryEvidence` (D-187)
VERBATIM -- this module never calls `analyze_prosodic_delivery` itself
(that would require real decoded audio samples, a provider-free but still
audio-dependent step out of this offline task's scope); it only ever
CONSUMES an already-built evidence object the caller supplies, or `None`
when unavailable (D-187/D-188's own "Prosodic must be optional" contract,
restated here). No candidate lead/tail/overlap DURATION is ever invented
by this module -- every candidate offset is an explicit caller/fixture
input (`candidate_audio_lead_sec`/`candidate_audio_tail_sec`); this module
only ever judges whether a GIVEN candidate is safe, never proposes one.

## Double-speech vs. meaning vs. word safety (three distinct firewalls)

- WORD SAFETY: a candidate window must never fall inside a spoken word.
  Checked against the widest word-timing evidence the caller has (defaults
  to the clip's own `.words`; a caller who has broader source-context word
  timings may supply them via `left_words`/`right_words` for a stronger
  proof).
- DOUBLE-SPEECH SAFETY: within the portion of the timeline where BOTH
  sides' audio would actually sound at once (the "double-speech window"),
  does EITHER side have real spoken words there? If both do, that is
  genuine competing speech (`CONFLICTED`/`NO_OVERLAP_REQUIRED`); if only
  one does, that is the safe, ordinary shape a J-cut/L-cut IS (one voice
  leads/trails into the other side's silence); if neither does, it is a
  trivially safe (if editorially pointless) micro-overlap of silence.
- MEANING SAFETY: whichever words DO fall inside the relevant window are
  classified via `classify_claim`; a `CRITICAL` claim type there (D-038's
  own negation/correction/number/diagnosis/etc. vocabulary) blocks the
  candidate regardless of double-speech status -- a critical word losing
  its ordinary audio/video pairing is itself a risk, independent of
  whether a second voice is competing with it.

## Abstention

Every decision carries `decision_status` (`SUPPORTED`/`SAFE_FALLBACK`/
`CONFLICTED`/`UNKNOWN`). `HARD_CUT` is always the physical fallback mode
when uncertain, but `decision_status` never collapses to `SUPPORTED` just
because the physical mode happens to be `HARD_CUT` -- per this task's own
"do not pretend HARD_CUT means editorial certainty" instruction.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Sequence, Tuple

from .contracts import DraftClip, Word
from .dialogue_pacing_transition import (
    DialogueTransitionPlan,
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
    plan_dialogue_pacing_transitions,
)
from .semantic_claims import CRITICAL, classify_claim

try:  # D-187 -- optional, type-only; never called here (see module docstring).
    from .prosodic_audio_v2 import ProsodicDeliveryEvidence
except ImportError:  # pragma: no cover -- defensive only, D-187 always ships alongside this module.
    ProsodicDeliveryEvidence = None  # type: ignore[assignment]

SCHEMA_VERSION = "cutsell.pacing_transition_decision.v1"

# --- Gap-decision vocabulary (D-215) ----------------------------------------
GAP_KEEP_PAUSE = "KEEP_PAUSE"
GAP_TIGHTEN = "TIGHTEN"
GAP_OVERLAP = "OVERLAP"
GAP_UNKNOWN = "UNKNOWN"

# --- Double-speech firewall vocabulary (D-215) ------------------------------
DOUBLE_SPEECH_SAFE_NO_OVERLAP = "SAFE_NO_OVERLAP"
DOUBLE_SPEECH_SAFE_J_CUT = "SAFE_J_CUT"
DOUBLE_SPEECH_SAFE_L_CUT = "SAFE_L_CUT"
DOUBLE_SPEECH_SAFE_MICRO_OVERLAP = "SAFE_MICRO_OVERLAP"
DOUBLE_SPEECH_NO_OVERLAP_REQUIRED = "NO_OVERLAP_REQUIRED"
DOUBLE_SPEECH_CONFLICTED = "CONFLICTED"
DOUBLE_SPEECH_UNKNOWN = "UNKNOWN"

# --- Coarse overlap-presence mirror (derived from the FINAL selected mode) --
SPEECH_OVERLAP_NONE = "NO_OVERLAP"
SPEECH_OVERLAP_PRESENT = "OVERLAP_PRESENT"
SPEECH_OVERLAP_STATUS_UNKNOWN = "UNKNOWN"

# --- Safety verdicts (word/meaning firewalls) -------------------------------
SAFETY_SAFE = "SAFE"
SAFETY_BLOCKED = "BLOCKED"
SAFETY_UNKNOWN = "UNKNOWN"

# --- Eligibility verdicts (per-mode) ----------------------------------------
ELIGIBLE = "ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
ELIGIBILITY_UNKNOWN = "UNKNOWN"

# --- Decision status (abstention contract) ----------------------------------
DECISION_SUPPORTED = "SUPPORTED"
DECISION_SAFE_FALLBACK = "SAFE_FALLBACK"
DECISION_CONFLICTED = "CONFLICTED"
DECISION_UNKNOWN = "UNKNOWN"

# --- Relationship hints (upstream evidence this module never computes) -----
RELATIONSHIP_CORRECTION = "correction"
RELATIONSHIP_CONTINUATION = "continuation"
RELATIONSHIP_RETRY = "retry"

# --- Conflict-flag vocabulary (closed set, named reasons only) -------------
CONFLICT_WORD_TIMING_MISSING = "word_timing_missing"
CONFLICT_WORD_SAFETY_BLOCKED = "word_safety_blocked"
CONFLICT_MEANING_CRITICAL = "meaning_critical_span"
CONFLICT_DOUBLE_SPEECH = "double_speech_conflict"
CONFLICT_CORRECTION_RELATIONSHIP = "correction_relationship"
CONFLICT_RETRY_RELATIONSHIP = "retry_relationship"
CONFLICT_OVERLAP_DISABLED = "overlap_disabled"
CONFLICT_NO_CANDIDATE_OFFERED = "no_candidate_offered"


def _word_window_overlap(words: Sequence[Word], window: Tuple[float, float]) -> Tuple[Word, ...]:
    w_start, w_end = window
    return tuple(w for w in words if float(w.start) < w_end and float(w.end) > w_start)


def _word_safety(words: Optional[Sequence[Word]], window: Optional[Tuple[float, float]]) -> str:
    """SAFE unless `window` provably crosses a real word; UNKNOWN when no
    word timing is available to check against (per this task's own "if
    timing is insufficient: fallback" instruction)."""
    if window is None:
        return SAFETY_SAFE
    if not words:  # None OR an empty sequence both mean "no ASR alignment for this clip"
        return SAFETY_UNKNOWN
    return SAFETY_BLOCKED if _word_window_overlap(words, window) else SAFETY_SAFE


def _meaning_safety(words: Optional[Sequence[Word]], window: Optional[Tuple[float, float]]) -> Tuple[str, Tuple[str, ...]]:
    """Classify whichever words fall inside `window` via the existing D-038
    `classify_claim` -- a CRITICAL claim type there blocks the candidate.
    No new detector; the exact same function D-038/D-040 already use."""
    if window is None or not words:
        return SAFETY_SAFE, ()
    overlapping = _word_window_overlap(words, window)
    if not overlapping:
        return SAFETY_SAFE, ()
    text = " ".join(str(w.text) for w in overlapping)
    claim_type, importance, _evidence = classify_claim(text)
    if importance == CRITICAL:
        return SAFETY_BLOCKED, (claim_type,)
    return SAFETY_SAFE, ()


def _prosody_supports_overlap(prosody: Optional["ProsodicDeliveryEvidence"]) -> Optional[bool]:
    """Transition-local Prosodic corroboration ONLY (D-187's own vocal_
    continuity_state/restart_or_interruption_state/hesitation_state) --
    never a BestTake ranking signal (D-188's own separate, unrelated
    consumer). Returns True (supports a tight/overlap join), False (a
    restart/hesitation/fragmented edge argues against one), or None
    (no prosodic evidence available -- always optional, never required)."""
    if prosody is None:
        return None
    if getattr(prosody, "restart_or_interruption_state", None) not in (None, "UNKNOWN", ""):
        state = prosody.restart_or_interruption_state
        if state and state not in ("NONE", "UNKNOWN"):
            return False
    if getattr(prosody, "vocal_continuity_state", None) == "FRAGMENTED":
        return False
    if getattr(prosody, "vocal_continuity_state", None) == "CONTINUOUS":
        return True
    return None


def _double_speech_window(
    left: DraftClip, right: DraftClip, *, lead: float, tail: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """The portion of each side's OWN source-time span that would sound
    DURING the overlap, plus the overlap's own duration -- geometry only,
    reused verbatim from D-214's own placement math (never a new timing
    model): a lead of `lead` sec on the right and a tail of `tail` sec on
    the left overlap for `min(lead, tail)` sec (or, if only one is
    offered, exactly that one -- the ordinary J-cut/L-cut single-sided
    shape)."""
    if lead > 0 and tail > 0:
        overlap_sec = min(lead, tail)
    elif lead > 0:
        overlap_sec = lead
    elif tail > 0:
        overlap_sec = tail
    else:
        overlap_sec = 0.0
    left_window = (max(left.start, left.end - overlap_sec), left.end)
    right_window = (right.start, min(right.end, right.start + overlap_sec))
    return left_window, right_window, overlap_sec


def decide_transition(
    left: DraftClip,
    right: DraftClip,
    *,
    dialogue_overlap_enabled: bool,
    boundary_diagnostics: Optional[Mapping] = None,
    left_words: Optional[Sequence[Word]] = None,
    right_words: Optional[Sequence[Word]] = None,
    left_prosody: Optional["ProsodicDeliveryEvidence"] = None,
    right_prosody: Optional["ProsodicDeliveryEvidence"] = None,
    candidate_audio_lead_sec: Optional[float] = None,
    candidate_audio_tail_sec: Optional[float] = None,
    relationship_hint: Optional[str] = None,
    transition_index: int = 0,
) -> DialogueTransitionPlan:
    """The ONE D-215 decision entry point. `left`/`right` are two adjacent,
    already Boundary-finalized `DraftClip`s (their `.start`/`.end` are
    NEVER read as mutable here). `left_words`/`right_words` default to
    each clip's own `.words`; a caller with broader source-context word
    timings may supply a wider sequence for a stronger word-safety proof.
    `candidate_audio_lead_sec`/`candidate_audio_tail_sec` are the ONLY
    numeric timing inputs this module ever looks at, and it never invents
    them -- a caller (test fixture, or a future D-216 live integration)
    offers a candidate; this function only ever judges its safety."""
    baseline = plan_dialogue_pacing_transitions(
        (left, right), boundary_diagnostics or {}, dialogue_overlap_enabled=dialogue_overlap_enabled,
    )[0]
    lw = left.words if left_words is None else tuple(left_words)
    rw = right.words if right_words is None else tuple(right_words)

    lead = float(candidate_audio_lead_sec) if candidate_audio_lead_sec is not None else 0.0
    tail = float(candidate_audio_tail_sec) if candidate_audio_tail_sec is not None else 0.0
    candidate_offered = candidate_audio_lead_sec is not None or candidate_audio_tail_sec is not None

    # --- Relationship-hint hard gates (upstream evidence, never computed here) ---
    if relationship_hint == RELATIONSHIP_RETRY:
        return _fallback_plan(
            baseline, decision_status=DECISION_CONFLICTED,
            fallback_reason=CONFLICT_RETRY_RELATIONSHIP, conflict_flags=(CONFLICT_RETRY_RELATIONSHIP,),
            speech_overlap_status=SPEECH_OVERLAP_NONE, double_speech_status=DOUBLE_SPEECH_UNKNOWN,
            meaning_safety_status=SAFETY_UNKNOWN, word_safety_status=SAFETY_UNKNOWN,
            gap_decision=GAP_UNKNOWN,
        )

    if not dialogue_overlap_enabled or not candidate_offered:
        reason = CONFLICT_OVERLAP_DISABLED if not dialogue_overlap_enabled else CONFLICT_NO_CANDIDATE_OFFERED
        gap_decision = _keep_pause_or_tighten(left, baseline, lw)
        return _fallback_plan(
            baseline, decision_status=DECISION_SUPPORTED,
            fallback_reason=(reason if not dialogue_overlap_enabled else None),
            conflict_flags=((reason,) if not dialogue_overlap_enabled else ()),
            speech_overlap_status=SPEECH_OVERLAP_NONE, double_speech_status=DOUBLE_SPEECH_SAFE_NO_OVERLAP,
            meaning_safety_status=SAFETY_SAFE, word_safety_status=SAFETY_SAFE,
            gap_decision=gap_decision,
        )

    if relationship_hint == RELATIONSHIP_CORRECTION:
        gap_decision = _keep_pause_or_tighten(left, baseline, lw)
        return _fallback_plan(
            baseline, decision_status=DECISION_SAFE_FALLBACK,
            fallback_reason=CONFLICT_CORRECTION_RELATIONSHIP, conflict_flags=(CONFLICT_CORRECTION_RELATIONSHIP,),
            speech_overlap_status=SPEECH_OVERLAP_NONE, double_speech_status=DOUBLE_SPEECH_NO_OVERLAP_REQUIRED,
            meaning_safety_status=SAFETY_SAFE, word_safety_status=SAFETY_SAFE,
            gap_decision=gap_decision,
        )

    # --- Word safety on the pre-cut lead/tail windows ---------------------
    left_window = (left.end, left.end + tail) if tail > 0 else None
    right_window = (right.start - lead, right.start) if lead > 0 else None
    left_word_safety = _word_safety(lw, left_window)
    right_word_safety = _word_safety(rw, right_window)
    word_conflicts = []
    if left_word_safety != SAFETY_SAFE:
        word_conflicts.append(left_word_safety)
    if right_word_safety != SAFETY_SAFE:
        word_conflicts.append(right_word_safety)
    word_safety_status = (
        SAFETY_UNKNOWN if SAFETY_UNKNOWN in word_conflicts
        else SAFETY_BLOCKED if SAFETY_BLOCKED in word_conflicts
        else SAFETY_SAFE
    )

    # --- Double-speech geometry --------------------------------------------
    dbl_left_window, dbl_right_window, overlap_sec = _double_speech_window(left, right, lead=lead, tail=tail)
    left_has_words = bool(_word_window_overlap(lw, dbl_left_window)) if lw else None
    right_has_words = bool(_word_window_overlap(rw, dbl_right_window)) if rw else None

    if lead > 0 and tail > 0:
        target_mode = MICRO_AUDIO_OVERLAP
        safe_label = DOUBLE_SPEECH_SAFE_MICRO_OVERLAP
    elif lead > 0:
        target_mode = J_CUT
        safe_label = DOUBLE_SPEECH_SAFE_J_CUT
    elif tail > 0:
        target_mode = L_CUT
        safe_label = DOUBLE_SPEECH_SAFE_L_CUT
    else:
        target_mode = baseline.mode
        safe_label = DOUBLE_SPEECH_SAFE_NO_OVERLAP

    # Real double speech = BOTH sides have actual words sounding at once
    # during the overlap window -- competing intelligible content, blocked
    # regardless of which shape was requested (per this task's own "if
    # both speakers/segments have required lexical content occupying the
    # same timeline interval: NO_OVERLAP_REQUIRED" instruction).
    if left_has_words is None or right_has_words is None:
        double_speech_status = DOUBLE_SPEECH_UNKNOWN
    elif left_has_words and right_has_words:
        double_speech_status = DOUBLE_SPEECH_NO_OVERLAP_REQUIRED
    else:
        double_speech_status = safe_label

    # --- Meaning safety on whatever the double-speech window contains ------
    meaning_left_status, left_claims = _meaning_safety(lw, dbl_left_window if left_has_words else left_window)
    meaning_right_status, right_claims = _meaning_safety(rw, dbl_right_window if right_has_words else right_window)
    meaning_safety_status = (
        SAFETY_UNKNOWN if SAFETY_UNKNOWN in (meaning_left_status, meaning_right_status)
        else SAFETY_BLOCKED if SAFETY_BLOCKED in (meaning_left_status, meaning_right_status)
        else SAFETY_SAFE
    )

    # --- Prosodic corroboration (optional, never required, never overriding a real block) ---
    prosody_signal_left = _prosody_supports_overlap(left_prosody)
    prosody_signal_right = _prosody_supports_overlap(right_prosody)
    prosody_blocks = prosody_signal_left is False or prosody_signal_right is False
    prosody_supports = prosody_signal_left is True or prosody_signal_right is True

    conflict_flags: list[str] = []
    if word_safety_status == SAFETY_UNKNOWN:
        conflict_flags.append(CONFLICT_WORD_TIMING_MISSING)
    elif word_safety_status == SAFETY_BLOCKED:
        conflict_flags.append(CONFLICT_WORD_SAFETY_BLOCKED)
    if meaning_safety_status == SAFETY_BLOCKED:
        conflict_flags.append(CONFLICT_MEANING_CRITICAL)
    if double_speech_status in (DOUBLE_SPEECH_CONFLICTED, DOUBLE_SPEECH_NO_OVERLAP_REQUIRED):
        conflict_flags.append(CONFLICT_DOUBLE_SPEECH)
    if prosody_blocks:
        conflict_flags.append("prosodic_restart_or_discontinuity")

    unknown_evidence = word_safety_status == SAFETY_UNKNOWN or meaning_safety_status == SAFETY_UNKNOWN or double_speech_status == DOUBLE_SPEECH_UNKNOWN
    blocked = (
        word_safety_status == SAFETY_BLOCKED
        or meaning_safety_status == SAFETY_BLOCKED
        or double_speech_status in (DOUBLE_SPEECH_CONFLICTED, DOUBLE_SPEECH_NO_OVERLAP_REQUIRED)
        or prosody_blocks
    )

    if unknown_evidence and not blocked:
        eligibility = ELIGIBILITY_UNKNOWN
        decision_status = DECISION_UNKNOWN
        selected_mode = baseline.mode
        fallback_reason = "insufficient_evidence"
    elif blocked:
        eligibility = NOT_ELIGIBLE
        decision_status = DECISION_SAFE_FALLBACK
        selected_mode = baseline.mode
        fallback_reason = conflict_flags[0] if conflict_flags else "blocked"
    else:
        eligibility = ELIGIBLE
        decision_status = DECISION_SUPPORTED
        selected_mode = target_mode
        fallback_reason = None

    speech_overlap_status = (
        SPEECH_OVERLAP_PRESENT if selected_mode in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
        else SPEECH_OVERLAP_NONE if selected_mode in (HARD_CUT, TIGHT_CUT)
        else SPEECH_OVERLAP_STATUS_UNKNOWN
    )
    gap_decision = (
        GAP_OVERLAP if selected_mode in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
        else GAP_TIGHTEN if selected_mode == TIGHT_CUT
        else _keep_pause_or_tighten(left, baseline, lw)
    )

    provenance = tuple(sorted(set(baseline.provenance) | {"pacing_transition_decision_v1", "semantic_claims"}
                              | ({"prosodic_audio_v2"} if (left_prosody is not None or right_prosody is not None) else set())))

    return replace(
        baseline,
        transition_index=transition_index,
        mode=selected_mode,
        overlap_duration=round(overlap_sec, 3) if selected_mode in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP) else baseline.overlap_duration,
        fallback_reason=fallback_reason,
        provenance=provenance,
        pacing_gap_decision=gap_decision,
        speech_overlap_status=speech_overlap_status,
        double_speech_status=double_speech_status,
        meaning_safety_status=meaning_safety_status,
        word_safety_status=word_safety_status,
        decision_status=decision_status,
        conflict_flags=tuple(conflict_flags),
    )


def _keep_pause_or_tighten(left: DraftClip, baseline: DialogueTransitionPlan, left_words: Optional[Sequence[Word]]) -> str:
    """KEEP_PAUSE when Boundary left a real inter-segment pause AND the
    left clip's own trailing content is itself meaning-critical (D-038
    reused, no new pause-duration threshold -- pause GEOMETRY, not an
    aesthetic ideal, decides whether there IS a pause at all; the baseline
    TIGHT_CUT/HARD_CUT mode -- Boundary's own already-applied-trim
    attribution -- decides whether one remains)."""
    if baseline.mode == TIGHT_CUT:
        return GAP_TIGHTEN
    if baseline.gap_removed_duration > 0.0:
        return GAP_TIGHTEN
    if left_words:
        _claim_type, importance, _evidence = classify_claim(str(left.text or ""))
        if importance == CRITICAL:
            return GAP_KEEP_PAUSE
    if not left_words:
        return GAP_UNKNOWN
    return GAP_TIGHTEN


def _fallback_plan(
    baseline: DialogueTransitionPlan, *, decision_status: str, fallback_reason: Optional[str],
    conflict_flags: Tuple[str, ...], speech_overlap_status: str, double_speech_status: str,
    meaning_safety_status: str, word_safety_status: str, gap_decision: str,
) -> DialogueTransitionPlan:
    provenance = tuple(sorted(set(baseline.provenance) | {"pacing_transition_decision_v1"}))
    return replace(
        baseline,
        fallback_reason=fallback_reason,
        provenance=provenance,
        pacing_gap_decision=gap_decision,
        speech_overlap_status=speech_overlap_status,
        double_speech_status=double_speech_status,
        meaning_safety_status=meaning_safety_status,
        word_safety_status=word_safety_status,
        decision_status=decision_status,
        conflict_flags=conflict_flags,
    )


def sequence_consistency_diagnostics(plans: Sequence[DialogueTransitionPlan]) -> dict:
    """A tiny, bounded, DIAGNOSTIC-ONLY pass -- never a global pacing
    optimizer (this task's own explicit boundary). Flags only the single
    obviously-contradictory shape this task names: two adjacent
    transitions sharing the identical `pacing_gap_decision` inputs
    (same `gap_removed_duration`, same `mode`) that nonetheless produced
    a DIFFERENT `pacing_gap_decision` -- a same-evidence, different-
    outcome inconsistency. Never rewrites a plan; only reports."""
    rows = tuple(plans)
    flags: list[dict] = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if (
            a.mode == b.mode
            and abs(a.gap_removed_duration - b.gap_removed_duration) < 1e-9
            and a.pacing_gap_decision != b.pacing_gap_decision
            and a.pacing_gap_decision is not None and b.pacing_gap_decision is not None
        ):
            flags.append({
                "left_index": a.transition_index, "right_index": b.transition_index,
                "reason": "same_evidence_different_gap_decision",
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(rows),
        "inconsistent_pair_count": len(flags),
        "inconsistent_pairs": flags,
    }


def dialogue_pacing_transition_decision_run_summary(plans: Sequence[DialogueTransitionPlan]) -> dict:
    """Fixture/batch-evaluation summary -- counts only, no master score."""
    rows = tuple(plans)
    mode_counts = {HARD_CUT: 0, TIGHT_CUT: 0, J_CUT: 0, L_CUT: 0, MICRO_AUDIO_OVERLAP: 0}
    gap_counts = {GAP_KEEP_PAUSE: 0, GAP_TIGHTEN: 0, GAP_OVERLAP: 0, GAP_UNKNOWN: 0}
    decision_counts = {DECISION_SUPPORTED: 0, DECISION_SAFE_FALLBACK: 0, DECISION_CONFLICTED: 0, DECISION_UNKNOWN: 0}
    meaning_block_count = 0
    word_safety_block_count = 0
    double_speech_block_count = 0
    prosodic_support_count = 0
    no_overlap_required_count = 0
    for row in rows:
        if row.mode in mode_counts:
            mode_counts[row.mode] += 1
        if row.pacing_gap_decision in gap_counts:
            gap_counts[row.pacing_gap_decision] += 1
        if row.decision_status in decision_counts:
            decision_counts[row.decision_status] += 1
        if row.meaning_safety_status == SAFETY_BLOCKED:
            meaning_block_count += 1
        if row.word_safety_status == SAFETY_BLOCKED:
            word_safety_block_count += 1
        if row.double_speech_status == DOUBLE_SPEECH_CONFLICTED:
            double_speech_block_count += 1
        if row.double_speech_status == DOUBLE_SPEECH_NO_OVERLAP_REQUIRED:
            no_overlap_required_count += 1
        if "prosodic_audio_v2" in row.provenance:
            prosodic_support_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_count": len(rows),
        "hard_cut_count": mode_counts[HARD_CUT],
        "tight_cut_count": mode_counts[TIGHT_CUT],
        "keep_pause_count": gap_counts[GAP_KEEP_PAUSE],
        "j_cut_count": mode_counts[J_CUT],
        "l_cut_count": mode_counts[L_CUT],
        "micro_overlap_count": mode_counts[MICRO_AUDIO_OVERLAP],
        "no_overlap_required_count": no_overlap_required_count,
        "fallback_count": decision_counts[DECISION_SAFE_FALLBACK],
        "conflicted_count": decision_counts[DECISION_CONFLICTED],
        "unknown_count": decision_counts[DECISION_UNKNOWN],
        "meaning_block_count": meaning_block_count,
        "word_safety_block_count": word_safety_block_count,
        "double_speech_block_count": double_speech_block_count,
        "prosodic_support_count": prosodic_support_count,
    }
