"""D-232 -- Pacing V2 Audio Join Treatment Decision Foundation. OFFLINE ONLY.

Chooses a treatment RECOMMENDATION (never executes it) from D-231's
`AudioJoinUnderstanding` -- D-098 Section 16's Layer 3 (AUDIO JOIN
TREATMENT), the SECOND, independent axis alongside Layer 2 (Transition
Decision, D-215, unchanged, owns `KEEP_PAUSE`/`HARD_CUT`/`TIGHT_CUT`/
`J_CUT`/`L_CUT`/`MICRO_AUDIO_OVERLAP`).

## Canonical position (restated, not renumbered)

    EVIDENCE
        v
    JOIN UNDERSTANDING          (D-231, upstream, unchanged)
        v
    TRANSITION DECISION          (D-215, SEPARATE AXIS, unchanged --
        v                        this module reads its OUTPUT mode as
                                 an input, never recomputes it)
    TIMING POLICY                 (D-220 for J/L; a future, separate
        v                        audio-treatment timing policy --
                                 NOT implemented here, see "Timing
                                 policy" below)
    AUDIO JOIN TREATMENT          <- this module (decision only)
        v
    RENDERER EXECUTION            (D-214, downstream, never called here)

This module owns ONLY the secondary Audio Join Treatment axis. It never
chooses/recomputes `KEEP_PAUSE`/`HARD_CUT`/`TIGHT_CUT`/`J_CUT`/`L_CUT`/
`MICRO_AUDIO_OVERLAP` -- `primary_transition_mode` is always a caller-
supplied, already-decided D-215 value, read-only.

## What this module explicitly does NOT do

Does not execute a crossfade, ambience layer, or any renderer window/mix
/fade -- `treatment` is a RECOMMENDATION string, never itself applied.
Does not choose `MICRO_AUDIO_OVERLAP` (impossible by construction: it is
not a member of `TREATMENT_VALUES`, the Layer-2 vocabulary this module
never touches). Does not invent a production treatment duration --
`candidate_duration_sec` is `None` unless a caller explicitly supplies an
already-bounded value; `timing_status` honestly reports `TIMING_POLICY_
NOT_YET_IMPLEMENTED` whenever it is `None` (this task's own "Timing
policy remains future work" instruction; a future, separate D-233 gate
owns it). Does not correct loudness or normalize gain (`loudness_polish_
status` OBSERVES a level/character mismatch, restated from D-231/D-230's
own loudness-ownership firewall, never corrects it -- D-024/Audio Polish
territory). Does not implement a room-tone classifier (`room_tone_
classification_status` surfaces D-230's own honest `"NOT_YET_AVAILABLE"`
verbatim). Does not mutate Boundary/Ordering/Family/BestTake. Has no
live wiring, no feature flag, is not called by `universal_clean_cut.py`.

## Evidence reuse (no fourth engine)

Every field this module reads comes directly from a `AudioJoinUnderstanding`
(D-231) instance -- `join_audio_role`, `understanding_status`, `acoustic_
continuity_status`, `level_continuity_status`, `word_safety_status`,
`meaning_safety_status`, `double_speech_status`, the four `*_evidence_
status` treatment-readiness fields, `no_treatment_evidence_status`,
`relationship_hint`, `prosodic_status`, `room_tone_classification_status`
-- never re-derived. `left_post_roll_handle_status`/`right_pre_roll_
handle_status` are optional caller-supplied D-223 `SourceAudioHandle.
handle_status` strings, passed straight through (this task's own "Reuse
existing D-223 SourceAudioHandle. Do not construct new handles"
instruction) -- purely diagnostic here; the actual ambience-readiness
GATE was already decided by D-231's own `ambience_carry_left/right_
evidence_status` fields, never re-evaluated from the raw handle here.

## Renderer capability status, preserved verbatim (D-229's own finding)

`RENDERER_CAPABILITY_SHORT_CROSSFADE = "EXTENSION_REQUIRED"`,
`RENDERER_CAPABILITY_AMBIENCE_CARRY_LEFT/RIGHT/BRIDGE = "SUPPORTED_NOW"`
-- these are D-229's own forensic findings (docs/CUTSELL_DECISIONS.md
D-229, items 30-34), restated here as named constants because this
module's own diagnostics surface them per-decision; they describe the
RENDERER's mechanical capability, never authorize execution, and are
never modified by any decision this module makes.

## Compatibility matrix (Layer 2 x Layer 3, per D-098 Section 16.6)

`_COMPATIBILITY` is a complete, static, deterministic mapping over
every `(primary_transition_mode, treatment)` pair this track's own six
Layer-2 modes and six Layer-3 treatments can form (36 pairs) --
`SUPPORTED`/`POSSIBLE_WITH_CONDITIONS`/`DISALLOWED`. `MICRO_AUDIO_
OVERLAP` paired with any `AMBIENCE_*` treatment is `DISALLOWED`
(this task's own explicit "D-232 must never select MICRO" instruction,
plus the CONTENT-ROLE distinction Section 16's own "Ambience Bridge vs
Micro Overlap" section draws: `MICRO_AUDIO_OVERLAP` is a dialogue/audio
transition mode with its own speech-overlap semantics; `AMBIENCE_*`
treatments are non-lexical-only by definition -- mutually exclusive by
CONTENT ROLE, not geometry). A candidate treatment that compatibility
disallows for the given `primary_transition_mode` is downgraded to
`CLICK_FADE`/`SAFE_FALLBACK` before being returned -- the compatibility
check is load-bearing in the real decision path, not an accessory.

## Firewalls (restated, not weakened)

- **Speech firewall**: `LEXICAL_SPEECH_PRESENT` on a side where a
  treatment would need to overlap that speech fails closed (folded into
  `speech_safety_status`, below).
- **Word firewall**: unknown word coverage never becomes a safe overlap
  (`understanding.word_safety_status == SAFETY_UNKNOWN` blocks advanced
  treatment identically to `SAFETY_BLOCKED`).
- **Meaning firewall**: `understanding.meaning_safety_status ==
  SAFETY_BLOCKED` blocks every treatment beyond `CLICK_FADE`/`NONE`
  (D-038's negation/numbers/factual-qualifier/correction/critical-clause
  vocabulary, reused via D-231, never re-classified here).
- **Double-speech firewall**: `JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL`
  blocks every treatment beyond `CLICK_FADE`/`NONE` unless the
  treatment is strictly non-lexical (`AMBIENCE_*`) AND the relevant
  side's own ambience-readiness is independently `TREATMENT_EVIDENCE_
  READY` -- folded into `speech_safety_status`.
- **Retry/Correction firewall**: `RELATIONSHIP_RETRY`/`RELATIONSHIP_
  CORRECTION` hard-veto every smoothing treatment (`SHORT_CROSSFADE`/
  `AMBIENCE_*`), restating D-215's own identical veto at Layer 2 --
  never re-derived, read straight from `understanding.relationship_
  hint`. `RELATIONSHIP_CONTINUATION` may support smoothing but is never
  itself sufficient (this task's own explicit instruction) -- it never
  appears as a condition anywhere in this module's own decision table.
- **Loudness ownership firewall**: `loudness_polish_status` OBSERVES,
  restated from D-230/D-231, never corrects.

## Timing policy status

`candidate_duration_sec` stays `None` and `timing_status` stays
`TIMING_POLICY_NOT_YET_IMPLEMENTED` for every decision this module ever
produces today -- no live caller populates `candidate_duration_sec`,
and this module never computes one. Treatment SELECTION is fully
separable from DURATION (this task's own explicit requirement) -- a
future, separately-authorized D-233 gate owns the timing question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .dialogue_pacing_transition import (
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
)
from .pacing_transition_decision import (
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_UNKNOWN,
)
from .pacing_v2_acoustic_edge_evidence import (
    CONTINUITY_DIFFERENT,
    CONTINUITY_SIMILAR,
    EVIDENCE_STATUS_CONFLICTED,
    EVIDENCE_STATUS_INSUFFICIENT,
    EVIDENCE_STATUS_SAFE_FALLBACK,
    EVIDENCE_STATUS_SUPPORTED,
    EVIDENCE_STATUS_UNKNOWN,
)
from .pacing_v2_audio_join_understanding import (
    ACOUSTIC_CONTINUITY_STATUS_UNKNOWN,
    AudioJoinUnderstanding,
    JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL,
    JOIN_DOUBLE_SPEECH_STATUS_UNKNOWN,
    JOIN_ROLE_CONFLICTED,
    LEVEL_CONTINUITY_INSUFFICIENT,
    LEVEL_CONTINUITY_LEFT_LOUDER,
    LEVEL_CONTINUITY_RIGHT_LOUDER,
    NO_TREATMENT_EVIDENCE,
    TREATMENT_EVIDENCE_READY,
    UNDERSTANDING_CONFLICTED,
    UNDERSTANDING_NOT_EVALUABLE,
)

# `KEEP_PAUSE` lives in `pacing_transition_decision.py` (D-215's own
# gap-decision vocabulary, GAP_KEEP_PAUSE) rather than `dialogue_pacing_
# transition.py`'s own primary-mode set -- imported separately, never
# redefined, per this task's own "restate D-215's own vocabulary,
# never a second one" convention.
from .pacing_transition_decision import GAP_KEEP_PAUSE as KEEP_PAUSE

SCHEMA_VERSION = "cutsell.pacing_v2_audio_join_treatment_decision.v1"

# ---------------------------------------------------------------------------
# Treatment vocabulary -- the FIRST module in this track to define it
# (D-231's own test suite explicitly proved it defines none). Exact
# canonical values per D-098 Section 16.5, no variants.
# ---------------------------------------------------------------------------
TREATMENT_NONE = "NONE"
TREATMENT_CLICK_FADE = "CLICK_FADE"
TREATMENT_SHORT_CROSSFADE = "SHORT_CROSSFADE"
TREATMENT_AMBIENCE_CARRY_LEFT = "AMBIENCE_CARRY_LEFT"
TREATMENT_AMBIENCE_CARRY_RIGHT = "AMBIENCE_CARRY_RIGHT"
TREATMENT_AMBIENCE_BRIDGE = "AMBIENCE_BRIDGE"
TREATMENT_VALUES = (
    TREATMENT_NONE, TREATMENT_CLICK_FADE, TREATMENT_SHORT_CROSSFADE,
    TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_AMBIENCE_BRIDGE,
)

# ---------------------------------------------------------------------------
# Treatment-status vocabulary -- reuses D-230's own EVIDENCE_STATUS_*
# values verbatim (exact same five-value shape the directive names),
# never a second definition of the same concept.
# ---------------------------------------------------------------------------
TREATMENT_STATUS_SUPPORTED = EVIDENCE_STATUS_SUPPORTED
TREATMENT_STATUS_SAFE_FALLBACK = EVIDENCE_STATUS_SAFE_FALLBACK
TREATMENT_STATUS_INSUFFICIENT_EVIDENCE = EVIDENCE_STATUS_INSUFFICIENT
TREATMENT_STATUS_CONFLICTED = EVIDENCE_STATUS_CONFLICTED
TREATMENT_STATUS_UNKNOWN = EVIDENCE_STATUS_UNKNOWN

# ---------------------------------------------------------------------------
# Compatibility-verdict vocabulary.
# ---------------------------------------------------------------------------
COMPATIBILITY_SUPPORTED = "SUPPORTED"
COMPATIBILITY_POSSIBLE_WITH_CONDITIONS = "POSSIBLE_WITH_CONDITIONS"
COMPATIBILITY_DISALLOWED = "DISALLOWED"
COMPATIBILITY_UNKNOWN = "UNKNOWN"

_PRIMARY_MODES = (KEEP_PAUSE, HARD_CUT, TIGHT_CUT, J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
_AMBIENCE_TREATMENTS = (TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_AMBIENCE_BRIDGE)

# Complete, static, deterministic Layer-2 x Layer-3 matrix (this task's
# own explicit requirement: "make precedence/compatibility deterministic").
_COMPATIBILITY = {
    (HARD_CUT, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (HARD_CUT, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (HARD_CUT, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_SUPPORTED,
    (HARD_CUT, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (HARD_CUT, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (HARD_CUT, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,

    (TIGHT_CUT, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (TIGHT_CUT, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (TIGHT_CUT, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_SUPPORTED,
    (TIGHT_CUT, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (TIGHT_CUT, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (TIGHT_CUT, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,

    (KEEP_PAUSE, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (KEEP_PAUSE, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (KEEP_PAUSE, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (KEEP_PAUSE, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (KEEP_PAUSE, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (KEEP_PAUSE, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,

    (J_CUT, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (J_CUT, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (J_CUT, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (J_CUT, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (J_CUT, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (J_CUT, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,

    (L_CUT, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (L_CUT, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (L_CUT, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (L_CUT, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (L_CUT, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,
    (L_CUT, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_POSSIBLE_WITH_CONDITIONS,

    (MICRO_AUDIO_OVERLAP, TREATMENT_NONE): COMPATIBILITY_SUPPORTED,
    (MICRO_AUDIO_OVERLAP, TREATMENT_CLICK_FADE): COMPATIBILITY_SUPPORTED,
    (MICRO_AUDIO_OVERLAP, TREATMENT_SHORT_CROSSFADE): COMPATIBILITY_DISALLOWED,
    (MICRO_AUDIO_OVERLAP, TREATMENT_AMBIENCE_CARRY_LEFT): COMPATIBILITY_DISALLOWED,
    (MICRO_AUDIO_OVERLAP, TREATMENT_AMBIENCE_CARRY_RIGHT): COMPATIBILITY_DISALLOWED,
    (MICRO_AUDIO_OVERLAP, TREATMENT_AMBIENCE_BRIDGE): COMPATIBILITY_DISALLOWED,
}


def compatibility(primary_transition_mode: str, treatment: str) -> str:
    """The one compatibility-lookup entry point. Never over-authorizes:
    an unrecognized primary mode or treatment value returns `UNKNOWN`,
    never `SUPPORTED`."""
    return _COMPATIBILITY.get((primary_transition_mode, treatment), COMPATIBILITY_UNKNOWN)


# ---------------------------------------------------------------------------
# Renderer capability status (D-229's own forensic findings, restated
# verbatim -- never modified by any decision this module makes).
# ---------------------------------------------------------------------------
RENDERER_CAPABILITY_CLICK_FADE = "EXISTING_LIVE_UNCHANGED"
RENDERER_CAPABILITY_SHORT_CROSSFADE = "EXTENSION_REQUIRED"
RENDERER_CAPABILITY_AMBIENCE_CARRY_LEFT = "SUPPORTED_NOW"
RENDERER_CAPABILITY_AMBIENCE_CARRY_RIGHT = "SUPPORTED_NOW"
RENDERER_CAPABILITY_AMBIENCE_BRIDGE = "SUPPORTED_NOW"
RENDERER_CAPABILITY_NONE = "NOT_APPLICABLE"
_RENDERER_CAPABILITY_BY_TREATMENT = {
    TREATMENT_NONE: RENDERER_CAPABILITY_NONE,
    TREATMENT_CLICK_FADE: RENDERER_CAPABILITY_CLICK_FADE,
    TREATMENT_SHORT_CROSSFADE: RENDERER_CAPABILITY_SHORT_CROSSFADE,
    TREATMENT_AMBIENCE_CARRY_LEFT: RENDERER_CAPABILITY_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT: RENDERER_CAPABILITY_AMBIENCE_CARRY_RIGHT,
    TREATMENT_AMBIENCE_BRIDGE: RENDERER_CAPABILITY_AMBIENCE_BRIDGE,
}


def renderer_capability_status(treatment: str) -> str:
    return _RENDERER_CAPABILITY_BY_TREATMENT.get(treatment, RENDERER_CAPABILITY_NONE)


# ---------------------------------------------------------------------------
# Timing-policy status vocabulary.
# ---------------------------------------------------------------------------
TIMING_POLICY_NOT_YET_IMPLEMENTED = "TIMING_POLICY_NOT_YET_IMPLEMENTED"
TIMING_POLICY_CANDIDATE_SUPPLIED = "CANDIDATE_SUPPLIED_BY_CALLER"

# ---------------------------------------------------------------------------
# Loudness-polish diagnostic vocabulary (observation only, never a fix).
# ---------------------------------------------------------------------------
LOUDNESS_POLISH_NEEDED = "LOUDNESS_POLISH_NEEDED"
LOUDNESS_POLISH_NOT_NEEDED = "LOUDNESS_POLISH_NOT_NEEDED"
LOUDNESS_POLISH_STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Conflict-flag vocabulary.
# ---------------------------------------------------------------------------
CONFLICT_SPEECH_SAFETY_BLOCKED = "speech_safety_blocked"
CONFLICT_MEANING_SAFETY_BLOCKED = "meaning_safety_blocked"
CONFLICT_DOUBLE_SPEECH_BLOCKED = "double_speech_blocked"
CONFLICT_RETRY_OR_CORRECTION_VETO = "retry_or_correction_veto"
CONFLICT_UNDERSTANDING_CONFLICTED = "join_understanding_conflicted"
CONFLICT_UNDERSTANDING_NOT_EVALUABLE = "join_understanding_not_evaluable"
CONFLICT_COMPATIBILITY_DISALLOWED = "compatibility_disallowed_downgrade"

PROVENANCE_AUDIO_JOIN_UNDERSTANDING = "audio_join_understanding_reused"
PROVENANCE_PRIMARY_TRANSITION_MODE = "primary_transition_mode_caller_supplied"
PROVENANCE_SOURCE_AUDIO_HANDLE_STATUS = "source_audio_handle_status_reused"

_SMOOTHING_TREATMENTS = (TREATMENT_SHORT_CROSSFADE,) + _AMBIENCE_TREATMENTS


@dataclass(frozen=True)
class AudioJoinTreatmentDecision:
    """One adjacent join's Audio Join Treatment RECOMMENDATION. Never an
    execution -- see module docstring for the full "does not" list. No
    transcript dump, no master score anywhere on this type."""

    schema_version: str
    transition_index: int
    left_clip_id: str
    right_clip_id: str

    primary_transition_mode: str

    treatment: str
    treatment_status: str
    treatment_reason: Optional[str]

    left_audio_role: str
    right_audio_role: str

    acoustic_continuity_status: str
    level_continuity_status: str

    speech_safety_status: str
    word_safety_status: str
    meaning_safety_status: str
    double_speech_status: str

    left_handle_status: Optional[str]
    right_handle_status: Optional[str]

    candidate_duration_sec: Optional[float]
    timing_status: str

    compatibility_status: str
    renderer_capability_status: str
    loudness_polish_status: str
    room_tone_classification_status: str

    fallback_reason: Optional[str]

    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _speech_safety_status(understanding: AudioJoinUnderstanding) -> str:
    """Aggregate speech-related firewall verdict -- combines word safety
    and double-speech evidence (both already computed by D-231) into the
    ONE gate this module's decision precedence checks first. Distinct
    from `word_safety_status` (join-local, edge-only) -- this field also
    folds in the double-speech observation, per this task's own "If
    required speech exists on both sides... fail closed" instruction."""
    if understanding.word_safety_status == SAFETY_UNKNOWN or \
            understanding.double_speech_status == JOIN_DOUBLE_SPEECH_STATUS_UNKNOWN:
        return SAFETY_UNKNOWN
    if understanding.word_safety_status == SAFETY_BLOCKED or \
            understanding.double_speech_status == JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL:
        return SAFETY_BLOCKED
    return SAFETY_SAFE


def _loudness_polish_status(understanding: AudioJoinUnderstanding) -> str:
    """Observation only -- see module docstring's loudness-ownership
    firewall. Needed exactly when the two edges share acoustic character
    (SIMILAR) but differ audibly in level -- the "ACOUSTIC CHARACTER !=
    LOUDNESS" case this task's own directive names."""
    if understanding.acoustic_continuity_status == CONTINUITY_SIMILAR and understanding.level_continuity_status in (
        LEVEL_CONTINUITY_LEFT_LOUDER, LEVEL_CONTINUITY_RIGHT_LOUDER,
    ):
        return LOUDNESS_POLISH_NEEDED
    if understanding.level_continuity_status == LEVEL_CONTINUITY_INSUFFICIENT or \
            understanding.acoustic_continuity_status == ACOUSTIC_CONTINUITY_STATUS_UNKNOWN:
        return LOUDNESS_POLISH_STATUS_UNKNOWN
    return LOUDNESS_POLISH_NOT_NEEDED


def _decide_treatment(
    understanding: AudioJoinUnderstanding, *, speech_safety_status: str,
) -> Tuple[str, str, Optional[str]]:
    """The one, exhaustive, priority-ordered decision table (this task's
    own recommended precedence, kept -- current evidence does not argue
    for a different order). Returns (treatment, treatment_status,
    treatment_reason). Never returns MICRO_AUDIO_OVERLAP (not a member
    of TREATMENT_VALUES, structurally impossible)."""
    # Step 0: no evidence at all -- true abstention, never a forced guess.
    if understanding.understanding_status == UNDERSTANDING_NOT_EVALUABLE:
        return TREATMENT_NONE, TREATMENT_STATUS_INSUFFICIENT_EVIDENCE, CONFLICT_UNDERSTANDING_NOT_EVALUABLE

    # Step 1: semantic/word/firewall conflict -> CLICK_FADE fallback
    # (always safe, purely technical, never touches meaning).
    if understanding.understanding_status == UNDERSTANDING_CONFLICTED or understanding.join_audio_role == JOIN_ROLE_CONFLICTED:
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_CONFLICTED, CONFLICT_UNDERSTANDING_CONFLICTED
    if understanding.meaning_safety_status == SAFETY_BLOCKED:
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_SAFE_FALLBACK, CONFLICT_MEANING_SAFETY_BLOCKED
    if speech_safety_status == SAFETY_BLOCKED:
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_SAFE_FALLBACK, CONFLICT_SPEECH_SAFETY_BLOCKED
    # Retry/Correction hard veto on every smoothing treatment (restates
    # D-215's own identical veto at Layer 2 -- never re-derived).
    if understanding.relationship_hint in (RELATIONSHIP_RETRY, RELATIONSHIP_CORRECTION):
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_SAFE_FALLBACK, CONFLICT_RETRY_OR_CORRECTION_VETO

    # Step 2: join already continuous -> NONE, a first-class success.
    if understanding.no_treatment_evidence_status == NO_TREATMENT_EVIDENCE:
        return TREATMENT_NONE, TREATMENT_STATUS_SUPPORTED, None

    # Step 3: acoustic discontinuity + retained-edge treatment feasible
    # -> SHORT_CROSSFADE. Requires speech/meaning safety fully RESOLVED
    # (not merely absent-of-a-known-block) since a crossfade touches the
    # retained edges directly -- an unresolved side must not silently
    # become eligible.
    if speech_safety_status == SAFETY_SAFE and understanding.meaning_safety_status == SAFETY_SAFE and \
            understanding.acoustic_continuity_status == CONTINUITY_DIFFERENT and \
            understanding.short_crossfade_evidence_status == TREATMENT_EVIDENCE_READY:
        return TREATMENT_SHORT_CROSSFADE, TREATMENT_STATUS_SUPPORTED, None

    # Step 4/5: safe LEFT/RIGHT ambience handle + discontinuity. Deliberately
    # NOT gated on retained-edge speech/meaning safety being fully resolved
    # -- an ambience-carry treatment never touches the retained edges at
    # all (D-231's own ambience-readiness fields are handle-only, already
    # independently firewalled), so an unrelated retained-edge UNKNOWN
    # must never block a handle-evidenced, independently-safe candidate.
    if understanding.acoustic_continuity_status == CONTINUITY_DIFFERENT and \
            understanding.ambience_carry_left_evidence_status == TREATMENT_EVIDENCE_READY:
        return TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_STATUS_SUPPORTED, None
    if understanding.acoustic_continuity_status == CONTINUITY_DIFFERENT and \
            understanding.ambience_carry_right_evidence_status == TREATMENT_EVIDENCE_READY:
        return TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_STATUS_SUPPORTED, None

    # Step 6: safe non-speech bridge evidence + discontinuity.
    if understanding.ambience_bridge_evidence_status == TREATMENT_EVIDENCE_READY:
        return TREATMENT_AMBIENCE_BRIDGE, TREATMENT_STATUS_SUPPORTED, None

    # Abstention: no advanced treatment could be reached above. An
    # unresolved (UNKNOWN) safety signal is checked HERE, last -- never
    # early, so it can never pre-empt an independently-safe ambience
    # candidate the steps above already would have returned.
    if speech_safety_status == SAFETY_UNKNOWN or understanding.meaning_safety_status == SAFETY_UNKNOWN:
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_INSUFFICIENT_EVIDENCE, "safety_evidence_unknown"
    if understanding.acoustic_continuity_status in (CONTINUITY_SIMILAR,):
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_SUPPORTED, None
    if understanding.acoustic_continuity_status == ACOUSTIC_CONTINUITY_STATUS_UNKNOWN:
        return TREATMENT_CLICK_FADE, TREATMENT_STATUS_INSUFFICIENT_EVIDENCE, "acoustic_continuity_unknown"
    return TREATMENT_CLICK_FADE, TREATMENT_STATUS_SAFE_FALLBACK, "no_advanced_treatment_evidence_ready"


def build_audio_join_treatment_decision(
    understanding: AudioJoinUnderstanding,
    *,
    primary_transition_mode: str,
    left_post_roll_handle_status: Optional[str] = None,
    right_pre_roll_handle_status: Optional[str] = None,
    candidate_duration_sec: Optional[float] = None,
) -> AudioJoinTreatmentDecision:
    """The one D-232 entry point. `understanding` is a real D-231
    `AudioJoinUnderstanding`; `primary_transition_mode` is D-215's own
    already-decided value for this SAME join (read-only, never
    recomputed here). One treatment recommendation per join (no
    stacking, per this task's own "Phase A simple" instruction)."""
    speech_safety = _speech_safety_status(understanding)
    treatment, treatment_status, treatment_reason = _decide_treatment(
        understanding, speech_safety_status=speech_safety,
    )

    fallback_reason = treatment_reason if treatment != TREATMENT_NONE and treatment_status != TREATMENT_STATUS_SUPPORTED else (
        treatment_reason if treatment == TREATMENT_NONE and treatment_status != TREATMENT_STATUS_SUPPORTED else None
    )

    conflict_flags = list(understanding.conflict_flags)
    compat = compatibility(primary_transition_mode, treatment)
    if compat == COMPATIBILITY_DISALLOWED:
        # Downgrade -- compatibility is load-bearing, not an accessory.
        treatment = TREATMENT_CLICK_FADE
        treatment_status = TREATMENT_STATUS_SAFE_FALLBACK
        treatment_reason = CONFLICT_COMPATIBILITY_DISALLOWED
        fallback_reason = CONFLICT_COMPATIBILITY_DISALLOWED
        conflict_flags.append(CONFLICT_COMPATIBILITY_DISALLOWED)
        compat = compatibility(primary_transition_mode, treatment)

    if treatment_reason == CONFLICT_SPEECH_SAFETY_BLOCKED:
        conflict_flags.append(CONFLICT_SPEECH_SAFETY_BLOCKED)
    if treatment_reason == CONFLICT_MEANING_SAFETY_BLOCKED:
        conflict_flags.append(CONFLICT_MEANING_SAFETY_BLOCKED)
    if treatment_reason == CONFLICT_RETRY_OR_CORRECTION_VETO:
        conflict_flags.append(CONFLICT_RETRY_OR_CORRECTION_VETO)
    if treatment_reason == CONFLICT_UNDERSTANDING_CONFLICTED:
        conflict_flags.append(CONFLICT_UNDERSTANDING_CONFLICTED)
    if treatment_reason == CONFLICT_UNDERSTANDING_NOT_EVALUABLE:
        conflict_flags.append(CONFLICT_UNDERSTANDING_NOT_EVALUABLE)

    timing_status = TIMING_POLICY_CANDIDATE_SUPPLIED if candidate_duration_sec is not None else TIMING_POLICY_NOT_YET_IMPLEMENTED

    provenance: set = {SCHEMA_VERSION, PROVENANCE_AUDIO_JOIN_UNDERSTANDING, PROVENANCE_PRIMARY_TRANSITION_MODE}
    provenance.update(understanding.provenance)
    if left_post_roll_handle_status is not None or right_pre_roll_handle_status is not None:
        provenance.add(PROVENANCE_SOURCE_AUDIO_HANDLE_STATUS)

    return AudioJoinTreatmentDecision(
        schema_version=SCHEMA_VERSION,
        transition_index=understanding.transition_index,
        left_clip_id=understanding.left_clip_id, right_clip_id=understanding.right_clip_id,
        primary_transition_mode=primary_transition_mode,
        treatment=treatment, treatment_status=treatment_status, treatment_reason=treatment_reason,
        left_audio_role=understanding.left_non_speech_status, right_audio_role=understanding.right_non_speech_status,
        acoustic_continuity_status=understanding.acoustic_continuity_status,
        level_continuity_status=understanding.level_continuity_status,
        speech_safety_status=speech_safety,
        word_safety_status=understanding.word_safety_status,
        meaning_safety_status=understanding.meaning_safety_status,
        double_speech_status=understanding.double_speech_status,
        left_handle_status=left_post_roll_handle_status, right_handle_status=right_pre_roll_handle_status,
        candidate_duration_sec=candidate_duration_sec, timing_status=timing_status,
        compatibility_status=compat, renderer_capability_status=renderer_capability_status(treatment),
        loudness_polish_status=_loudness_polish_status(understanding),
        room_tone_classification_status=understanding.room_tone_classification_status,
        fallback_reason=fallback_reason,
        conflict_flags=tuple(conflict_flags),
        provenance=tuple(sorted(provenance)),
    )


def audio_join_treatment_decision_diagnostics(decision: AudioJoinTreatmentDecision) -> dict:
    """Compact, JSON-safe diagnostics row -- no transcript dump."""
    return {
        "schema_version": decision.schema_version,
        "transition_index": decision.transition_index,
        "left_clip_id": decision.left_clip_id, "right_clip_id": decision.right_clip_id,
        "primary_transition_mode": decision.primary_transition_mode,
        "treatment": decision.treatment, "treatment_status": decision.treatment_status,
        "treatment_reason": decision.treatment_reason,
        "left_audio_role": decision.left_audio_role, "right_audio_role": decision.right_audio_role,
        "acoustic_continuity_status": decision.acoustic_continuity_status,
        "level_continuity_status": decision.level_continuity_status,
        "speech_safety_status": decision.speech_safety_status,
        "word_safety_status": decision.word_safety_status,
        "meaning_safety_status": decision.meaning_safety_status,
        "double_speech_status": decision.double_speech_status,
        "left_handle_status": decision.left_handle_status, "right_handle_status": decision.right_handle_status,
        "candidate_duration_sec": decision.candidate_duration_sec, "timing_status": decision.timing_status,
        "compatibility_status": decision.compatibility_status,
        "renderer_capability_status": decision.renderer_capability_status,
        "loudness_polish_status": decision.loudness_polish_status,
        "room_tone_classification_status": decision.room_tone_classification_status,
        "fallback_reason": decision.fallback_reason,
        "conflict_flags": list(decision.conflict_flags),
        "provenance": list(decision.provenance),
    }


def audio_join_treatment_decision_run_summary(rows: Sequence[AudioJoinTreatmentDecision]) -> dict:
    """No master score -- plain counts only, per the directive's own
    named fields."""
    def _count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    return {
        "schema_version": SCHEMA_VERSION,
        "join_count": len(rows),
        "none_count": _count(lambda r: r.treatment == TREATMENT_NONE),
        "click_fade_count": _count(lambda r: r.treatment == TREATMENT_CLICK_FADE),
        "short_crossfade_count": _count(lambda r: r.treatment == TREATMENT_SHORT_CROSSFADE),
        "ambience_left_count": _count(lambda r: r.treatment == TREATMENT_AMBIENCE_CARRY_LEFT),
        "ambience_right_count": _count(lambda r: r.treatment == TREATMENT_AMBIENCE_CARRY_RIGHT),
        "ambience_bridge_count": _count(lambda r: r.treatment == TREATMENT_AMBIENCE_BRIDGE),
        "safe_fallback_count": _count(lambda r: r.treatment_status == TREATMENT_STATUS_SAFE_FALLBACK),
        "insufficient_evidence_count": _count(lambda r: r.treatment_status == TREATMENT_STATUS_INSUFFICIENT_EVIDENCE),
        "conflicted_count": _count(lambda r: r.treatment_status == TREATMENT_STATUS_CONFLICTED),
        "unknown_count": _count(lambda r: r.treatment_status == TREATMENT_STATUS_UNKNOWN),
        "acoustic_discontinuity_count": _count(lambda r: r.acoustic_continuity_status == CONTINUITY_DIFFERENT),
        "loudness_polish_needed_count": _count(lambda r: r.loudness_polish_status == LOUDNESS_POLISH_NEEDED),
        "word_block_count": _count(lambda r: r.word_safety_status == SAFETY_BLOCKED),
        "meaning_block_count": _count(lambda r: r.meaning_safety_status == SAFETY_BLOCKED),
        "double_speech_block_count": _count(lambda r: r.double_speech_status == JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL),
    }
