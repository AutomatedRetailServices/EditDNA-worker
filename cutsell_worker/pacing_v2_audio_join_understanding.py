"""D-231 -- Pacing V2 Audio Join Understanding Foundation. OFFLINE ONLY.

Combines already-existing evidence (D-230 acoustic edge evidence, D-223
`SourceAudioHandle`, D-217's relationship-hint/Prosodic evidence, D-038's
`classify_claim`) into ONE structured interpretation of a single adjacent
audio join, per D-098 Section 16.3's Layer 1 (JOIN UNDERSTANDING).

## Canonical position (restated from D-098 Section 16.2, not renumbered)

    EVIDENCE
        v
    JOIN UNDERSTANDING          <- this module
        v
    TRANSITION DECISION          (D-215 `pacing_transition_decision.py`,
        v                        unchanged, downstream, never called here)
    TIMING POLICY                 (D-220, downstream)
        v
    AUDIO JOIN TREATMENT          (future D-232, downstream)
        v
    RENDERER EXECUTION            (D-214, downstream)

This module owns JOIN UNDERSTANDING for AUDIO JOIN TREATMENT only. It
NEVER calls `pacing_transition_decision.decide_transition` (that is a
DOWNSTREAM authority reading DIFFERENT, mode-specific, candidate-window
evidence -- calling it here would be backwards in the canonical pipeline
shape) and never itself is called by it.

## What this module explicitly does NOT do

Does not choose or execute `SHORT_CROSSFADE`/`AMBIENCE_CARRY_LEFT`/
`AMBIENCE_CARRY_RIGHT`/`AMBIENCE_BRIDGE` -- the four `*_evidence_status`
readiness fields below mean "evidence exists to evaluate this treatment
LATER", never "apply this treatment now" (this task's own explicit
"if such booleans risk conflating evidence with authority, prefer
categorical readiness statuses" instruction -- hence a three-value
categorical status, never a boolean). Does not decide J_CUT/L_CUT/
MICRO_AUDIO_OVERLAP (D-215's own unchanged authority). Does not decide
crossfade/ambience/bridge DURATION (a future, separate timing policy,
D-098 Section 16.9). Does not execute a renderer window, mix, or fade.
Does not mutate Boundary/Ordering/Family/BestTake. Does not correct
loudness or normalize gain (D-230's own loudness-ownership firewall,
restated, not reopened). Does not implement a room-tone classifier
(`ROOM_TONE_CLASSIFICATION_STATUS` remains D-230's own honest
`"NOT_YET_AVAILABLE"`, surfaced here verbatim). Has no live wiring, no
feature flag, is not called by `universal_clean_cut.py`.

## Evidence reuse (no fourth engine, per this task's own instruction)

- Speech occupancy / non-speech / silence per edge: `pacing_v2_acoustic_
  edge_evidence.AcousticEdgeEvidence` (D-230), read directly (`speech_
  status`, `non_speech_status`, `silence_status`) -- never re-derived.
- Acoustic continuity: `pacing_v2_acoustic_edge_evidence.
  AcousticContinuityComparison.continuity_status` (D-230), read
  directly -- no second spectral comparator.
- Level continuity: `AcousticContinuityComparison.level_delta_db`
  (D-230, already gain-un-normalized RMS ratio in dB), categorized here
  via ONE new, explicitly labeled, bounded threshold (this task's own
  "no magic similarity score first" allowance, mirroring D-230's own
  `ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_*` naming). Never a
  gain correction.
- SourceAudioHandle availability: `pacing_v2_source_audio_handle.
  SourceAudioHandle` (D-223) objects consumed directly by id/status --
  never rebuilt.
- Relationship hint: the exact `RELATIONSHIP_CORRECTION`/`RELATIONSHIP_
  CONTINUATION`/`RELATIONSHIP_RETRY` values `pacing_transition_decision`
  (D-215) and `pacing_v2_evidence_adapter` (D-217) already define,
  passed straight through -- P1 is never recomputed here.
- Prosodic edge evidence: duck-typed exactly like `pacing_transition_
  decision._prosody_supports_overlap` (D-215) reads it (`restart_or_
  interruption_state`/`vocal_continuity_state` via `getattr`) -- no new
  Prosodic call, no BestTake ranking.
- Word/meaning safety vocabulary: `SAFETY_SAFE`/`SAFETY_BLOCKED`/
  `SAFETY_UNKNOWN` imported directly from `pacing_transition_decision`
  (D-215) -- the SAME closed vocabulary, not a redefinition. Meaning
  safety itself is computed via `semantic_claims.classify_claim`
  (D-038), the exact function `pacing_transition_decision._meaning_
  safety` already calls -- no duplicate semantic engine. Word safety
  is derived purely from the two join-adjacent `AcousticEdgeEvidence`
  objects' own already-computed `speech_status` -- no new word-timing
  scan.

## Word-safety vs. meaning-safety: a layer distinction, not a duplicate

At Layer 2 (D-215), `word_safety_status`/`meaning_safety_status` are
MODE-SPECIFIC: they judge a particular candidate lead/tail WINDOW no
timing policy has chosen yet at this layer. At Layer 1 (this module),
the equivalent fields are JOIN-LOCAL and mode-agnostic: they describe
whether protected content already sits at the join's own two retained
edges (D-230's `EDGE_LEFT_END`/`EDGE_RIGHT_START`), independent of
whatever candidate window a downstream Timing Policy might later offer.
Reusing the identical `SAFETY_*` vocabulary is deliberate (this task's
own "reuse existing meaning helpers" instruction) -- the two layers
describe genuinely different (but same-vocabulary) facts, exactly as
D-230's own edge/handle distinction already established one vocabulary
reused across two different evidence shapes.

## D-230 bug-fix contract, preserved (restated, not reopened)

D-230 fixed a genuine bug where evidence-QUALITY gaps (no audio
supplied, window too short) were being conflated with true FIREWALL
conflicts (discarded/retry/meaning-critical), wrongly downgrading a
known `LEXICAL_SPEECH_PRESENT` edge to `SPEECH_STATUS_UNKNOWN`-shaped
outcomes. This module reads `AcousticEdgeEvidence.speech_status`
directly (already correctly firewalled by D-230's own fix) and never
re-derives it, so that distinction cannot be re-collapsed here.

## Double-speech (join-local, new vocabulary, not `pacing_transition_
decision`'s own mode-specific one)

`pacing_transition_decision.DOUBLE_SPEECH_SAFE_J_CUT`/`SAFE_L_CUT`/
`SAFE_MICRO_OVERLAP`/etc. describe a SPECIFIC candidate overlap window's
safety verdict -- meaningless before a Timing Policy exists. This
module instead reports whether EACH side's own join-adjacent edge
independently carries known lexical speech (`JOIN_DOUBLE_SPEECH_*`,
below) -- a strictly weaker, mode-agnostic observation a future D-232
gate combines with actual candidate geometry, never a pre-computed
verdict for any specific mode. Naming it distinctly avoids exactly the
kind of geometric/semantic vocabulary collision D-229/D-230 already
found between `MICRO_AUDIO_OVERLAP` and `AMBIENCE_BRIDGE`.

## Room-tone honesty, restated

`ROOM_TONE_CLASSIFICATION_STATUS` is imported verbatim from D-230
(`"NOT_YET_AVAILABLE"`) and surfaced in every diagnostic row; this
module never emits `ROOM_TONE_MATCH`/`ROOM_TONE_MISMATCH` -- only the
already-existing `ACOUSTIC_CONTINUITY_SIMILAR`/`DIFFERENT`/etc. values.

## Pairwise only, no global optimizer

`build_audio_join_understanding` evaluates exactly ONE adjacent join.
A caller building a full sequence calls it once per adjacent pair (see
`audio_join_understanding_run_summary` for the batch-level counts-only
aggregation, no master score).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .pacing_v2_acoustic_edge_evidence import (
    AcousticContinuityComparison,
    AcousticEdgeEvidence,
    CONTINUITY_CONFLICTED,
    CONTINUITY_DIFFERENT,
    CONTINUITY_INSUFFICIENT,
    CONTINUITY_SIMILAR,
    EVIDENCE_STATUS_CONFLICTED,
    NON_SPEECH_SAFE_CANDIDATE,
    NON_SPEECH_SILENCE,
    NON_SPEECH_SPEECH_PRESENT,
    NON_SPEECH_UNCONFIRMED,
    NON_SPEECH_UNKNOWN,
    ROOM_TONE_CLASSIFICATION_STATUS,
    SILENCE_STATUS_UNKNOWN,
    SPEECH_STATUS_LEXICAL_PRESENT,
    SPEECH_STATUS_NO_LEXICAL_OBSERVED,
    SPEECH_STATUS_UNKNOWN,
)
from .pacing_transition_decision import (
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_UNKNOWN,
)
from .pacing_v2_source_audio_handle import (
    HANDLE_STATUS_SAFE_NON_SPEECH,
    SourceAudioHandle,
)
from .semantic_claims import CRITICAL, classify_claim

SCHEMA_VERSION = "cutsell.pacing_v2_audio_join_understanding.v1"

# Re-exported so a caller/test never has to reach into D-230 directly for
# the two relationship/safety vocabularies this module passes through.
__all__ = [
    "SCHEMA_VERSION",
    "AudioJoinUnderstanding",
    "build_audio_join_understanding",
    "audio_join_understanding_diagnostics",
    "audio_join_understanding_run_summary",
    "JOIN_ROLE_CLEAN_DIRECT_JOIN",
    "JOIN_ROLE_SPEECH_TO_SPEECH",
    "JOIN_ROLE_SPEECH_TO_NON_SPEECH",
    "JOIN_ROLE_NON_SPEECH_TO_SPEECH",
    "JOIN_ROLE_NON_SPEECH_TO_NON_SPEECH",
    "JOIN_ROLE_SILENCE_BOUNDARY",
    "JOIN_ROLE_ACOUSTIC_DISCONTINUITY",
    "JOIN_ROLE_AMBIGUOUS",
    "JOIN_ROLE_CONFLICTED",
    "JOIN_ROLE_UNKNOWN",
    "UNDERSTANDING_AVAILABLE",
    "UNDERSTANDING_PARTIAL",
    "UNDERSTANDING_CONFLICTED",
    "UNDERSTANDING_NOT_EVALUABLE",
    "UNDERSTANDING_UNKNOWN",
    "LEVEL_CONTINUITY_SIMILAR",
    "LEVEL_CONTINUITY_LEFT_LOUDER",
    "LEVEL_CONTINUITY_RIGHT_LOUDER",
    "LEVEL_CONTINUITY_INSUFFICIENT",
    "TREATMENT_EVIDENCE_READY",
    "TREATMENT_EVIDENCE_NOT_READY",
    "TREATMENT_EVIDENCE_UNKNOWN",
    "NO_TREATMENT_EVIDENCE",
    "ADDITIONAL_TREATMENT_EVIDENCE_PRESENT",
    "TREATMENT_NEED_UNKNOWN",
    "PROSODIC_JOIN_CONTINUOUS",
    "PROSODIC_JOIN_RESTART",
    "PROSODIC_JOIN_HESITATION",
    "PROSODIC_JOIN_UNAVAILABLE",
    "PROSODIC_JOIN_UNKNOWN",
    "JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL",
    "JOIN_DOUBLE_SPEECH_LEFT_ONLY_LEXICAL",
    "JOIN_DOUBLE_SPEECH_RIGHT_ONLY_LEXICAL",
    "JOIN_DOUBLE_SPEECH_NEITHER_LEXICAL",
    "JOIN_DOUBLE_SPEECH_STATUS_UNKNOWN",
    "ACOUSTIC_CONTINUITY_STATUS_UNKNOWN",
]

# ---------------------------------------------------------------------------
# Join Audio Role vocabulary (this task's own named ten, adjusted to
# actual evidence per the task's own "adjust based on actual evidence"
# instruction -- none removed, none renamed).
# ---------------------------------------------------------------------------
JOIN_ROLE_CLEAN_DIRECT_JOIN = "CLEAN_DIRECT_JOIN"
JOIN_ROLE_SPEECH_TO_SPEECH = "SPEECH_TO_SPEECH"
JOIN_ROLE_SPEECH_TO_NON_SPEECH = "SPEECH_TO_NON_SPEECH"
JOIN_ROLE_NON_SPEECH_TO_SPEECH = "NON_SPEECH_TO_SPEECH"
JOIN_ROLE_NON_SPEECH_TO_NON_SPEECH = "NON_SPEECH_TO_NON_SPEECH"
JOIN_ROLE_SILENCE_BOUNDARY = "SILENCE_BOUNDARY"
JOIN_ROLE_ACOUSTIC_DISCONTINUITY = "ACOUSTIC_DISCONTINUITY"
JOIN_ROLE_AMBIGUOUS = "AMBIGUOUS"
JOIN_ROLE_CONFLICTED = "CONFLICTED"
JOIN_ROLE_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Join (understanding) status vocabulary.
# ---------------------------------------------------------------------------
UNDERSTANDING_AVAILABLE = "AVAILABLE"
UNDERSTANDING_PARTIAL = "PARTIAL"
UNDERSTANDING_CONFLICTED = "CONFLICTED"
UNDERSTANDING_NOT_EVALUABLE = "NOT_EVALUABLE"
UNDERSTANDING_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Level-continuity vocabulary (categorical, never a gain correction).
# ---------------------------------------------------------------------------
LEVEL_CONTINUITY_SIMILAR = "SIMILAR_LEVEL"
LEVEL_CONTINUITY_LEFT_LOUDER = "LEFT_LOUDER"
LEVEL_CONTINUITY_RIGHT_LOUDER = "RIGHT_LOUDER"
LEVEL_CONTINUITY_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_LEVEL_SIMILARITY_THRESHOLD_DB = 3.0
# ^ THE ONE bounded, isolated, explicitly-labeled threshold this module
# adds (mirrors D-230's own naming convention exactly). Below this
# absolute dB delta, two edges are reported SIMILAR_LEVEL; never used to
# gate a firewall, purely descriptive.

# ---------------------------------------------------------------------------
# Acoustic-continuity fallback (when no comparison was supplied at all --
# distinct from D-230's own CONTINUITY_INSUFFICIENT, which means "a
# comparison was attempted and lacked signature evidence").
# ---------------------------------------------------------------------------
ACOUSTIC_CONTINUITY_STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Treatment-evidence readiness vocabulary (categorical, never a boolean --
# this task's own explicit "prefer categorical readiness statuses"
# instruction). Means "evaluable later", never "apply now".
# ---------------------------------------------------------------------------
TREATMENT_EVIDENCE_READY = "TREATMENT_EVIDENCE_READY"
TREATMENT_EVIDENCE_NOT_READY = "TREATMENT_EVIDENCE_NOT_READY"
TREATMENT_EVIDENCE_UNKNOWN = "TREATMENT_EVIDENCE_UNKNOWN"

# ---------------------------------------------------------------------------
# No-overprocessing signal vocabulary.
# ---------------------------------------------------------------------------
NO_TREATMENT_EVIDENCE = "NO_ADDITIONAL_AUDIO_TREATMENT_EVIDENCE"
ADDITIONAL_TREATMENT_EVIDENCE_PRESENT = "ADDITIONAL_TREATMENT_EVIDENCE_PRESENT"
TREATMENT_NEED_UNKNOWN = "TREATMENT_NEED_UNKNOWN"

# ---------------------------------------------------------------------------
# Prosodic join-level vocabulary (descriptive, not a BestTake ranking).
# ---------------------------------------------------------------------------
PROSODIC_JOIN_CONTINUOUS = "CONTINUOUS"
PROSODIC_JOIN_RESTART = "RESTART"
PROSODIC_JOIN_HESITATION = "HESITATION"
PROSODIC_JOIN_UNAVAILABLE = "UNAVAILABLE"
PROSODIC_JOIN_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Join-local double-speech vocabulary (deliberately distinct names from
# pacing_transition_decision's own mode-specific DOUBLE_SPEECH_* set --
# see module docstring).
# ---------------------------------------------------------------------------
JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL = "BOTH_SIDES_LEXICAL"
JOIN_DOUBLE_SPEECH_LEFT_ONLY_LEXICAL = "LEFT_ONLY_LEXICAL"
JOIN_DOUBLE_SPEECH_RIGHT_ONLY_LEXICAL = "RIGHT_ONLY_LEXICAL"
JOIN_DOUBLE_SPEECH_NEITHER_LEXICAL = "NEITHER_LEXICAL"
JOIN_DOUBLE_SPEECH_STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Conflict-flag vocabulary.
# ---------------------------------------------------------------------------
CONFLICT_LEFT_EDGE_MISSING = "left_edge_evidence_missing"
CONFLICT_RIGHT_EDGE_MISSING = "right_edge_evidence_missing"
CONFLICT_LEFT_EDGE_CONFLICTED = "left_edge_evidence_conflicted"
CONFLICT_RIGHT_EDGE_CONFLICTED = "right_edge_evidence_conflicted"
CONFLICT_CONTINUITY_CONFLICTED = "acoustic_continuity_conflicted"
CONFLICT_MEANING_UNKNOWN = "meaning_safety_unknown"
CONFLICT_WORD_SAFETY_UNKNOWN = "word_safety_unknown"

PROVENANCE_ACOUSTIC_EDGE_EVIDENCE = "pacing_v2_acoustic_edge_evidence_reused"
PROVENANCE_ACOUSTIC_CONTINUITY = "acoustic_continuity_comparison_reused"
PROVENANCE_SOURCE_AUDIO_HANDLE = "source_audio_handle_reused"
PROVENANCE_RELATIONSHIP_HINT = "relationship_hint_reused"
PROVENANCE_PROSODIC_EVIDENCE = "prosodic_edge_evidence_reused"
PROVENANCE_MEANING_SAFETY = "semantic_claims_classify_claim_reused"


@dataclass(frozen=True)
class AudioJoinUnderstanding:
    """One adjacent join's structured interpretation. Evidence only --
    see module docstring for the full list of what this never decides.
    No transcript dump, no master score anywhere on this type."""

    schema_version: str
    transition_index: int

    left_clip_id: str
    right_clip_id: str
    left_source_asset_id: str
    right_source_asset_id: str

    left_edge_evidence_id: Optional[str]
    right_edge_evidence_id: Optional[str]

    left_handle_ids: Tuple[str, ...]
    right_handle_ids: Tuple[str, ...]

    left_speech_status: str
    right_speech_status: str
    left_non_speech_status: str
    right_non_speech_status: str
    left_silence_status: str
    right_silence_status: str

    acoustic_continuity_status: str
    level_continuity_status: str

    relationship_hint: Optional[str]
    prosodic_status: str

    word_safety_status: str
    meaning_safety_status: str
    double_speech_status: str

    join_audio_role: str
    understanding_status: str

    short_crossfade_evidence_status: str
    ambience_carry_left_evidence_status: str
    ambience_carry_right_evidence_status: str
    ambience_bridge_evidence_status: str
    no_treatment_evidence_status: str

    room_tone_classification_status: str

    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _edge_speech(edge: Optional[AcousticEdgeEvidence]) -> str:
    return edge.speech_status if edge is not None else SPEECH_STATUS_UNKNOWN


def _edge_non_speech(edge: Optional[AcousticEdgeEvidence]) -> str:
    return edge.non_speech_status if edge is not None else NON_SPEECH_UNKNOWN


def _edge_silence(edge: Optional[AcousticEdgeEvidence]) -> str:
    return edge.silence_status if edge is not None else SILENCE_STATUS_UNKNOWN


def _acoustic_continuity_status(comparison: Optional[AcousticContinuityComparison]) -> str:
    if comparison is None:
        return ACOUSTIC_CONTINUITY_STATUS_UNKNOWN
    return comparison.continuity_status


def _level_continuity_status(comparison: Optional[AcousticContinuityComparison]) -> str:
    if comparison is None or comparison.level_delta_db is None:
        return LEVEL_CONTINUITY_INSUFFICIENT
    delta = comparison.level_delta_db
    if abs(delta) < ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_LEVEL_SIMILARITY_THRESHOLD_DB:
        return LEVEL_CONTINUITY_SIMILAR
    # level_delta_db = 20*log10(right_rms / left_rms) (D-230's own
    # compare_acoustic_edges convention) -- positive means RIGHT louder.
    return LEVEL_CONTINUITY_RIGHT_LOUDER if delta > 0 else LEVEL_CONTINUITY_LEFT_LOUDER


def _word_safety_status(left_speech: str, right_speech: str) -> str:
    """Join-local, mode-agnostic word safety: BLOCKED means "known
    lexical speech already sits at this join's own retained edge" (a
    future treatment must protect it), never a specific candidate
    window's verdict (see module docstring's layer distinction)."""
    if left_speech == SPEECH_STATUS_LEXICAL_PRESENT or right_speech == SPEECH_STATUS_LEXICAL_PRESENT:
        return SAFETY_BLOCKED
    if left_speech == SPEECH_STATUS_UNKNOWN or right_speech == SPEECH_STATUS_UNKNOWN:
        return SAFETY_UNKNOWN
    return SAFETY_SAFE


def _meaning_safety_status(
    left_edge_words: Sequence[Tuple[float, float, str]],
    right_edge_words: Sequence[Tuple[float, float, str]],
    *, left_word_coverage_unknown: bool, right_word_coverage_unknown: bool,
) -> str:
    """Reuses `semantic_claims.classify_claim` (D-038) directly on
    whatever near-edge word text is supplied -- the SAME function
    `pacing_transition_decision._meaning_safety` already calls, never a
    second implementation. `*_word_coverage_unknown` mirrors D-230's own
    firewall: unknown coverage never becomes SAFE by default."""
    if left_word_coverage_unknown or right_word_coverage_unknown:
        return SAFETY_UNKNOWN
    for words in (left_edge_words, right_edge_words):
        if not words:
            continue
        text = " ".join(str(w[2]) for w in words)
        _claim_type, importance, _evidence = classify_claim(text)
        if importance == CRITICAL:
            return SAFETY_BLOCKED
    return SAFETY_SAFE


def _join_double_speech_status(left_speech: str, right_speech: str) -> str:
    if left_speech == SPEECH_STATUS_UNKNOWN or right_speech == SPEECH_STATUS_UNKNOWN:
        return JOIN_DOUBLE_SPEECH_STATUS_UNKNOWN
    left_lexical = left_speech == SPEECH_STATUS_LEXICAL_PRESENT
    right_lexical = right_speech == SPEECH_STATUS_LEXICAL_PRESENT
    if left_lexical and right_lexical:
        return JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL
    if left_lexical:
        return JOIN_DOUBLE_SPEECH_LEFT_ONLY_LEXICAL
    if right_lexical:
        return JOIN_DOUBLE_SPEECH_RIGHT_ONLY_LEXICAL
    return JOIN_DOUBLE_SPEECH_NEITHER_LEXICAL


def _prosodic_join_status(left_prosody: Optional[object], right_prosody: Optional[object]) -> str:
    """Duck-typed exactly like `pacing_transition_decision._prosody_
    supports_overlap` reads Prosodic evidence (`restart_or_interruption_
    state`/`vocal_continuity_state` via `getattr`) -- no new Prosodic
    call, no BestTake ranking, no emotion inference."""
    if left_prosody is None and right_prosody is None:
        return PROSODIC_JOIN_UNAVAILABLE
    for prosody in (left_prosody, right_prosody):
        if prosody is None:
            continue
        state = getattr(prosody, "restart_or_interruption_state", None)
        if state and state not in ("NONE", "UNKNOWN"):
            return PROSODIC_JOIN_RESTART
    for prosody in (left_prosody, right_prosody):
        if prosody is None:
            continue
        if getattr(prosody, "vocal_continuity_state", None) == "FRAGMENTED":
            return PROSODIC_JOIN_HESITATION
    for prosody in (left_prosody, right_prosody):
        if prosody is None:
            continue
        if getattr(prosody, "vocal_continuity_state", None) == "CONTINUOUS":
            return PROSODIC_JOIN_CONTINUOUS
    return PROSODIC_JOIN_UNKNOWN


def _join_audio_role(
    *, left_edge: Optional[AcousticEdgeEvidence], right_edge: Optional[AcousticEdgeEvidence],
    left_speech: str, right_speech: str, left_non_speech: str, right_non_speech: str,
    acoustic_continuity_status: str,
) -> str:
    """Priority-ordered, exhaustive. Descriptive understanding only --
    never a treatment selection (module docstring)."""
    left_conflicted = left_edge is not None and left_edge.evidence_status == EVIDENCE_STATUS_CONFLICTED
    right_conflicted = right_edge is not None and right_edge.evidence_status == EVIDENCE_STATUS_CONFLICTED
    if left_conflicted or right_conflicted or acoustic_continuity_status == CONTINUITY_CONFLICTED:
        return JOIN_ROLE_CONFLICTED
    if left_edge is None and right_edge is None:
        return JOIN_ROLE_UNKNOWN

    left_lexical = left_speech == SPEECH_STATUS_LEXICAL_PRESENT
    right_lexical = right_speech == SPEECH_STATUS_LEXICAL_PRESENT
    if left_lexical and right_lexical:
        return JOIN_ROLE_SPEECH_TO_SPEECH
    if left_lexical and right_non_speech in (NON_SPEECH_SAFE_CANDIDATE, NON_SPEECH_UNCONFIRMED, NON_SPEECH_SILENCE):
        return JOIN_ROLE_SPEECH_TO_NON_SPEECH
    if right_lexical and left_non_speech in (NON_SPEECH_SAFE_CANDIDATE, NON_SPEECH_UNCONFIRMED, NON_SPEECH_SILENCE):
        return JOIN_ROLE_NON_SPEECH_TO_SPEECH
    if left_non_speech == NON_SPEECH_SILENCE or right_non_speech == NON_SPEECH_SILENCE:
        if not (left_lexical or right_lexical):
            return JOIN_ROLE_SILENCE_BOUNDARY
    if left_non_speech in (NON_SPEECH_SAFE_CANDIDATE, NON_SPEECH_UNCONFIRMED) and \
            right_non_speech in (NON_SPEECH_SAFE_CANDIDATE, NON_SPEECH_UNCONFIRMED):
        if acoustic_continuity_status == CONTINUITY_DIFFERENT:
            return JOIN_ROLE_ACOUSTIC_DISCONTINUITY
        return JOIN_ROLE_NON_SPEECH_TO_NON_SPEECH
    if left_speech == SPEECH_STATUS_UNKNOWN or right_speech == SPEECH_STATUS_UNKNOWN:
        return JOIN_ROLE_UNKNOWN
    if not (left_lexical or right_lexical) and acoustic_continuity_status == CONTINUITY_SIMILAR:
        return JOIN_ROLE_CLEAN_DIRECT_JOIN
    return JOIN_ROLE_AMBIGUOUS


def _understanding_status(
    *, left_edge: Optional[AcousticEdgeEvidence], right_edge: Optional[AcousticEdgeEvidence],
    join_audio_role: str, acoustic_continuity_status: str,
) -> str:
    if join_audio_role == JOIN_ROLE_CONFLICTED:
        return UNDERSTANDING_CONFLICTED
    if left_edge is None and right_edge is None:
        return UNDERSTANDING_NOT_EVALUABLE
    if join_audio_role == JOIN_ROLE_UNKNOWN:
        return UNDERSTANDING_UNKNOWN
    if left_edge is None or right_edge is None or acoustic_continuity_status in (
        CONTINUITY_INSUFFICIENT, ACOUSTIC_CONTINUITY_STATUS_UNKNOWN,
    ):
        return UNDERSTANDING_PARTIAL
    return UNDERSTANDING_AVAILABLE


def _short_crossfade_evidence_status(
    *, left_edge: Optional[AcousticEdgeEvidence], right_edge: Optional[AcousticEdgeEvidence],
    acoustic_continuity_status: str, word_safety_status: str, meaning_safety_status: str,
) -> str:
    """D-229 established SHORT_CROSSFADE does NOT require SourceAudioHandle
    -- readiness depends only on the two retained edges being present and
    comparable, plus resolved (not UNKNOWN) safety evidence. READY never
    means "safe" -- it means "evaluable" (module docstring)."""
    if left_edge is None or right_edge is None:
        return TREATMENT_EVIDENCE_NOT_READY
    if acoustic_continuity_status in (CONTINUITY_INSUFFICIENT, ACOUSTIC_CONTINUITY_STATUS_UNKNOWN):
        return TREATMENT_EVIDENCE_UNKNOWN
    if acoustic_continuity_status == CONTINUITY_CONFLICTED:
        return TREATMENT_EVIDENCE_UNKNOWN
    if word_safety_status == SAFETY_UNKNOWN or meaning_safety_status == SAFETY_UNKNOWN:
        return TREATMENT_EVIDENCE_UNKNOWN
    return TREATMENT_EVIDENCE_READY


def _ambience_carry_evidence_status(handle: Optional[SourceAudioHandle], handle_edge: Optional[AcousticEdgeEvidence]) -> str:
    """Requires a safe non-speech handle AND actual non-silent acoustic
    material -- a silence-only handle is NOT ambience-carry material
    (this task's own restated D-230 finding)."""
    if handle is None:
        return TREATMENT_EVIDENCE_NOT_READY
    if handle.handle_status != HANDLE_STATUS_SAFE_NON_SPEECH:
        return TREATMENT_EVIDENCE_NOT_READY
    if handle_edge is None:
        return TREATMENT_EVIDENCE_UNKNOWN
    if handle_edge.non_speech_status == NON_SPEECH_SAFE_CANDIDATE:
        return TREATMENT_EVIDENCE_READY
    if handle_edge.non_speech_status in (NON_SPEECH_SILENCE, NON_SPEECH_SPEECH_PRESENT):
        return TREATMENT_EVIDENCE_NOT_READY
    return TREATMENT_EVIDENCE_UNKNOWN


def _ambience_bridge_evidence_status(
    left_ambience_status: str, right_ambience_status: str, acoustic_continuity_status: str,
) -> str:
    """Requires safe non-speech material on BOTH sides AND a measured
    acoustic discontinuity worth smoothing -- descriptors only, per
    D-230's own room-tone honesty (`ROOM_TONE_CLASSIFICATION_STATUS`
    stays `NOT_YET_AVAILABLE`, never claimed here either)."""
    if TREATMENT_EVIDENCE_UNKNOWN in (left_ambience_status, right_ambience_status):
        return TREATMENT_EVIDENCE_UNKNOWN
    if acoustic_continuity_status in (CONTINUITY_INSUFFICIENT, ACOUSTIC_CONTINUITY_STATUS_UNKNOWN, CONTINUITY_CONFLICTED):
        return TREATMENT_EVIDENCE_UNKNOWN
    if left_ambience_status == TREATMENT_EVIDENCE_READY and right_ambience_status == TREATMENT_EVIDENCE_READY \
            and acoustic_continuity_status == CONTINUITY_DIFFERENT:
        return TREATMENT_EVIDENCE_READY
    return TREATMENT_EVIDENCE_NOT_READY


def _no_treatment_evidence_status(
    *, acoustic_continuity_status: str, level_continuity_status: str, word_safety_status: str,
    double_speech_status: str, any_treatment_ready: bool,
) -> str:
    """The "no overprocessing" signal this task requires -- a future
    treatment engine must be able to choose NONE. Never itself a
    treatment decision."""
    if acoustic_continuity_status == ACOUSTIC_CONTINUITY_STATUS_UNKNOWN or word_safety_status == SAFETY_UNKNOWN:
        return TREATMENT_NEED_UNKNOWN
    clean = (
        acoustic_continuity_status == CONTINUITY_SIMILAR
        and level_continuity_status == LEVEL_CONTINUITY_SIMILAR
        and word_safety_status != SAFETY_BLOCKED
        and double_speech_status in (JOIN_DOUBLE_SPEECH_NEITHER_LEXICAL,)
    )
    if clean:
        return NO_TREATMENT_EVIDENCE
    if any_treatment_ready:
        return ADDITIONAL_TREATMENT_EVIDENCE_PRESENT
    return TREATMENT_NEED_UNKNOWN


def build_audio_join_understanding(
    *,
    transition_index: int,
    left_clip_id: str,
    right_clip_id: str,
    left_source_asset_id: str,
    right_source_asset_id: str,
    left_edge_evidence: Optional[AcousticEdgeEvidence] = None,
    right_edge_evidence: Optional[AcousticEdgeEvidence] = None,
    continuity_comparison: Optional[AcousticContinuityComparison] = None,
    left_post_roll_handle: Optional[SourceAudioHandle] = None,
    right_pre_roll_handle: Optional[SourceAudioHandle] = None,
    left_post_roll_handle_edge_evidence: Optional[AcousticEdgeEvidence] = None,
    right_pre_roll_handle_edge_evidence: Optional[AcousticEdgeEvidence] = None,
    relationship_hint: Optional[str] = None,
    left_prosody: Optional[object] = None,
    right_prosody: Optional[object] = None,
    left_edge_words: Sequence[Tuple[float, float, str]] = (),
    right_edge_words: Sequence[Tuple[float, float, str]] = (),
) -> AudioJoinUnderstanding:
    """The one D-231 entry point. Builds ONE `AudioJoinUnderstanding` for
    ONE adjacent join (pairwise only, no global optimizer). Every input
    is already-computed evidence from an earlier authorized module
    (D-217/D-223/D-230); this function combines it, deriving nothing
    from raw media or raw ASR itself.

    `left_edge_evidence`/`right_edge_evidence` are the two join-adjacent
    D-230 `AcousticEdgeEvidence` objects for the RETAINED clip edges
    (`EDGE_LEFT_END` on the left clip, `EDGE_RIGHT_START` on the right
    clip). `left_post_roll_handle`/`right_pre_roll_handle` are the two
    D-223 `SourceAudioHandle`s bordering this same join (left's trailing
    POST_ROLL, right's leading PRE_ROLL); their own D-230 edge evidence
    (`EDGE_POST_HANDLE`/`EDGE_PRE_HANDLE`) is supplied separately since a
    handle may exist without acoustic evidence having been built for it
    yet (D-230's own `audio=None` convention, reused here)."""
    left_speech = _edge_speech(left_edge_evidence)
    right_speech = _edge_speech(right_edge_evidence)
    left_non_speech = _edge_non_speech(left_edge_evidence)
    right_non_speech = _edge_non_speech(right_edge_evidence)
    left_silence = _edge_silence(left_edge_evidence)
    right_silence = _edge_silence(right_edge_evidence)

    acoustic_continuity_status = _acoustic_continuity_status(continuity_comparison)
    level_continuity_status = _level_continuity_status(continuity_comparison)

    left_word_coverage_unknown = left_edge_evidence is None or left_speech == SPEECH_STATUS_UNKNOWN
    right_word_coverage_unknown = right_edge_evidence is None or right_speech == SPEECH_STATUS_UNKNOWN
    word_safety_status = _word_safety_status(left_speech, right_speech)
    meaning_safety_status = _meaning_safety_status(
        left_edge_words, right_edge_words,
        left_word_coverage_unknown=left_word_coverage_unknown,
        right_word_coverage_unknown=right_word_coverage_unknown,
    )
    double_speech_status = _join_double_speech_status(left_speech, right_speech)
    prosodic_status = _prosodic_join_status(left_prosody, right_prosody)

    join_audio_role = _join_audio_role(
        left_edge=left_edge_evidence, right_edge=right_edge_evidence,
        left_speech=left_speech, right_speech=right_speech,
        left_non_speech=left_non_speech, right_non_speech=right_non_speech,
        acoustic_continuity_status=acoustic_continuity_status,
    )
    understanding_status = _understanding_status(
        left_edge=left_edge_evidence, right_edge=right_edge_evidence,
        join_audio_role=join_audio_role, acoustic_continuity_status=acoustic_continuity_status,
    )

    short_crossfade_status = _short_crossfade_evidence_status(
        left_edge=left_edge_evidence, right_edge=right_edge_evidence,
        acoustic_continuity_status=acoustic_continuity_status,
        word_safety_status=word_safety_status, meaning_safety_status=meaning_safety_status,
    )
    ambience_left_status = _ambience_carry_evidence_status(left_post_roll_handle, left_post_roll_handle_edge_evidence)
    ambience_right_status = _ambience_carry_evidence_status(right_pre_roll_handle, right_pre_roll_handle_edge_evidence)
    ambience_bridge_status = _ambience_bridge_evidence_status(
        ambience_left_status, ambience_right_status, acoustic_continuity_status,
    )
    any_treatment_ready = TREATMENT_EVIDENCE_READY in (
        short_crossfade_status, ambience_left_status, ambience_right_status, ambience_bridge_status,
    )
    no_treatment_status = _no_treatment_evidence_status(
        acoustic_continuity_status=acoustic_continuity_status, level_continuity_status=level_continuity_status,
        word_safety_status=word_safety_status, double_speech_status=double_speech_status,
        any_treatment_ready=any_treatment_ready,
    )

    conflict_flags: list = []
    if left_edge_evidence is None:
        conflict_flags.append(CONFLICT_LEFT_EDGE_MISSING)
    elif left_edge_evidence.evidence_status == EVIDENCE_STATUS_CONFLICTED:
        conflict_flags.append(CONFLICT_LEFT_EDGE_CONFLICTED)
    if right_edge_evidence is None:
        conflict_flags.append(CONFLICT_RIGHT_EDGE_MISSING)
    elif right_edge_evidence.evidence_status == EVIDENCE_STATUS_CONFLICTED:
        conflict_flags.append(CONFLICT_RIGHT_EDGE_CONFLICTED)
    if acoustic_continuity_status == CONTINUITY_CONFLICTED:
        conflict_flags.append(CONFLICT_CONTINUITY_CONFLICTED)
    if meaning_safety_status == SAFETY_UNKNOWN:
        conflict_flags.append(CONFLICT_MEANING_UNKNOWN)
    if word_safety_status == SAFETY_UNKNOWN:
        conflict_flags.append(CONFLICT_WORD_SAFETY_UNKNOWN)
    for edge in (left_edge_evidence, right_edge_evidence, left_post_roll_handle_edge_evidence, right_pre_roll_handle_edge_evidence):
        if edge is not None:
            conflict_flags.extend(edge.conflict_flags)

    left_handle_ids = (left_post_roll_handle.handle_id,) if left_post_roll_handle is not None else ()
    right_handle_ids = (right_pre_roll_handle.handle_id,) if right_pre_roll_handle is not None else ()

    provenance: set = {SCHEMA_VERSION}
    if left_edge_evidence is not None or right_edge_evidence is not None:
        provenance.add(PROVENANCE_ACOUSTIC_EDGE_EVIDENCE)
    if continuity_comparison is not None:
        provenance.add(PROVENANCE_ACOUSTIC_CONTINUITY)
    if left_post_roll_handle is not None or right_pre_roll_handle is not None:
        provenance.add(PROVENANCE_SOURCE_AUDIO_HANDLE)
    if relationship_hint is not None:
        provenance.add(PROVENANCE_RELATIONSHIP_HINT)
    if left_prosody is not None or right_prosody is not None:
        provenance.add(PROVENANCE_PROSODIC_EVIDENCE)
    if left_edge_words or right_edge_words:
        provenance.add(PROVENANCE_MEANING_SAFETY)
    for edge in (left_edge_evidence, right_edge_evidence, left_post_roll_handle_edge_evidence, right_pre_roll_handle_edge_evidence):
        if edge is not None:
            provenance.update(edge.provenance)
    if continuity_comparison is not None:
        provenance.update(continuity_comparison.provenance)

    return AudioJoinUnderstanding(
        schema_version=SCHEMA_VERSION,
        transition_index=transition_index,
        left_clip_id=left_clip_id, right_clip_id=right_clip_id,
        left_source_asset_id=left_source_asset_id, right_source_asset_id=right_source_asset_id,
        left_edge_evidence_id=left_edge_evidence.evidence_id if left_edge_evidence else None,
        right_edge_evidence_id=right_edge_evidence.evidence_id if right_edge_evidence else None,
        left_handle_ids=left_handle_ids, right_handle_ids=right_handle_ids,
        left_speech_status=left_speech, right_speech_status=right_speech,
        left_non_speech_status=left_non_speech, right_non_speech_status=right_non_speech,
        left_silence_status=left_silence, right_silence_status=right_silence,
        acoustic_continuity_status=acoustic_continuity_status,
        level_continuity_status=level_continuity_status,
        relationship_hint=relationship_hint,
        prosodic_status=prosodic_status,
        word_safety_status=word_safety_status,
        meaning_safety_status=meaning_safety_status,
        double_speech_status=double_speech_status,
        join_audio_role=join_audio_role,
        understanding_status=understanding_status,
        short_crossfade_evidence_status=short_crossfade_status,
        ambience_carry_left_evidence_status=ambience_left_status,
        ambience_carry_right_evidence_status=ambience_right_status,
        ambience_bridge_evidence_status=ambience_bridge_status,
        no_treatment_evidence_status=no_treatment_status,
        room_tone_classification_status=ROOM_TONE_CLASSIFICATION_STATUS,
        conflict_flags=tuple(conflict_flags),
        provenance=tuple(sorted(provenance)),
    )


def audio_join_understanding_diagnostics(understanding: AudioJoinUnderstanding) -> dict:
    """Compact, JSON-safe diagnostics row -- no transcript dump (this
    task's own explicit instruction)."""
    return {
        "schema_version": understanding.schema_version,
        "transition_index": understanding.transition_index,
        "left_clip_id": understanding.left_clip_id, "right_clip_id": understanding.right_clip_id,
        "left_source_asset_id": understanding.left_source_asset_id,
        "right_source_asset_id": understanding.right_source_asset_id,
        "left_edge_evidence_id": understanding.left_edge_evidence_id,
        "right_edge_evidence_id": understanding.right_edge_evidence_id,
        "left_handle_ids": list(understanding.left_handle_ids),
        "right_handle_ids": list(understanding.right_handle_ids),
        "left_speech_status": understanding.left_speech_status,
        "right_speech_status": understanding.right_speech_status,
        "left_non_speech_status": understanding.left_non_speech_status,
        "right_non_speech_status": understanding.right_non_speech_status,
        "left_silence_status": understanding.left_silence_status,
        "right_silence_status": understanding.right_silence_status,
        "acoustic_continuity_status": understanding.acoustic_continuity_status,
        "level_continuity_status": understanding.level_continuity_status,
        "relationship_hint": understanding.relationship_hint,
        "prosodic_status": understanding.prosodic_status,
        "word_safety_status": understanding.word_safety_status,
        "meaning_safety_status": understanding.meaning_safety_status,
        "double_speech_status": understanding.double_speech_status,
        "join_audio_role": understanding.join_audio_role,
        "understanding_status": understanding.understanding_status,
        "short_crossfade_evidence_status": understanding.short_crossfade_evidence_status,
        "ambience_carry_left_evidence_status": understanding.ambience_carry_left_evidence_status,
        "ambience_carry_right_evidence_status": understanding.ambience_carry_right_evidence_status,
        "ambience_bridge_evidence_status": understanding.ambience_bridge_evidence_status,
        "no_treatment_evidence_status": understanding.no_treatment_evidence_status,
        "room_tone_classification_status": understanding.room_tone_classification_status,
        "conflict_flags": list(understanding.conflict_flags),
        "provenance": list(understanding.provenance),
    }


def audio_join_understanding_run_summary(rows: Sequence[AudioJoinUnderstanding]) -> dict:
    """No master score (this task's own explicit instruction) -- plain
    counts only, per the directive's own named fields."""
    def _count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    return {
        "schema_version": SCHEMA_VERSION,
        "join_count": len(rows),
        "speech_to_speech_count": _count(lambda r: r.join_audio_role == JOIN_ROLE_SPEECH_TO_SPEECH),
        "speech_to_non_speech_count": _count(lambda r: r.join_audio_role == JOIN_ROLE_SPEECH_TO_NON_SPEECH),
        "non_speech_to_speech_count": _count(lambda r: r.join_audio_role == JOIN_ROLE_NON_SPEECH_TO_SPEECH),
        "non_speech_to_non_speech_count": _count(lambda r: r.join_audio_role == JOIN_ROLE_NON_SPEECH_TO_NON_SPEECH),
        "silence_boundary_count": _count(lambda r: r.join_audio_role == JOIN_ROLE_SILENCE_BOUNDARY),
        "acoustically_similar_count": _count(lambda r: r.acoustic_continuity_status == CONTINUITY_SIMILAR),
        "acoustically_different_count": _count(lambda r: r.acoustic_continuity_status == CONTINUITY_DIFFERENT),
        "acoustic_insufficient_count": _count(
            lambda r: r.acoustic_continuity_status in (CONTINUITY_INSUFFICIENT, ACOUSTIC_CONTINUITY_STATUS_UNKNOWN)
        ),
        "safe_left_handle_count": _count(lambda r: r.ambience_carry_left_evidence_status == TREATMENT_EVIDENCE_READY),
        "safe_right_handle_count": _count(lambda r: r.ambience_carry_right_evidence_status == TREATMENT_EVIDENCE_READY),
        "crossfade_evaluable_count": _count(lambda r: r.short_crossfade_evidence_status == TREATMENT_EVIDENCE_READY),
        "ambience_left_evaluable_count": _count(lambda r: r.ambience_carry_left_evidence_status == TREATMENT_EVIDENCE_READY),
        "ambience_right_evaluable_count": _count(lambda r: r.ambience_carry_right_evidence_status == TREATMENT_EVIDENCE_READY),
        "ambience_bridge_evaluable_count": _count(lambda r: r.ambience_bridge_evidence_status == TREATMENT_EVIDENCE_READY),
        "no_treatment_evidence_count": _count(lambda r: r.no_treatment_evidence_status == NO_TREATMENT_EVIDENCE),
        "conflicted_count": _count(lambda r: r.understanding_status == UNDERSTANDING_CONFLICTED),
        "not_evaluable_count": _count(lambda r: r.understanding_status == UNDERSTANDING_NOT_EVALUABLE),
    }
