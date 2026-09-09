"""D-194: P1 Editorial Moment & Sequence Understanding -- Phase A, TYPED
FOUNDATION + DETERMINISTIC LOCAL CLASSIFICATION.

Per ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 15
(D-178A/D-178A.1) and ``docs/CUTSELL_DECISIONS.md`` D-193/D-194. This
module answers P1's own two core questions, and NOTHING else:

    WHAT ROLE DOES THIS SOURCE-REAL MOMENT PLAY IN THE RECORDING /
    EDITORIAL PROCESS?

    WHAT LOCAL EDITORIAL SEQUENCE DO ADJACENT / RELATED MOMENTS FORM?

It never answers "which take wins", "what should be deleted", "what is
the best hook", or "what order should the final ad use globally" (P2 /
Commercial Moment / BestTake / Family / Boundary / Pacing territory,
all untouched).

## What this module is NOT (D-193's own authority boundary, restated)

This module NEVER decides final Proposition Identity, final Retry
Identity, final family topology, a BestTake winner, a Boundary trim, a
Pacing transition, or a render plan. It is not called by any production
call site as of this task: ``pipeline.py``, ``flow_b.py``,
``bounded_finalist_arbiter.py``, ``bounded_finalist_authority.py``,
``composite_resolver.py``, ``realization_resolver.py``,
``boundary_engine_pass.py``, ``dialogue_pacing_transition.py``,
``take_grouping.py``, ``take_grouping_provider.py``,
``deterministic_best_take_authority.py``, ``take_judge.py``,
``semantic_authority_observability.py`` are all confirmed unaware of
this module's existence (module-leaf grep tests in
``tests/test_cutsell_d194_editorial_moment_sequence.py``), and it
imports nothing from any of them. There is no feature flag: nothing
gates this module because nothing calls it yet.

## What this module recomputes vs. reuses (NO PARALLEL ONTOLOGY)

Recomputes NOTHING about ASR, audio, or visual content, and invents no
competing state engine. Every input this module's classifiers consume
is either:

- read directly off an already-built ``LanguageAttempt`` (D-168,
  ``language_utterance_attempt.py``, unchanged) -- ``attempt_state``,
  ``meaning_completion``, ``confidence``, ``source_start``/``source_
  end``, ``attempt_id``;
- an optional, caller-supplied ``BehaviorHypothesis`` tuple (D-155,
  ``raw_understanding_map.py``, unchanged) -- used ONLY as corroboration
  for the three behavior-derived moment roles (``BREAKING_CHARACTER``,
  ``POST_TAKE_RESET``, ``PRE_TAKE_SETUP``) a clean ``LanguageAttempt``
  alone cannot express;
- an optional, caller-supplied relation-candidate string (D-157's
  ``AttemptRelationHypothesis.relation`` or D-169's
  ``RelationEvidence.relation_candidate`` vocabulary, both unchanged)
  used only to distinguish ``RETRY``/``NEW_AUDIENCE_BEAT`` moment roles
  and, at the sequence level, ``proposition_progression_status``;
- an optional, caller-supplied ``ProsodicDeliveryEvidence`` (D-187,
  ``prosodic_audio_v2.py``, unchanged) or a plain ``visual_reset_
  present`` bool -- consumed ONLY as corroboration/conflict evidence,
  never as a basis to establish or upgrade a role by itself (the
  QUALITY-vs-STRUCTURE FIREWALL, enforced structurally below).

The genuinely NEW things this module adds are exactly the two D-193
identified: the ``CLEAN_AUDIENCE_DELIVERY``/``PREASSEMBLED_FINAL_
SEQUENCE`` concepts and the local sequence-classification logic itself
that combines already-existing per-moment evidence into a bounded
categorical hypothesis. It invents no new detector, no new numeric
threshold, and no new provider call.

## Confidence and provenance (no invented scores, no averaging)

Categorical only: ``SUPPORTED``/``WEAK``/``MIXED``/``UNKNOWN`` -- the
exact string values ``language_spine.py``/``language_utterance_
attempt.py`` already use (imported, not redefined with different
spellings). Per this task's own CONFLICT contract, any detected
disagreement between evidence sources is recorded as an explicit
``conflict_flags`` entry and forces ``confidence`` to ``MIXED`` -- never
averaged into a falsely confident classification.

## Quality-vs-structure firewall (binding, structurally enforced)

``moment_role``/``sequence_kind`` are decided from STRUCTURAL evidence
(``LanguageAttempt.attempt_state``, behavior hypotheses, relation
evidence) ONLY. Good prosody alone, good visual performance alone, or
their absence, can never establish or upgrade ``CLEAN_AUDIENCE_
DELIVERY``/``PREASSEMBLED_FINAL_SEQUENCE`` -- ``prosodic_evidence``/
``visual_reset_present`` are consulted only AFTER the structural role is
already decided, purely to record corroboration or an explicit
conflict flag (see ``classify_editorial_moment``'s own body).

## False-positive / chronology / jump-cut firewalls (binding)

``PREASSEMBLED_FINAL_SEQUENCE`` requires a CONJUNCTION of structural
evidence: every moment in the window must already independently
classify ``CLEAN_AUDIENCE_DELIVERY`` (no retry/correction/abandonment/
reset/recording-process moment anywhere in the window) AND the
caller-supplied pairwise relation evidence must show
``proposition_progression_status == FORWARD_PROGRESS``. Multiple clean
takes with NO relation evidence supplied (``relation_candidates=()``,
the honest default) can only ever reach the weaker ``CLEAN_DELIVERY_
SEQUENCE`` -- see ``_classify_sequence_kind``. Source chronology, gap
length, and ``jump_cut_evidence`` are never read by ``_classify_
sequence_kind`` at all (only recorded, unused, in ``continuity_
status``) -- they cannot influence the sequence-kind decision no matter
their value, closing the chronology and jump-cut firewalls by
construction, not by convention.

## Local sequence only (this is P1, not P2)

``classify_editorial_sequence`` requires an explicit, caller-provided,
already-bounded window of 2+ ``EditorialMoment`` objects from the SAME
source. It performs no search across a whole RAW asset, and reads no
distant-source evidence. ``earlier_source_redundancy_status`` is always
``NOT_EVALUATED`` here -- whole-video distant-source redundancy is P2
territory (D-193's own explicit boundary), never computed in this
module.

## Provider role

None. No OpenAI/Gemini/whole-video call anywhere in this module -- it
is a pure, deterministic, offline classifier over already-computed
typed evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence, Tuple

from .language_proposition_relation import (
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
)
from .language_utterance_attempt import (
    ATTEMPT_ABANDONED,
    ATTEMPT_CLEAN,
    ATTEMPT_CONTINUATION,
    ATTEMPT_CORRECTION,
    ATTEMPT_FALSE_START,
    ATTEMPT_RECORDING_PROCESS,
    ATTEMPT_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguageAttempt,
    MEANING_COMPLETE,
)
from .raw_understanding_map import (
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP,
    BehaviorHypothesis,
)

SCHEMA_VERSION = "cutsell.editorial_moment_sequence.v1"

# ---------------------------------------------------------------------------
# Moment-role vocabulary (D-193's own 12-value target set). Only
# CLEAN_AUDIENCE_DELIVERY is a genuinely new concept relative to existing
# BEHAVIOR_*/ATTEMPT_* vocabularies -- every other value mirrors an existing
# label by name (see module docstring's "NO PARALLEL ONTOLOGY" section).
# ---------------------------------------------------------------------------
MOMENT_ROLE_PRE_TAKE_SETUP = "PRE_TAKE_SETUP"
MOMENT_ROLE_RECORDING_PROCESS = "RECORDING_PROCESS"
MOMENT_ROLE_FALSE_START = "FALSE_START"
MOMENT_ROLE_ABANDONED_ATTEMPT = "ABANDONED_ATTEMPT"
MOMENT_ROLE_RETRY = "RETRY"
MOMENT_ROLE_CORRECTION = "CORRECTION"
MOMENT_ROLE_CONTINUATION = "CONTINUATION"
MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY = "CLEAN_AUDIENCE_DELIVERY"
MOMENT_ROLE_POST_TAKE_RESET = "POST_TAKE_RESET"
MOMENT_ROLE_BREAKING_CHARACTER = "BREAKING_CHARACTER"
MOMENT_ROLE_NEW_AUDIENCE_BEAT = "NEW_AUDIENCE_BEAT"
MOMENT_ROLE_UNCERTAIN = "UNCERTAIN"
ALLOWED_MOMENT_ROLES: frozenset[str] = frozenset({
    MOMENT_ROLE_PRE_TAKE_SETUP, MOMENT_ROLE_RECORDING_PROCESS, MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_ABANDONED_ATTEMPT, MOMENT_ROLE_RETRY, MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_CONTINUATION, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, MOMENT_ROLE_POST_TAKE_RESET,
    MOMENT_ROLE_BREAKING_CHARACTER, MOMENT_ROLE_NEW_AUDIENCE_BEAT, MOMENT_ROLE_UNCERTAIN,
})

# PREASSEMBLED_FINAL_SEQUENCE is deliberately absent from the moment-role
# vocabulary -- per the directive, it is primarily a SEQUENCE classification,
# never forced onto an individual moment's own role (module docstring).

# ---------------------------------------------------------------------------
# Sequence-kind vocabulary (compact, 7 values -- no state zoo).
# ---------------------------------------------------------------------------
SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE = "RECORDING_PROCESS_SEQUENCE"
SEQUENCE_KIND_RETRY_SERIES = "RETRY_SERIES"
SEQUENCE_KIND_BLOOPER_SERIES = "BLOOPER_SERIES"
SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE = "CLEAN_DELIVERY_SEQUENCE"
SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE = "PREASSEMBLED_FINAL_SEQUENCE"
SEQUENCE_KIND_MIXED = "MIXED"
SEQUENCE_KIND_UNCERTAIN = "UNCERTAIN"
ALLOWED_SEQUENCE_KINDS: frozenset[str] = frozenset({
    SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE, SEQUENCE_KIND_RETRY_SERIES,
    SEQUENCE_KIND_BLOOPER_SERIES, SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE,
    SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE, SEQUENCE_KIND_MIXED, SEQUENCE_KIND_UNCERTAIN,
})

# ---------------------------------------------------------------------------
# Audience-delivery / recording-process status vocabularies (shared between
# moment and sequence level; PARTIAL is a sequence-only aggregate value).
# ---------------------------------------------------------------------------
AUDIENCE_DELIVERY_SUPPORTED = "AUDIENCE_DELIVERY_SUPPORTED"
AUDIENCE_DELIVERY_PARTIAL = "AUDIENCE_DELIVERY_PARTIAL"
AUDIENCE_DELIVERY_NOT_SUPPORTED = "AUDIENCE_DELIVERY_NOT_SUPPORTED"
AUDIENCE_DELIVERY_UNCERTAIN = "AUDIENCE_DELIVERY_UNCERTAIN"
ALLOWED_AUDIENCE_DELIVERY_STATUSES: frozenset[str] = frozenset({
    AUDIENCE_DELIVERY_SUPPORTED, AUDIENCE_DELIVERY_PARTIAL,
    AUDIENCE_DELIVERY_NOT_SUPPORTED, AUDIENCE_DELIVERY_UNCERTAIN,
})

RECORDING_PROCESS_PRESENT = "RECORDING_PROCESS_PRESENT"
RECORDING_PROCESS_ABSENT = "RECORDING_PROCESS_ABSENT"
ALLOWED_RECORDING_PROCESS_STATUSES: frozenset[str] = frozenset({
    RECORDING_PROCESS_PRESENT, RECORDING_PROCESS_ABSENT,
})

# ---------------------------------------------------------------------------
# Proposition-progression vocabulary (sequence level). REPETITION is kept in
# the schema for completeness but is NEVER emitted by this V1 deriver -- no
# existing signal in language_proposition_relation.py cleanly distinguishes
# "same proposition restated with no new information" from RELATION_UNCERTAIN
# today (that case currently falls through to RELATION_UNCERTAIN there, per
# its own classify_relation_candidate). Same honest-gap precedent as D-155's
# own FALSE_START/PRE_TAKE_SETUP labels ("in the allowed vocabulary but never
# produced by any live deriver") -- not invented here.
# ---------------------------------------------------------------------------
PROGRESSION_FORWARD_PROGRESS = "FORWARD_PROGRESS"
PROGRESSION_REPETITION = "REPETITION"
PROGRESSION_CORRECTION = "CORRECTION"
PROGRESSION_RETRY = "RETRY"
PROGRESSION_MIXED = "MIXED"
PROGRESSION_UNKNOWN = "UNKNOWN"
ALLOWED_PROGRESSION_STATUSES: frozenset[str] = frozenset({
    PROGRESSION_FORWARD_PROGRESS, PROGRESSION_REPETITION, PROGRESSION_CORRECTION,
    PROGRESSION_RETRY, PROGRESSION_MIXED, PROGRESSION_UNKNOWN,
})

# Relation-candidate values that, in the ABSENCE of any RETRY/CORRECTION
# pair, are treated as compatible with forward proposition progression
# (module docstring's false-positive firewall).
_FORWARD_COMPATIBLE_RELATIONS: frozenset[str] = frozenset({
    RELATION_CONTINUATION, RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT,
    RELATION_DISTINCT_PROPOSITION,
})

# ---------------------------------------------------------------------------
# Internal-redundancy / continuity / earlier-source-redundancy vocabularies.
# ---------------------------------------------------------------------------
INTERNAL_REDUNDANCY_PRESENT = "INTERNAL_REDUNDANCY_PRESENT"
INTERNAL_REDUNDANCY_NOT_PRESENT = "INTERNAL_REDUNDANCY_NOT_PRESENT"
INTERNAL_REDUNDANCY_NOT_EVALUATED = "INTERNAL_REDUNDANCY_NOT_EVALUATED"
ALLOWED_INTERNAL_REDUNDANCY_STATUSES: frozenset[str] = frozenset({
    INTERNAL_REDUNDANCY_PRESENT, INTERNAL_REDUNDANCY_NOT_PRESENT, INTERNAL_REDUNDANCY_NOT_EVALUATED,
})

# No reliable edit-transition (jump-cut) signal exists in this codebase today
# -- per the directive, this capability is represented honestly rather than
# invented. `jump_cut_evidence` is accepted purely as caller-supplied,
# corroboration-only context; it is NEVER read by `_classify_sequence_kind`.
CONTINUITY_NOT_AVAILABLE = "NOT_AVAILABLE"
CONTINUITY_TRANSITION_EVIDENCE_PRESENT = "TRANSITION_EVIDENCE_PRESENT"
CONTINUITY_NO_TRANSITION_EVIDENCE = "NO_TRANSITION_EVIDENCE"
ALLOWED_CONTINUITY_STATUSES: frozenset[str] = frozenset({
    CONTINUITY_NOT_AVAILABLE, CONTINUITY_TRANSITION_EVIDENCE_PRESENT, CONTINUITY_NO_TRANSITION_EVIDENCE,
})

# Always this one value in D-194's deterministic local base -- whole-video
# distant-source redundancy search is P2 territory, never performed here.
EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED = "NOT_EVALUATED"

# Behavior-hypothesis labels this module reads for role refinement -- reused
# verbatim from raw_understanding_map.py (D-155, unchanged), never a new
# behavior vocabulary.
_BEHAVIOR_LABEL_TO_MOMENT_ROLE: dict[str, str] = {
    BEHAVIOR_BREAKING_CHARACTER: MOMENT_ROLE_BREAKING_CHARACTER,
    BEHAVIOR_POST_TAKE_RESET: MOMENT_ROLE_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP: MOMENT_ROLE_PRE_TAKE_SETUP,
}

# Roles that structurally imply retry-family/breakage content -- used by both
# the moment-role attempt_state mapping and the sequence-kind classifier.
_RESET_OR_BREAK_ROLES: frozenset[str] = frozenset({
    MOMENT_ROLE_POST_TAKE_RESET, MOMENT_ROLE_BREAKING_CHARACTER,
})
_RETRY_FAMILY_ROLES: frozenset[str] = frozenset({
    MOMENT_ROLE_RETRY, MOMENT_ROLE_ABANDONED_ATTEMPT, MOMENT_ROLE_FALSE_START, MOMENT_ROLE_CORRECTION,
})


def _editorial_moment_id(
    source_asset_id: str, source_start: float, source_end: float, moment_role: str, attempt_ids: Sequence[str],
) -> str:
    """Deterministic, content+timing-anchored id -- mirrors ``canonical_
    identity.mint_source_span_id``'s exact hashing shape under a distinct
    ``emom_`` prefix, minted here rather than registered in
    ``canonical_identity.py`` because nothing yet reads this id to make an
    editorial decision (D-050A's own "no consumer yet" precedent, reused by
    every Language Spine module to date)."""
    raw = "|".join((
        source_asset_id, f"{float(source_start):.3f}", f"{float(source_end):.3f}",
        moment_role, ",".join(sorted(str(v) for v in attempt_ids if v)),
    )).encode("utf-8")
    return "emom_" + hashlib.sha256(raw).hexdigest()[:20]


def _editorial_sequence_id(source_asset_id: str, moment_ids: Sequence[str]) -> str:
    """Deterministic, MEMBERSHIP-anchored id -- mirrors ``canonical_
    identity.mint_attempt_id``'s exact shape (sorted member id set, never
    order- or timestamp-anchored) under a distinct ``eseq_`` prefix."""
    raw = "|".join((source_asset_id, "|".join(sorted(str(v) for v in moment_ids if v)))).encode("utf-8")
    return "eseq_" + hashlib.sha256(raw).hexdigest()[:20]


# ---------------------------------------------------------------------------
# EditorialMoment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EditorialMoment:
    """One bounded moment's editorial-process-role hypothesis -- evidence,
    never an authority verdict. References child evidence by stable id
    (``attempt_ids``/``proposition_candidate_ids``/``related_span_ids``)
    rather than copying full child objects (this task's own "reference by
    stable IDs, do not copy full child objects" instruction)."""
    source_asset_id: str
    editorial_moment_id: str
    source_start: float
    source_end: float
    source_span_id: str | None
    attempt_ids: Tuple[str, ...]
    proposition_candidate_ids: Tuple[str, ...]
    related_span_ids: Tuple[str, ...]
    moment_role: str
    audience_delivery_status: str
    recording_process_status: str
    completion_status: str
    local_sequence_position: int | None
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _role_from_behavior(behavior_hypotheses: Tuple[BehaviorHypothesis, ...]) -> str | None:
    labels = {h.label for h in behavior_hypotheses}
    # Priority order mirrors raw_understanding_map.py's own precedence
    # (breaking character is the most specific/rare signal; reset is the
    # next most specific; pre-take setup is the weakest/least-observed).
    if BEHAVIOR_BREAKING_CHARACTER in labels:
        return MOMENT_ROLE_BREAKING_CHARACTER
    if BEHAVIOR_POST_TAKE_RESET in labels:
        return MOMENT_ROLE_POST_TAKE_RESET
    if BEHAVIOR_PRE_TAKE_SETUP in labels:
        return MOMENT_ROLE_PRE_TAKE_SETUP
    return None


def classify_editorial_moment(
    attempt: LanguageAttempt,
    *,
    source_span_id: str | None = None,
    proposition_candidate_ids: Tuple[str, ...] = (),
    related_span_ids: Tuple[str, ...] = (),
    behavior_hypotheses: Tuple[BehaviorHypothesis, ...] = (),
    relation_to_predecessor: str | None = None,
    local_sequence_position: int | None = None,
    prosodic_evidence: object | None = None,
    visual_reset_present: bool | None = None,
) -> EditorialMoment:
    """The one canonical LanguageAttempt -> EditorialMoment classifier this
    task requires. Pure function; no I/O, no provider/network call, no
    global context. STRUCTURE FIRST (module docstring's quality-vs-
    structure firewall): ``moment_role`` is decided from ``attempt.
    attempt_state`` + optional behavior/relation evidence ONLY;
    ``prosodic_evidence``/``visual_reset_present`` are consulted strictly
    afterward, and can only add provenance or an explicit conflict flag --
    never establish or upgrade a role by themselves."""
    conflict_flags: list[str] = []
    provenance: list[str] = ["LANGUAGE_ATTEMPT_STATE"]

    attempt_state = attempt.attempt_state
    behavior_role = _role_from_behavior(behavior_hypotheses)
    if behavior_hypotheses:
        provenance.append("BEHAVIOR_HYPOTHESIS")
    if relation_to_predecessor is not None:
        provenance.append("RELATION_EVIDENCE")

    if attempt_state == ATTEMPT_RECORDING_PROCESS:
        role = MOMENT_ROLE_RECORDING_PROCESS
    elif attempt_state == ATTEMPT_FALSE_START:
        role = MOMENT_ROLE_FALSE_START
    elif attempt_state == ATTEMPT_ABANDONED:
        role = MOMENT_ROLE_ABANDONED_ATTEMPT
    elif attempt_state == ATTEMPT_CORRECTION:
        role = MOMENT_ROLE_CORRECTION
    elif attempt_state == ATTEMPT_CONTINUATION:
        role = MOMENT_ROLE_CONTINUATION
    elif attempt_state == ATTEMPT_UNCERTAIN:
        role = MOMENT_ROLE_UNCERTAIN
    elif attempt_state == ATTEMPT_CLEAN:
        if behavior_role is not None:
            role = behavior_role
        elif relation_to_predecessor == RELATION_RETRY:
            role = MOMENT_ROLE_RETRY
        elif attempt.meaning_completion != MEANING_COMPLETE:
            # A structurally "clean" attempt whose meaning is not COMPLETE
            # has no basis to assert CLEAN_AUDIENCE_DELIVERY -- fail toward
            # UNCERTAIN rather than guess (CLAUDE.md's "WHEN UNCERTAIN, KEEP"
            # restated here as "when uncertain, do not assert a role").
            role = MOMENT_ROLE_UNCERTAIN
        elif relation_to_predecessor == RELATION_NEW_AUDIENCE_BEAT:
            role = MOMENT_ROLE_NEW_AUDIENCE_BEAT
        else:
            role = MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    else:
        # An attempt_state value outside this module's known set (future
        # vocabulary growth in D-168) -- never guess a role from it.
        role = MOMENT_ROLE_UNCERTAIN

    # --- CONFLICT contract: structural state vs. corroborating behavior
    # evidence disagreement is recorded, never silently resolved. ---
    if attempt_state == ATTEMPT_CONTINUATION and behavior_role in _RESET_OR_BREAK_ROLES:
        conflict_flags.append("CONTINUATION_STATE_VS_RESET_OR_BREAK_BEHAVIOR_EVIDENCE")

    # --- Prosodic/visual corroboration: provenance + conflict only, NEVER a
    # basis to establish or upgrade a role (quality-vs-structure firewall). ---
    if prosodic_evidence is not None:
        provenance.append("PROSODIC_CORROBORATION")
        continuity = getattr(prosodic_evidence, "vocal_continuity_state", None)
        if role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY and continuity == "FRAGMENTED":
            conflict_flags.append("PROSODIC_FRAGMENTED_CONTINUITY_VS_CLEAN_AUDIENCE_DELIVERY_STRUCTURE")
    if visual_reset_present:
        provenance.append("VISUAL_CORROBORATION")
        if role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY:
            conflict_flags.append("VISUAL_RESET_EVIDENCE_VS_CLEAN_AUDIENCE_DELIVERY_STRUCTURE")

    audience_delivery_status = (
        AUDIENCE_DELIVERY_SUPPORTED if role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
        else AUDIENCE_DELIVERY_UNCERTAIN if role == MOMENT_ROLE_UNCERTAIN
        else AUDIENCE_DELIVERY_NOT_SUPPORTED
    )
    recording_process_status = (
        RECORDING_PROCESS_PRESENT if role == MOMENT_ROLE_RECORDING_PROCESS else RECORDING_PROCESS_ABSENT
    )

    base_confidence = attempt.confidence
    confidence = CONFIDENCE_MIXED if conflict_flags else base_confidence

    editorial_moment_id = _editorial_moment_id(
        attempt.source_asset_id, attempt.source_start, attempt.source_end, role, (attempt.attempt_id,),
    )

    return EditorialMoment(
        source_asset_id=attempt.source_asset_id,
        editorial_moment_id=editorial_moment_id,
        source_start=attempt.source_start,
        source_end=attempt.source_end,
        source_span_id=source_span_id,
        attempt_ids=(attempt.attempt_id,),
        proposition_candidate_ids=tuple(proposition_candidate_ids),
        related_span_ids=tuple(related_span_ids),
        moment_role=role,
        audience_delivery_status=audience_delivery_status,
        recording_process_status=recording_process_status,
        completion_status=attempt.meaning_completion,
        local_sequence_position=local_sequence_position,
        confidence=confidence,
        conflict_flags=tuple(conflict_flags),
        provenance=tuple(provenance),
    )


# ---------------------------------------------------------------------------
# EditorialSequenceHypothesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EditorialSequenceHypothesis:
    """One bounded LOCAL sequence's structural-role hypothesis over 2+
    already-classified ``EditorialMoment``s from the same source -- a
    hypothesis only, never a final structural/authority decision. See
    module docstring's "Local sequence only" section: this is P1, never a
    whole-video (P2) search."""
    source_asset_id: str
    sequence_id: str
    moment_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    sequence_kind: str
    sequence_completeness: str
    audience_delivery_status: str
    recording_process_status: str
    proposition_progression_status: str
    internal_redundancy_status: str
    continuity_status: str
    earlier_source_redundancy_status: str
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _progression_status_from_relations(relation_candidates: Tuple[str, ...]) -> str:
    """Deterministic aggregation over caller-supplied pairwise relation
    evidence (D-157/D-169 vocabulary, reused verbatim -- never recomputed
    here). See module docstring's false-positive firewall: an EMPTY
    ``relation_candidates`` tuple (the honest default when no relation
    evidence was supplied) always yields UNKNOWN, never FORWARD_PROGRESS."""
    if not relation_candidates:
        return PROGRESSION_UNKNOWN
    if any(r == RELATION_CORRECTION for r in relation_candidates):
        return PROGRESSION_CORRECTION
    if any(r == RELATION_RETRY for r in relation_candidates):
        return PROGRESSION_RETRY
    non_uncertain = [r for r in relation_candidates if r != RELATION_UNCERTAIN]
    if not non_uncertain:
        return PROGRESSION_UNKNOWN
    if all(r in _FORWARD_COMPATIBLE_RELATIONS for r in non_uncertain):
        return PROGRESSION_FORWARD_PROGRESS
    return PROGRESSION_MIXED


def _classify_sequence_kind(
    roles: Tuple[str, ...], relation_candidates: Tuple[str, ...], progression: str,
) -> str:
    """The one canonical sequence-kind classifier. Deterministic,
    precedence-ordered, never averaged. ``relation_candidates``/
    ``progression`` are consulted only for the RETRY/CORRECTION escalation
    and the CLEAN_DELIVERY_SEQUENCE vs. PREASSEMBLED_FINAL_SEQUENCE split
    -- chronology, gaps, and jump-cut evidence are never parameters here at
    all (the chronology/jump-cut firewalls, closed by construction)."""
    has_reset_or_break = any(role in _RESET_OR_BREAK_ROLES for role in roles)
    has_retry_family = any(role in _RETRY_FAMILY_ROLES for role in roles)
    has_recording_process = any(role == MOMENT_ROLE_RECORDING_PROCESS for role in roles)
    retry_or_correction_relation = any(r in (RELATION_RETRY, RELATION_CORRECTION) for r in relation_candidates)
    all_clean = bool(roles) and all(role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY for role in roles)
    some_clean = any(role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY for role in roles)
    has_uncertain = any(role == MOMENT_ROLE_UNCERTAIN for role in roles)

    if has_reset_or_break:
        return SEQUENCE_KIND_BLOOPER_SERIES
    if has_retry_family or retry_or_correction_relation:
        return SEQUENCE_KIND_RETRY_SERIES
    if has_recording_process:
        return SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE
    if all_clean:
        return (
            SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE if progression == PROGRESSION_FORWARD_PROGRESS
            else SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE
        )
    if has_uncertain and not some_clean:
        return SEQUENCE_KIND_UNCERTAIN
    return SEQUENCE_KIND_MIXED


def _aggregate_audience_delivery(moments: Tuple[EditorialMoment, ...]) -> str:
    statuses = {m.audience_delivery_status for m in moments}
    if statuses == {AUDIENCE_DELIVERY_SUPPORTED}:
        return AUDIENCE_DELIVERY_SUPPORTED
    if AUDIENCE_DELIVERY_SUPPORTED in statuses:
        return AUDIENCE_DELIVERY_PARTIAL
    if statuses == {AUDIENCE_DELIVERY_UNCERTAIN}:
        return AUDIENCE_DELIVERY_UNCERTAIN
    return AUDIENCE_DELIVERY_NOT_SUPPORTED


def _aggregate_recording_process(moments: Tuple[EditorialMoment, ...]) -> str:
    return (
        RECORDING_PROCESS_PRESENT if any(m.recording_process_status == RECORDING_PROCESS_PRESENT for m in moments)
        else RECORDING_PROCESS_ABSENT
    )


def _sequence_confidence(sequence_kind: str, moments: Tuple[EditorialMoment, ...], conflict_flags: Tuple[str, ...]) -> str:
    """Categorical only; never averaged. Any conflict forces MIXED
    (module docstring's CONFLICT contract, restated at sequence level)."""
    if conflict_flags:
        return CONFIDENCE_MIXED
    if sequence_kind == SEQUENCE_KIND_UNCERTAIN:
        return CONFIDENCE_UNKNOWN
    if sequence_kind == SEQUENCE_KIND_MIXED:
        return CONFIDENCE_MIXED
    if sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE:
        # A weaker structural claim than PREASSEMBLED_FINAL_SEQUENCE by
        # design -- "multiple clean takes" alone is real but insufficient
        # evidence (false-positive firewall), so this never reports SUPPORTED.
        return CONFIDENCE_WEAK
    moment_confidences = {m.confidence for m in moments}
    if moment_confidences == {CONFIDENCE_SUPPORTED}:
        return CONFIDENCE_SUPPORTED
    if CONFIDENCE_MIXED in moment_confidences:
        return CONFIDENCE_MIXED
    return CONFIDENCE_WEAK


def classify_editorial_sequence(
    moments: Sequence[EditorialMoment],
    *,
    relation_candidates: Tuple[str, ...] = (),
    jump_cut_evidence: bool | None = None,
    internal_redundancy_status: str | None = None,
) -> EditorialSequenceHypothesis:
    """The one canonical EditorialMoment(s) -> EditorialSequenceHypothesis
    classifier this task requires. Pure function; no I/O, no provider
    call, no global/whole-video context. Requires 2+ moments from the SAME
    source (module docstring's "Local sequence only" section) -- this is a
    caller-provided, already-bounded local window, never a search."""
    moments = tuple(moments)
    if len(moments) < 2:
        raise ValueError("classify_editorial_sequence requires 2+ EditorialMoment objects (local sequence only)")
    source_asset_ids = {m.source_asset_id for m in moments}
    if len(source_asset_ids) != 1:
        raise ValueError("classify_editorial_sequence requires all moments to share one source_asset_id")

    ordered = tuple(sorted(moments, key=lambda m: (m.source_start, m.source_end, m.editorial_moment_id)))
    roles = tuple(m.moment_role for m in ordered)
    progression = _progression_status_from_relations(relation_candidates)
    sequence_kind = _classify_sequence_kind(roles, relation_candidates, progression)

    conflict_flags: list[str] = []
    for m in ordered:
        if m.conflict_flags:
            conflict_flags.append(f"MOMENT_LEVEL_CONFLICT_PROPAGATED:{m.editorial_moment_id}")

    if internal_redundancy_status is None:
        redundancy = INTERNAL_REDUNDANCY_NOT_EVALUATED
    elif internal_redundancy_status in ALLOWED_INTERNAL_REDUNDANCY_STATUSES:
        redundancy = internal_redundancy_status
    else:
        raise ValueError(f"unknown internal_redundancy_status: {internal_redundancy_status!r}")

    if jump_cut_evidence is None:
        continuity_status = CONTINUITY_NOT_AVAILABLE
    elif jump_cut_evidence:
        continuity_status = CONTINUITY_TRANSITION_EVIDENCE_PRESENT
    else:
        continuity_status = CONTINUITY_NO_TRANSITION_EVIDENCE

    provenance = ["EDITORIAL_MOMENT_CLASSIFICATION"]
    if relation_candidates:
        provenance.append("RELATION_EVIDENCE")
    if jump_cut_evidence is not None:
        provenance.append("JUMP_CUT_EVIDENCE")
    if internal_redundancy_status is not None:
        provenance.append("CALLER_SUPPLIED_REDUNDANCY_EVIDENCE")

    sequence_id = _editorial_sequence_id(ordered[0].source_asset_id, [m.editorial_moment_id for m in ordered])
    confidence = _sequence_confidence(sequence_kind, ordered, tuple(conflict_flags))

    return EditorialSequenceHypothesis(
        source_asset_id=ordered[0].source_asset_id,
        sequence_id=sequence_id,
        moment_ids=tuple(m.editorial_moment_id for m in ordered),
        source_start=min(m.source_start for m in ordered),
        source_end=max(m.source_end for m in ordered),
        sequence_kind=sequence_kind,
        sequence_completeness=ordered[-1].completion_status,
        audience_delivery_status=_aggregate_audience_delivery(ordered),
        recording_process_status=_aggregate_recording_process(ordered),
        proposition_progression_status=progression,
        internal_redundancy_status=redundancy,
        continuity_status=continuity_status,
        earlier_source_redundancy_status=EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED,
        confidence=confidence,
        conflict_flags=tuple(conflict_flags),
        provenance=tuple(provenance),
    )


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts/status-only -- same pattern as D-119/D-125/
# D-152/D-155/D-157/D-163/D-166/D-167/D-168/D-169's own compact summaries).
# No transcript dump.
# ---------------------------------------------------------------------------
def editorial_moment_diagnostics(moment: EditorialMoment) -> dict:
    return {
        "editorial_moment_id": moment.editorial_moment_id,
        "moment_role": moment.moment_role,
        "audience_delivery_status": moment.audience_delivery_status,
        "recording_process_status": moment.recording_process_status,
        "completion_status": moment.completion_status,
        "confidence": moment.confidence,
        "conflict": list(moment.conflict_flags),
        "source_start": moment.source_start,
        "source_end": moment.source_end,
    }


def editorial_sequence_diagnostics(sequence: EditorialSequenceHypothesis) -> dict:
    return {
        "sequence_id": sequence.sequence_id,
        "sequence_kind": sequence.sequence_kind,
        "moment_count": len(sequence.moment_ids),
        "sequence_completeness": sequence.sequence_completeness,
        "audience_delivery_status": sequence.audience_delivery_status,
        "recording_process_status": sequence.recording_process_status,
        "proposition_progression_status": sequence.proposition_progression_status,
        "internal_redundancy_status": sequence.internal_redundancy_status,
        "continuity_status": sequence.continuity_status,
        "earlier_source_redundancy_status": sequence.earlier_source_redundancy_status,
        "confidence": sequence.confidence,
        "conflict": list(sequence.conflict_flags),
        "source_start": sequence.source_start,
        "source_end": sequence.source_end,
    }


def editorial_moment_sequence_run_summary(
    moments: Sequence[EditorialMoment], sequences: Sequence[EditorialSequenceHypothesis],
) -> dict:
    """Pure aggregator over already-classified moments/sequences -- the
    exact field list this task's own "RUN/OFFLINE SUMMARY" section
    requires, plus a few honest extra role/kind counts for completeness."""
    moments = tuple(moments)
    sequences = tuple(sequences)

    role_counts: dict[str, int] = {role: 0 for role in ALLOWED_MOMENT_ROLES}
    for m in moments:
        role_counts[m.moment_role] = role_counts.get(m.moment_role, 0) + 1

    kind_counts: dict[str, int] = {kind: 0 for kind in ALLOWED_SEQUENCE_KINDS}
    for s in sequences:
        kind_counts[s.sequence_kind] = kind_counts.get(s.sequence_kind, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "editorial_moment_count": len(moments),
        "clean_audience_delivery_count": role_counts[MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY],
        "recording_process_count": role_counts[MOMENT_ROLE_RECORDING_PROCESS],
        "false_start_count": role_counts[MOMENT_ROLE_FALSE_START],
        "abandoned_attempt_count": role_counts[MOMENT_ROLE_ABANDONED_ATTEMPT],
        "retry_count": role_counts[MOMENT_ROLE_RETRY],
        "correction_count": role_counts[MOMENT_ROLE_CORRECTION],
        "continuation_count": role_counts[MOMENT_ROLE_CONTINUATION],
        "post_take_reset_count": role_counts[MOMENT_ROLE_POST_TAKE_RESET],
        "breaking_character_count": role_counts[MOMENT_ROLE_BREAKING_CHARACTER],
        "pre_take_setup_count": role_counts[MOMENT_ROLE_PRE_TAKE_SETUP],
        "new_audience_beat_count": role_counts[MOMENT_ROLE_NEW_AUDIENCE_BEAT],
        "uncertain_moment_count": role_counts[MOMENT_ROLE_UNCERTAIN],
        "moment_conflict_count": sum(1 for m in moments if m.conflict_flags),
        "editorial_sequence_count": len(sequences),
        "recording_process_sequence_count": kind_counts[SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE],
        "retry_series_count": kind_counts[SEQUENCE_KIND_RETRY_SERIES],
        "blooper_series_count": kind_counts[SEQUENCE_KIND_BLOOPER_SERIES],
        "clean_delivery_sequence_count": kind_counts[SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE],
        "preassembled_final_sequence_count": kind_counts[SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE],
        "mixed_sequence_count": kind_counts[SEQUENCE_KIND_MIXED],
        "uncertain_sequence_count": kind_counts[SEQUENCE_KIND_UNCERTAIN],
        "sequence_conflict_count": sum(1 for s in sequences if s.conflict_flags),
    }
