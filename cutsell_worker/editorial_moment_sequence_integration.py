"""D-195: P1 Editorial Moment & Sequence Understanding -- Phase B,
CANONICAL EVIDENCE / LIVE PIPELINE DIAGNOSTIC INTEGRATION.

Per ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 15 and
``docs/CUTSELL_DECISIONS.md`` D-193/D-194/D-195. This module is the ONE
adapter layer between D-194's pure classifiers (``editorial_moment_
sequence.py``, unchanged) and the REAL, already-computed pipeline objects
this task is authorized to consume:

    RawUnderstandingMap (D-155)         -- optional, capability signal only
    WatchListenUnderstanding (D-157)    -- PRIMARY live evidence source
    LanguageAttempt (D-168)             -- optional, preferred when supplied
    PropositionCandidate (D-169)        -- optional, reference only
    RelationEvidence (D-169)            -- optional, preferred when supplied
    ProsodicDeliveryEvidence (D-187)    -- optional, corroboration only

## Why WatchListenUnderstanding is the primary source (forensic finding)

As of this task, ``LanguageAttempt``/``PropositionCandidate``/
``RelationEvidence`` (D-168/D-169's own Phase B/C Language Spine) are
NOT constructed as a general per-source collection anywhere in the live
``flow_b.py``/``pipeline.py`` call path -- confirmed by direct grep:
only ``language_spine_consumer_migration.py`` (D-171, two narrow,
per-call legacy-comparison call sites, never a source-wide collection)
and ``watch_listen_relation_discovery.py`` construct them at all.
Building a full ``LanguageAttempt``/``PropositionCandidate`` collection
here, from scratch, for every source, would be RUNNING LANGUAGE SPINE
CONSTRUCTION -- explicitly forbidden by this task's own directive
("Do NOT rerun ... Language Spine construction"). This module therefore
NEVER calls ``segment_language_utterances``/``build_language_attempts``/
``build_proposition_candidates``/``build_relation_evidence`` (D-166/
D-168/D-169, all unchanged, all zero-imported here).

Instead: ``RawUnderstandingMap`` + ``WatchListenUnderstanding`` (D-155/
D-157) ARE constructed live, per source, in ``flow_b.py`` today --
confirmed by direct read. ``WatchListenUnderstanding.UnderstandingSpan``
already carries, per real span, the exact evidence classes P1 needs
(``behavior_state_hypotheses`` -- reused verbatim from D-155;
``attempt_relation_hypotheses`` -- D-157's own RETRY/CORRECTION/
CONTINUATION/COMPLEMENTARY/NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION/
UNCERTAIN vocabulary, the same relation vocabulary D-169's
``RelationEvidence.relation_candidate`` mirrors verbatim;
``meaning_completion_hypothesis``; per-zone usability). This module's
ONE new piece of logic is a small, pure, DOCUMENTED translation
(`_derive_language_attempt`) from that ALREADY-COMPUTED D-157 evidence
into a ``LanguageAttempt``-shaped value, ONLY so D-194's classifier
(which requires a real ``LanguageAttempt`` as its first argument) can be
called at all. This is an honest APPROXIMATION of the real D-168
grouping algorithm (it does not replicate cross-span utterance merging),
clearly labelled as such, and used ONLY as a fallback: when a caller
already has a real ``LanguageAttempt``/``PropositionCandidate``/
``RelationEvidence`` for a span (a future, separately-authorized
integration step), this module prefers those verbatim and never invokes
the fallback.

## No recomputation of perception (structurally true, not merely claimed)

This module imports NOTHING from ``asr.py``, ``local_performance.py``,
``audio_silence.py``, ``prosodic_audio_v2.py``, ``whole_video_openai.py``,
or any semantic/whole-video provider module. It decodes no media, calls
no provider, and re-derives no completion/restart/behavior signal that
D-155/D-157 did not already compute. When an upstream object is absent
for a span, that span's evidence is marked missing/UNKNOWN -- it is
never rebuilt locally (module-leaf tests assert this).

## One-way diagnostic flow (binding)

Nothing in this module writes back into ``take_grouping``, Family
Formation, or the D-150 semantic-authority thread. It is a pure reader
of already-computed evidence and a pure producer of D-194's own
diagnostic types -- see ``pipeline.py``'s own wiring for the same
one-way, default-off, diagnostics-only pattern D-183/D-184/D-191 already
established.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Mapping, Sequence, Tuple

from .contracts import CandidateTake
from .editorial_moment_sequence import (
    EditorialMoment,
    EditorialSequenceHypothesis,
    classify_editorial_moment,
    classify_editorial_sequence,
    editorial_moment_diagnostics,
    editorial_moment_sequence_run_summary,
    editorial_sequence_diagnostics,
)
from .language_proposition_relation import PropositionCandidate, RelationEvidence
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
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
    _BRIEF_DURATION_CEILING_SEC,
    _BRIEF_WORD_CEILING,
    _word_count,
)
from .raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_RECORDING_PROCESS,
    MAP_STATUS_FAILED,
    BehaviorHypothesis,
    RawUnderstandingMap,
)
from .watch_listen_understanding import (
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    USABILITY_QUESTIONABLE,
    USABILITY_UNUSABLE,
    AttemptRelationHypothesis,
    UnderstandingSpan,
    WatchListenUnderstanding,
)

SCHEMA_VERSION = "cutsell.editorial_moment_sequence_integration.v1"

_DIAGNOSTICS_ENV = "CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def editorial_moment_sequence_diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Default OFF. When OFF, nothing in this module is ever called by
    ``pipeline.py`` -- selection/family/BestTake/D-191/Boundary/Pacing/
    render output stays byte-identical (this is a diagnostics-only
    capability, not an authority; there is no authority flag)."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


# ---------------------------------------------------------------------------
# Capability-status vocabulary.
# ---------------------------------------------------------------------------
CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"
CAPABILITY_DISABLED = "DISABLED"
ALLOWED_CAPABILITY_STATUSES: frozenset[str] = frozenset({
    CAPABILITY_AVAILABLE, CAPABILITY_PARTIAL, CAPABILITY_NOT_EVALUABLE, CAPABILITY_DISABLED,
})

# Relation-candidate values treated as decisive enough to select as the
# "dominant" relation for a span pair when D-157 emitted more than one
# hypothesis for it (rare -- see watch_listen_understanding._relation_for_pair,
# which can append both a RETRY and a CORRECTION entry for the same pair).
_RELATION_PRIORITY: Tuple[str, ...] = (
    RELATION_CORRECTION, RELATION_RETRY, RELATION_NEW_AUDIENCE_BEAT,
    RELATION_CONTINUATION, RELATION_COMPLEMENTARY, RELATION_DISTINCT_PROPOSITION,
    RELATION_UNCERTAIN,
)
_CONFIDENCE_RANK: Mapping[str, int] = {
    CONFIDENCE_SUPPORTED: 3, CONFIDENCE_MIXED: 2, CONFIDENCE_WEAK: 1, CONFIDENCE_UNKNOWN: 0,
}


def _dominant_relation(hypotheses: Tuple[AttemptRelationHypothesis, ...]) -> Tuple[str | None, str]:
    """Picks the single most decisive ALREADY-COMPUTED D-157 relation
    hypothesis for a span pair -- never a new relation-detection heuristic,
    only a selection among evidence D-157 already produced. Returns
    ``(None, CONFIDENCE_UNKNOWN)`` for the source's first span (no
    predecessor) or when the only hypothesis present is the genuine
    "insufficient evidence" UNCERTAIN/UNKNOWN case -- both are honestly
    "no relation asserted", matching D-194's own ``relation_to_predecessor
    =None`` default."""
    if not hypotheses:
        return None, CONFIDENCE_UNKNOWN

    def _rank(h: AttemptRelationHypothesis) -> tuple:
        priority = _RELATION_PRIORITY.index(h.relation) if h.relation in _RELATION_PRIORITY else len(_RELATION_PRIORITY)
        return (_CONFIDENCE_RANK.get(h.confidence, 0), -priority)

    best = max(hypotheses, key=_rank)
    if best.relation == RELATION_UNCERTAIN and best.confidence == CONFIDENCE_UNKNOWN:
        return None, CONFIDENCE_UNKNOWN
    return best.relation, best.confidence


def _behavior_state_from_labels(labels: frozenset[str]) -> str | None:
    """Bounded translation of D-155's ``BehaviorHypothesis.label`` set into
    the subset of D-168's ``LanguageAttempt.attempt_state`` vocabulary that
    has a real behavior-level correlate. Returns ``None`` when no
    behavior label forces a non-CLEAN state -- the caller then falls back
    to the meaning-completion-driven state, exactly mirroring D-194's own
    moment classifier precedence for the same three vocabularies."""
    if BEHAVIOR_RECORDING_PROCESS in labels:
        return ATTEMPT_RECORDING_PROCESS
    if BEHAVIOR_ABANDONED_ATTEMPT in labels:
        return None  # resolved by _abandoned_or_false_start below (needs word/duration)
    if BEHAVIOR_FALSE_START in labels:
        return ATTEMPT_FALSE_START
    return None


def _abandoned_or_false_start(text_raw: str, duration_sec: float) -> str:
    """Reuses D-168's OWN brief-word/duration ceiling constants verbatim
    (``_BRIEF_WORD_CEILING``/``_BRIEF_DURATION_CEILING_SEC``) -- the exact
    same split ``language_utterance_attempt._abandoned_attempt_state``
    already applies. Not a new threshold."""
    brief = _word_count(text_raw) < _BRIEF_WORD_CEILING and duration_sec < _BRIEF_DURATION_CEILING_SEC
    return ATTEMPT_FALSE_START if brief else ATTEMPT_ABANDONED


def _derive_language_attempt(
    take: CandidateTake, span: UnderstandingSpan, relation_to_predecessor: str | None,
) -> LanguageAttempt:
    """FALLBACK adapter only -- used when no real ``LanguageAttempt`` is
    supplied for this span (the current live-pipeline default; see module
    docstring). Every field is either read VERBATIM off the already-
    computed ``CandidateTake``/``UnderstandingSpan`` (text, timing,
    behavior hypotheses, meaning-completion hypothesis, behavior
    confidence) or derived via the bounded translations above. This is an
    honest APPROXIMATION of D-168's real per-source grouping algorithm
    (it does not merge multiple spans into one cross-span Attempt) --
    documented, never silently passed off as the real thing."""
    labels = frozenset(h.label for h in span.behavior_state_hypotheses)
    meaning = span.meaning_completion_hypothesis

    behavior_state = _behavior_state_from_labels(labels)
    if behavior_state is not None:
        state = behavior_state
    elif BEHAVIOR_ABANDONED_ATTEMPT in labels:
        state = _abandoned_or_false_start(take.text or "", max(0.0, span.source_end - span.source_start))
    elif relation_to_predecessor == RELATION_CORRECTION:
        state = ATTEMPT_CORRECTION
    elif meaning == MEANING_INCOMPLETE:
        state = ATTEMPT_CONTINUATION
    elif meaning == MEANING_UNCERTAIN:
        state = ATTEMPT_UNCERTAIN
    else:
        state = ATTEMPT_CLEAN

    return LanguageAttempt(
        source_asset_id=span.source_asset_id,
        attempt_id=take.clip_id,  # reuse the REAL existing canonical clip identity -- no new id minted
        utterance_ids=(take.clip_id,),
        source_start=span.source_start,
        source_end=span.source_end,
        text_raw=str(take.text or ""),
        text_normalized=str(take.text or "").lower(),
        attempt_state=state,
        meaning_completion=meaning,
        restart_evidence=relation_to_predecessor in (RELATION_RETRY, RELATION_CORRECTION),
        correction_evidence=relation_to_predecessor == RELATION_CORRECTION,
        continuation_evidence=state == ATTEMPT_CONTINUATION,
        recording_process_evidence=state == ATTEMPT_RECORDING_PROCESS,
        confidence=span.behavior_confidence,
        provenance="WATCH_LISTEN_UNDERSTANDING_FALLBACK_ADAPTER",
    )


@dataclass(frozen=True)
class EditorialMomentUnderstanding:
    """Per-source P1 aggregate -- bounded, diagnostics-only. References
    already-built ``EditorialMoment``/``EditorialSequenceHypothesis``
    objects by value (both are already-bounded, small, frozen types --
    never a giant upstream object copied in)."""
    source_asset_id: str
    moments: Tuple[EditorialMoment, ...]
    sequence_hypotheses: Tuple[EditorialSequenceHypothesis, ...]
    moment_count: int
    sequence_count: int
    capability_status: str
    missing_evidence: Tuple[str, ...]
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _understanding_span_for_take(
    take: CandidateTake, spans_by_id: Mapping[str, UnderstandingSpan],
) -> UnderstandingSpan | None:
    return spans_by_id.get(take.clip_id)


def build_editorial_moments_for_source(
    *,
    source_asset_id: str,
    takes_for_source: Iterable[CandidateTake],
    understanding_spans_by_id: Mapping[str, UnderstandingSpan],
    language_attempts_by_span_id: Mapping[str, LanguageAttempt] | None = None,
    proposition_candidate_ids_by_attempt_id: Mapping[str, Tuple[str, ...]] | None = None,
    prosodic_evidence_by_span_id: Mapping[str, object] | None = None,
) -> Tuple[Tuple[EditorialMoment, ...], int, int, dict[int, str]]:
    """Builds one ``EditorialMoment`` per eligible take, in deterministic
    ``(source_start, source_end, clip_id)`` order. A take with NO matching
    ``UnderstandingSpan`` is SKIPPED (never manufactured) -- per this
    task's own "do not create moments from arbitrary token windows /
    ambiguous source mapping" instruction. Returns ``(moments,
    unresolved_count, fallback_language_attempt_count, relation_by_
    position)`` -- the last a ``{local_sequence_position: relation_
    candidate}`` map of every ACTUALLY-RESOLVED (non-``None``) dominant
    relation, for the caller to pass straight into ``build_editorial_
    sequences_for_moments`` without recomputing it."""
    ordered = sorted(
        (t for t in takes_for_source if t.source_asset_id == source_asset_id),
        key=lambda t: (t.start, t.end, t.clip_id),
    )
    language_attempts_by_span_id = language_attempts_by_span_id or {}
    proposition_candidate_ids_by_attempt_id = proposition_candidate_ids_by_attempt_id or {}
    prosodic_evidence_by_span_id = prosodic_evidence_by_span_id or {}

    moments: list[EditorialMoment] = []
    relation_by_position: dict[int, str] = {}
    unresolved_count = 0
    fallback_language_attempt_count = 0
    position = 0
    previous_relation_lookup_span: UnderstandingSpan | None = None
    for take in ordered:
        span = _understanding_span_for_take(take, understanding_spans_by_id)
        if span is None or not take.clip_id or take.start is None or take.end is None or take.end < take.start:
            unresolved_count += 1
            continue

        relation_to_predecessor, _relation_confidence = (
            _dominant_relation(span.attempt_relation_hypotheses) if previous_relation_lookup_span is not None
            else (None, CONFIDENCE_UNKNOWN)
        )

        real_attempt = language_attempts_by_span_id.get(take.clip_id)
        if real_attempt is not None:
            attempt = real_attempt
        else:
            attempt = _derive_language_attempt(take, span, relation_to_predecessor)
            fallback_language_attempt_count += 1

        visual_reset_present = span.exit_usability in (USABILITY_UNUSABLE, USABILITY_QUESTIONABLE)

        moment = classify_editorial_moment(
            attempt,
            source_span_id=take.clip_id,
            proposition_candidate_ids=proposition_candidate_ids_by_attempt_id.get(attempt.attempt_id, ()),
            related_span_ids=(previous_relation_lookup_span.span_id,) if previous_relation_lookup_span is not None else (),
            behavior_hypotheses=span.behavior_state_hypotheses,
            relation_to_predecessor=relation_to_predecessor,
            local_sequence_position=position,
            prosodic_evidence=prosodic_evidence_by_span_id.get(take.clip_id),
            visual_reset_present=visual_reset_present,
        )
        if relation_to_predecessor is not None:
            relation_by_position[position] = relation_to_predecessor
        moments.append(moment)
        previous_relation_lookup_span = span
        position += 1

    return tuple(moments), unresolved_count, fallback_language_attempt_count, relation_by_position


def build_editorial_sequences_for_moments(
    moments: Tuple[EditorialMoment, ...],
    *,
    relation_candidates_by_position: Mapping[int, str] | None = None,
    local_groups: Sequence[Sequence[int]] | None = None,
) -> Tuple[EditorialSequenceHypothesis, ...]:
    """Forms bounded LOCAL sequences only. Per this task's own "audit
    whether canonical source ordering already exists / do not invent a
    tuned within-X-seconds rule" instruction: with no ``local_groups``
    supplied, this treats the WHOLE per-source moment list (already
    bounded to one source_asset_id by the caller) as ONE local window --
    no numeric adjacency threshold is invented. A caller MAY instead
    supply explicit ``local_groups`` (index tuples into ``moments``) for a
    finer partition; this module still performs no search of its own."""
    relation_candidates_by_position = relation_candidates_by_position or {}
    if local_groups is None:
        groups: Tuple[Tuple[int, ...], ...] = (tuple(range(len(moments))),) if len(moments) >= 2 else ()
    else:
        groups = tuple(tuple(g) for g in local_groups if len(g) >= 2)

    sequences: list[EditorialSequenceHypothesis] = []
    for group in groups:
        group_moments = [moments[i] for i in sorted(group)]
        relation_candidates = tuple(
            relation_candidates_by_position[i] for i in sorted(group)[1:]
            if i in relation_candidates_by_position
        )
        sequences.append(classify_editorial_sequence(
            group_moments,
            relation_candidates=relation_candidates,
            jump_cut_evidence=None,  # no reliable edit-transition signal exists in this codebase -- honest, never invented
            internal_redundancy_status=None,  # not evaluated in Phase B -- no duplicate-detection heuristic invented
        ))
    return tuple(sequences)


def _aggregate_confidence(
    moments: Tuple[EditorialMoment, ...], sequences: Tuple[EditorialSequenceHypothesis, ...],
) -> str:
    all_flags = [f for m in moments for f in m.conflict_flags] + [f for s in sequences for f in s.conflict_flags]
    if all_flags:
        return CONFIDENCE_MIXED
    confidences = {m.confidence for m in moments} | {s.confidence for s in sequences}
    if not confidences:
        return CONFIDENCE_UNKNOWN
    if confidences == {CONFIDENCE_SUPPORTED}:
        return CONFIDENCE_SUPPORTED
    if CONFIDENCE_MIXED in confidences:
        return CONFIDENCE_MIXED
    return CONFIDENCE_WEAK


def build_editorial_moment_understanding_for_source(
    *,
    source_asset_id: str,
    takes_for_source: Iterable[CandidateTake],
    watch_listen_understanding: WatchListenUnderstanding | None,
    raw_understanding_map: RawUnderstandingMap | None = None,
    language_attempts_by_span_id: Mapping[str, LanguageAttempt] | None = None,
    proposition_candidates: Iterable[PropositionCandidate] | None = None,
    relation_evidence: Iterable[RelationEvidence] | None = None,
    prosodic_evidence_by_span_id: Mapping[str, object] | None = None,
    local_groups: Sequence[Sequence[int]] | None = None,
) -> EditorialMomentUnderstanding:
    """The one canonical per-source P1 Phase B builder. Pure; no I/O, no
    provider call, no perception recomputation (see module docstring).
    ``proposition_candidates``/``relation_evidence`` are accepted as
    OPTIONAL real D-169 collections for future callers that already have
    them -- when given, their ids are folded in as reference evidence
    (``proposition_candidate_ids_by_attempt_id``); when absent (the
    current live-pipeline default), moments simply carry no proposition
    references, honestly reported via ``missing_evidence``."""
    missing_evidence: list[str] = []

    if raw_understanding_map is not None and raw_understanding_map.track_status.get(
        "raw_understanding_map_status"
    ) == MAP_STATUS_FAILED:
        return EditorialMomentUnderstanding(
            source_asset_id=source_asset_id, moments=(), sequence_hypotheses=(),
            moment_count=0, sequence_count=0, capability_status=CAPABILITY_NOT_EVALUABLE,
            missing_evidence=("RAW_UNDERSTANDING_MAP_FAILED",), confidence=CONFIDENCE_UNKNOWN,
            conflict_flags=(), provenance=("RAW_UNDERSTANDING_MAP",),
        )

    if watch_listen_understanding is None or not watch_listen_understanding.understanding_spans:
        return EditorialMomentUnderstanding(
            source_asset_id=source_asset_id, moments=(), sequence_hypotheses=(),
            moment_count=0, sequence_count=0, capability_status=CAPABILITY_NOT_EVALUABLE,
            missing_evidence=("WATCH_LISTEN_UNDERSTANDING_ABSENT",), confidence=CONFIDENCE_UNKNOWN,
            conflict_flags=(), provenance=(),
        )

    understanding_spans_by_id = {s.span_id: s for s in watch_listen_understanding.understanding_spans}
    takes_tuple = tuple(t for t in takes_for_source if t.source_asset_id == source_asset_id)

    proposition_candidate_ids_by_attempt_id: dict[str, Tuple[str, ...]] = {}
    if proposition_candidates is not None:
        for prop in proposition_candidates:
            for attempt_id in prop.attempt_ids:
                proposition_candidate_ids_by_attempt_id.setdefault(attempt_id, ())
                proposition_candidate_ids_by_attempt_id[attempt_id] += (prop.proposition_candidate_id,)
    else:
        missing_evidence.append("PROPOSITION_CANDIDATE_NOT_SUPPLIED")

    if relation_evidence is not None:
        # Real D-169 RelationEvidence, when supplied, is reference-only
        # here (Phase B's relation-to-predecessor decision still comes
        # from the live, per-span D-157 hypotheses above -- both
        # vocabularies mirror each other verbatim, see module docstring).
        missing_evidence_relation_supplied = True
    else:
        missing_evidence.append("RELATION_EVIDENCE_NOT_SUPPLIED")
        missing_evidence_relation_supplied = False
    del missing_evidence_relation_supplied

    if language_attempts_by_span_id is None:
        missing_evidence.append("LANGUAGE_ATTEMPT_NOT_SUPPLIED")

    if not prosodic_evidence_by_span_id:
        missing_evidence.append("PROSODIC_EVIDENCE_NOT_SUPPLIED")

    moments, unresolved_count, fallback_count, relation_by_position = build_editorial_moments_for_source(
        source_asset_id=source_asset_id,
        takes_for_source=takes_tuple,
        understanding_spans_by_id=understanding_spans_by_id,
        language_attempts_by_span_id=language_attempts_by_span_id,
        proposition_candidate_ids_by_attempt_id=proposition_candidate_ids_by_attempt_id,
        prosodic_evidence_by_span_id=prosodic_evidence_by_span_id,
    )

    if not moments:
        capability_status = CAPABILITY_NOT_EVALUABLE
    elif unresolved_count > 0:
        capability_status = CAPABILITY_PARTIAL
    else:
        capability_status = CAPABILITY_AVAILABLE

    sequences = build_editorial_sequences_for_moments(
        moments, relation_candidates_by_position=relation_by_position, local_groups=local_groups,
    )

    if fallback_count and "LANGUAGE_ATTEMPT_NOT_SUPPLIED" not in missing_evidence:
        missing_evidence.append("LANGUAGE_ATTEMPT_NOT_SUPPLIED")
    if unresolved_count:
        missing_evidence.append(f"UNRESOLVED_SOURCE_MAPPING_COUNT:{unresolved_count}")

    return EditorialMomentUnderstanding(
        source_asset_id=source_asset_id,
        moments=moments,
        sequence_hypotheses=sequences,
        moment_count=len(moments),
        sequence_count=len(sequences),
        capability_status=capability_status,
        missing_evidence=tuple(missing_evidence),
        confidence=_aggregate_confidence(moments, sequences),
        conflict_flags=tuple(sorted(set(
            f for m in moments for f in m.conflict_flags
        ) | set(
            f for s in sequences for f in s.conflict_flags
        ))),
        provenance=("WATCH_LISTEN_UNDERSTANDING",) + (("RAW_UNDERSTANDING_MAP",) if raw_understanding_map is not None else ()),
    )


def build_editorial_moment_understanding_for_sources(
    *,
    sources: Iterable[str],
    takes: Iterable[CandidateTake],
    watch_listen_understandings: Iterable[WatchListenUnderstanding],
    raw_understanding_maps: Iterable[RawUnderstandingMap] = (),
    **kwargs,
) -> Tuple[EditorialMomentUnderstanding, ...]:
    """Batch form, one per source, in ``sources`` order -- deterministic
    regardless of ``takes``/``watch_listen_understandings`` input order."""
    takes = tuple(takes)
    wlu_by_source = {u.source_asset_id: u for u in watch_listen_understandings}
    raw_by_source = {m.source_asset_id: m for m in raw_understanding_maps}
    out = []
    for source_asset_id in sources:
        out.append(build_editorial_moment_understanding_for_source(
            source_asset_id=source_asset_id,
            takes_for_source=takes,
            watch_listen_understanding=wlu_by_source.get(source_asset_id),
            raw_understanding_map=raw_by_source.get(source_asset_id),
            **kwargs,
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Diagnostics / run summary.
# ---------------------------------------------------------------------------
def editorial_moment_understanding_diagnostics(understanding: EditorialMomentUnderstanding) -> dict:
    return {
        "source_asset_id": understanding.source_asset_id,
        "moment_count": understanding.moment_count,
        "sequence_count": understanding.sequence_count,
        "capability_status": understanding.capability_status,
        "missing_evidence": list(understanding.missing_evidence),
        "confidence": understanding.confidence,
        "conflict": list(understanding.conflict_flags),
        "moments": [editorial_moment_diagnostics(m) for m in understanding.moments],
        "sequences": [editorial_sequence_diagnostics(s) for s in understanding.sequence_hypotheses],
    }


def editorial_moment_understanding_run_summary(
    understandings: Iterable[EditorialMomentUnderstanding],
) -> dict:
    understandings = tuple(understandings)
    all_moments = tuple(m for u in understandings for m in u.moments)
    all_sequences = tuple(s for u in understandings for s in u.sequence_hypotheses)

    base = editorial_moment_sequence_run_summary(all_moments, all_sequences)

    statuses = {u.capability_status for u in understandings}
    if not understandings:
        p1_status = CAPABILITY_NOT_EVALUABLE
    elif statuses == {CAPABILITY_AVAILABLE}:
        p1_status = CAPABILITY_AVAILABLE
    elif statuses == {CAPABILITY_NOT_EVALUABLE}:
        p1_status = CAPABILITY_NOT_EVALUABLE
    else:
        p1_status = CAPABILITY_PARTIAL

    return {
        **base,
        "schema_version": SCHEMA_VERSION,
        "source_count": len(understandings),
        "p1_editorial_moment_status": p1_status,
        "p1_missing_language_count": sum(
            1 for u in understandings if "LANGUAGE_ATTEMPT_NOT_SUPPLIED" in u.missing_evidence
        ),
        "p1_missing_behavior_count": sum(1 for m in all_moments if not m.provenance or "BEHAVIOR_HYPOTHESIS" not in m.provenance),
        "p1_missing_relation_count": sum(1 for m in all_moments if "RELATION_EVIDENCE" not in m.provenance),
    }
