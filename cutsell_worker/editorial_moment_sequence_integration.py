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
import hashlib
import os
from typing import Iterable, Mapping, Sequence, Tuple

from .contracts import CandidateTake
from .editorial_moment_sequence import (
    AUDIENCE_DELIVERY_UNCERTAIN,
    EditorialMoment,
    EditorialSequenceHypothesis,
    MOMENT_ROLE_UNCERTAIN,
    classify_editorial_moment,
    classify_editorial_sequence,
    editorial_moment_diagnostics,
    editorial_moment_sequence_run_summary,
    editorial_sequence_diagnostics,
)
from .language_utterance_attempt import CONFIDENCE_SUPPORTED
from .language_proposition_relation import PropositionCandidate, RelationEvidence
from .language_spine_live_integration import (
    LANGUAGE_EVIDENCE_CANONICAL,
    LANGUAGE_EVIDENCE_D157_FALLBACK,
    RELATION_SOURCE_AGREEMENT,
    RELATION_SOURCE_CANONICAL_ONLY,
    RELATION_SOURCE_CONFLICT_ABSTAINED,
    RELATION_SOURCE_D157_ONLY,
    RELATION_SOURCE_MISSING,
    LiveLanguageSpineEvidence,
    fuse_relation_evidence,
    language_attempts_by_span_id_for_source,
    live_language_spine_diagnostics,
    proposition_candidate_ids_by_attempt_id_for,
    relation_evidence_by_proposition_pair,
)
from .structured_editorial_relation import (
    ATTEMPT_CONTINUATION,
    ATTEMPT_CORRECTION,
    ATTEMPT_RETRY,
    ATTEMPT_UNKNOWN,
    BEAT_NEW_AUDIENCE,
    BEAT_SAME,
    BEAT_UNKNOWN,
    GROUPING_ACTION_JOIN,
    GROUPING_ACTION_SPLIT,
    GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY,
    GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE,
    GROUPING_REASON_NO_PREDECESSOR,
    PROPOSITION_COMPLEMENTARY,
    PROPOSITION_DISTINCT,
    PROPOSITION_PROGRESSION,
    PROPOSITION_SAME,
    PROPOSITION_UNKNOWN,
    StructuredEditorialRelationEvidence,
    build_structured_editorial_relation,
    grouping_effective_relation as _grouping_effective_relation,
    structured_editorial_relation_diagnostics,
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


def _relation_evidence_confidence_for(
    relation_evidence_source: str, d157_confidence: str, canonical_confidence: str | None,
) -> str:
    """D-239U (docs/CUTSELL_DECISIONS.md D-239T/D-239U): the ALREADY-
    COMPUTED categorical confidence of whichever relation evidence
    ``fuse_relation_evidence`` actually used to decide ``relation_to_
    predecessor`` for this edge -- never a new relation-confidence
    computation, only a selection among values D-157 (``_dominant_
    relation``'s own second return value)/D-169 (``RelationEvidence.
    confidence``) already produced, mirroring ``_dominant_relation``'s own
    "selection among already-computed evidence" contract. Threaded into
    ``classify_editorial_moment``'s ``relation_confidence`` parameter so a
    RETRY/NEW_AUDIENCE_BEAT role's own ``role_evidence_confidence`` is
    sourced from the SAME relation evidence that established the role,
    never from the unrelated ``LanguageAttempt.confidence``."""
    if relation_evidence_source == RELATION_SOURCE_D157_ONLY:
        return d157_confidence
    if relation_evidence_source == RELATION_SOURCE_CANONICAL_ONLY:
        return canonical_confidence if canonical_confidence is not None else CONFIDENCE_UNKNOWN
    if relation_evidence_source == RELATION_SOURCE_AGREEMENT:
        candidates = [d157_confidence] + ([canonical_confidence] if canonical_confidence is not None else [])
        return max(candidates, key=lambda c: _CONFIDENCE_RANK.get(c, 0))
    if relation_evidence_source == RELATION_SOURCE_CONFLICT_ABSTAINED:
        return CONFIDENCE_MIXED
    return CONFIDENCE_UNKNOWN  # RELATION_SOURCE_MISSING, or any future value


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
    local_groups: Tuple["EditorialLocalGroup", ...] = ()
    # D-198: the exact ALREADY-COMPUTED dominant D-157 relation used by
    # D-197's own grouper for moments[i]'s predecessor edge (or ``None``
    # for the source's first moment, or when no relation was resolved) --
    # index-aligned with ``moments``. Diagnostic only: no downstream
    # authority reads this; it is the SAME value (never re-derived, never
    # re-ranked) build_editorial_local_groups already consumed to decide
    # membership. See docs/CUTSELL_DECISIONS.md D-198.
    moment_relation_to_predecessor: Tuple[str | None, ...] = ()
    # D-199: index-aligned per-moment sourcing diagnostics -- which
    # evidence source actually produced this moment's own LanguageAttempt
    # (CANONICAL_LANGUAGE_SPINE / D157_FALLBACK) and its relation-to-
    # predecessor value (D157_ONLY / CANONICAL_ONLY / AGREEMENT /
    # CONFLICT_ABSTAINED / MISSING). Diagnostic only. See
    # docs/CUTSELL_DECISIONS.md D-199.
    moment_language_evidence_source: Tuple[str | None, ...] = ()
    moment_relation_evidence_source: Tuple[str, ...] = ()
    # D-200.3: index-aligned dimension-aware structured relation evidence
    # (``None`` for a source's first moment) plus the D-197-grouping-only
    # translation actually consumed for that edge -- diagnostic only, no
    # downstream authority reads this. See docs/CUTSELL_DECISIONS.md D-200.3.
    moment_structured_relation: Tuple[StructuredEditorialRelationEvidence | None, ...] = ()
    moment_grouping_effective_relation: Tuple[str | None, ...] = ()
    moment_grouping_action: Tuple[str, ...] = ()
    moment_grouping_reason: Tuple[str, ...] = ()


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
    relation_evidence_by_pair: Mapping[Tuple[str, str], "RelationEvidence"] | None = None,
    provenance_out: dict | None = None,
) -> Tuple[Tuple[EditorialMoment, ...], int, int, dict[int, str]]:
    """Builds one ``EditorialMoment`` per eligible take, in deterministic
    ``(source_start, source_end, clip_id)`` order. A take with NO matching
    ``UnderstandingSpan`` is SKIPPED (never manufactured) -- per this
    task's own "do not create moments from arbitrary token windows /
    ambiguous source mapping" instruction. Returns ``(moments,
    unresolved_count, fallback_language_attempt_count, relation_by_
    position)`` -- ``relation_by_position`` is a ``{local_sequence_
    position: relation_candidate}`` map of every ACTUALLY-RESOLVED
    (non-``None``) FUSED dominant relation, for the caller to pass
    straight into ``build_editorial_sequences_for_moments`` without
    recomputing it.

    D-199 (docs/CUTSELL_DECISIONS.md D-199): the RETURN ARITY of this
    function is UNCHANGED from D-198 -- every existing caller (including
    the full D-194/D-195/D-197/D-198 test suites, which unpack exactly 4
    values) keeps working unmodified with both new flags OFF, per the
    "default OFF must preserve current D-198 behavior exactly" contract.
    The two new D-199 provenance maps (``relation_evidence_source_by_
    position``, ``attempt_source_by_position``) are exposed ONLY via the
    optional ``provenance_out`` mutable-dict out-parameter -- when the
    caller passes a ``dict``, this function populates it with those two
    keys; when omitted (``None``, the default), no extra work/allocation
    beyond the two local dicts already needed internally, and no ambient
    change to any existing caller.

    ``relation_evidence_by_pair`` is the OPTIONAL real canonical D-169
    ``RelationEvidence`` (keyed by proposition-candidate-id pair, see
    ``language_spine_live_integration.relation_evidence_by_proposition_
    pair``) -- when a canonical relation is found for a predecessor edge,
    it is FUSED with the existing D-157 Watch+Listen relation via
    ``language_spine_live_integration.fuse_relation_evidence`` (agreement/
    conflict-abstention/single-source contract, never a silent majority
    vote or override -- see that module's own docstring). This changes
    WHICH relation value reaches D-197's own unchanged join/split rule
    set; it does not change that rule set itself."""
    ordered = sorted(
        (t for t in takes_for_source if t.source_asset_id == source_asset_id),
        key=lambda t: (t.start, t.end, t.clip_id),
    )
    language_attempts_by_span_id = language_attempts_by_span_id or {}
    proposition_candidate_ids_by_attempt_id = proposition_candidate_ids_by_attempt_id or {}
    prosodic_evidence_by_span_id = prosodic_evidence_by_span_id or {}
    relation_evidence_by_pair = relation_evidence_by_pair or {}

    moments: list[EditorialMoment] = []
    relation_by_position: dict[int, str] = {}
    relation_evidence_source_by_position: dict[int, str] = {}
    attempt_source_by_position: dict[int, str] = {}
    # D-200.3 (docs/CUTSELL_DECISIONS.md D-200.3): dimension-aware relation
    # evidence, ALWAYS computed (pure, no I/O) alongside the pre-existing
    # flat fusion above -- never REPLACING `relation_to_predecessor`'s own
    # value fed to `classify_editorial_moment` below (D-194's moment-role
    # classification stays byte-identical, per this task's own "Do NOT
    # modify D-194 classification" instruction). `grouping_relation_by_
    # position` carries the SEPARATE, D-197-grouping-only translation
    # (`structured_editorial_relation.grouping_effective_relation`) that
    # the caller (`build_editorial_moment_understanding_for_source`) may
    # choose to feed into `build_editorial_local_groups` INSTEAD OF
    # `relation_by_position`, when the live Language-Spine diagnostics
    # flag is on (see that function's own docstring for the exact gate).
    structured_relation_by_position: dict[int, StructuredEditorialRelationEvidence | None] = {}
    grouping_relation_by_position: dict[int, str] = {}
    grouping_action_by_position: dict[int, str] = {}
    grouping_reason_by_position: dict[int, str] = {}
    unresolved_count = 0
    fallback_language_attempt_count = 0
    position = 0
    previous_relation_lookup_span: UnderstandingSpan | None = None
    previous_attempt: LanguageAttempt | None = None
    for take in ordered:
        span = _understanding_span_for_take(take, understanding_spans_by_id)
        if span is None or not take.clip_id or take.start is None or take.end is None or take.end < take.start:
            unresolved_count += 1
            continue

        d157_relation, d157_relation_confidence = (
            _dominant_relation(span.attempt_relation_hypotheses) if previous_relation_lookup_span is not None
            else (None, CONFIDENCE_UNKNOWN)
        )

        real_attempt = language_attempts_by_span_id.get(take.clip_id)
        if real_attempt is not None:
            attempt = real_attempt
            attempt_source_by_position[position] = LANGUAGE_EVIDENCE_CANONICAL
        else:
            attempt = _derive_language_attempt(take, span, d157_relation)
            fallback_language_attempt_count += 1
            attempt_source_by_position[position] = LANGUAGE_EVIDENCE_D157_FALLBACK

        # D-199: canonical D-169 relation lookup, via each side's own real
        # proposition_candidate_id -- only resolvable when BOTH sides used
        # a real canonical LanguageAttempt (a fallback-derived attempt's
        # id never appears in a real PropositionCandidate's attempt_ids,
        # so this is automatically None whenever either side fell back --
        # no extra branching needed for that case).
        canonical_relation = None
        canonical_relation_evidence_obj = None
        if previous_attempt is not None:
            pred_props = proposition_candidate_ids_by_attempt_id.get(previous_attempt.attempt_id, ())
            cur_props = proposition_candidate_ids_by_attempt_id.get(attempt.attempt_id, ())
            if pred_props and cur_props:
                evidence = relation_evidence_by_pair.get((pred_props[0], cur_props[0]))
                if evidence is not None:
                    canonical_relation = evidence.relation_candidate
                    canonical_relation_evidence_obj = evidence

        if previous_relation_lookup_span is not None:
            relation_to_predecessor, relation_evidence_source = fuse_relation_evidence(d157_relation, canonical_relation)
        else:
            relation_to_predecessor, relation_evidence_source = None, RELATION_SOURCE_MISSING
        relation_evidence_source_by_position[position] = relation_evidence_source
        # D-239U (docs/CUTSELL_DECISIONS.md D-239T/D-239U): the already-
        # computed confidence of whichever relation evidence just decided
        # `relation_to_predecessor` above -- fed to `classify_editorial_
        # moment`'s `relation_confidence` parameter ONLY, so a RETRY/
        # NEW_AUDIENCE_BEAT role's `role_evidence_confidence` is sourced
        # from the SAME relation evidence, never a new computation.
        relation_role_evidence_confidence = _relation_evidence_confidence_for(
            relation_evidence_source, d157_relation_confidence,
            canonical_relation_evidence_obj.confidence if canonical_relation_evidence_obj is not None else None,
        )

        # D-200.3: dimension-aware structured evidence for this predecessor
        # edge -- pure, no re-derivation of any D-157/D-169-internal
        # computation (reads only `span.attempt_relation_hypotheses`, the
        # SAME tuple `_dominant_relation` above already read, and the SAME
        # `canonical_relation_evidence_obj` looked up above). `None` for
        # the source's first moment (no predecessor) -- never fabricated.
        if previous_relation_lookup_span is not None:
            structured_relation = build_structured_editorial_relation(
                left_source_span_id=previous_relation_lookup_span.span_id,
                right_source_span_id=span.span_id,
                d157_hypotheses=span.attempt_relation_hypotheses,
                canonical_relation_evidence=canonical_relation_evidence_obj,
            )
        else:
            structured_relation = None
        structured_relation_by_position[position] = structured_relation
        grouping_value, grouping_action, grouping_reason = _grouping_effective_relation(structured_relation)
        grouping_action_by_position[position] = grouping_action
        grouping_reason_by_position[position] = grouping_reason
        if grouping_value is not None:
            grouping_relation_by_position[position] = grouping_value

        visual_reset_present = span.exit_usability in (USABILITY_UNUSABLE, USABILITY_QUESTIONABLE)

        moment = classify_editorial_moment(
            attempt,
            source_span_id=take.clip_id,
            proposition_candidate_ids=proposition_candidate_ids_by_attempt_id.get(attempt.attempt_id, ()),
            related_span_ids=(previous_relation_lookup_span.span_id,) if previous_relation_lookup_span is not None else (),
            behavior_hypotheses=span.behavior_state_hypotheses,
            relation_to_predecessor=relation_to_predecessor,
            relation_confidence=relation_role_evidence_confidence,
            local_sequence_position=position,
            prosodic_evidence=prosodic_evidence_by_span_id.get(take.clip_id),
            visual_reset_present=visual_reset_present,
        )
        if relation_to_predecessor is not None:
            relation_by_position[position] = relation_to_predecessor
        moments.append(moment)
        previous_relation_lookup_span = span
        previous_attempt = attempt
        position += 1

    if provenance_out is not None:
        provenance_out["relation_evidence_source_by_position"] = relation_evidence_source_by_position
        provenance_out["attempt_source_by_position"] = attempt_source_by_position
        # D-200.3: dimension-aware evidence + the D-197-grouping-only
        # translation, exposed the SAME optional-out-parameter way as the
        # two D-199 maps above -- zero extra cost for any caller that
        # doesn't pass `provenance_out` (unchanged for every pre-D-200.3
        # caller/test that unpacks the 4-tuple return directly).
        provenance_out["structured_relation_by_position"] = structured_relation_by_position
        provenance_out["grouping_relation_by_position"] = grouping_relation_by_position
        provenance_out["grouping_action_by_position"] = grouping_action_by_position
        provenance_out["grouping_reason_by_position"] = grouping_reason_by_position

    return (
        tuple(moments), unresolved_count, fallback_language_attempt_count, relation_by_position,
    )


# ---------------------------------------------------------------------------
# D-197: P1 LOCAL SEQUENCE GROUP FORMATION.
#
# D-196's real-media diagnostic RAW (docs/CUTSELL_DECISIONS.md D-196) proved
# ``build_editorial_sequences_for_moments``'s own whole-source default
# (below) is genuinely reached by the live pipeline and produces exactly
# the failure mode the D-196 directive named in advance:
# LOCAL_GROUPING_TOO_BROAD (one 353s sequence spanning topically unrelated
# real regions). This section replaces that whole-source default, for the
# live-pipeline entry point only (``build_editorial_moment_understanding_
# for_source`` below), with a deterministic, STRUCTURAL grouper.
#
# Forensic audit of which ALREADY-COMPUTED relation values mean what (per
# ``watch_listen_understanding._relation_for_pair``'s own real derivation,
# read directly rather than assumed):
#
#   RELATION_RETRY        -- lexical restart + prior-attempt abandonment/
#                             reset/pause evidence: this moment IS a retry
#                             of its predecessor. Same local recording
#                             structure. JOIN.
#   RELATION_CORRECTION   -- lexical restart immediately after an
#                             apparently-complete prior statement, no
#                             abandonment/reset evidence: this moment
#                             corrects its predecessor. Same local
#                             recording structure. JOIN.
#   RELATION_CONTINUATION -- no restart; prior span incomplete/non-
#                             terminal; tight gap; no measured pause: this
#                             moment continues its predecessor's own
#                             utterance. Same local recording structure.
#                             JOIN.
#   RELATION_COMPLEMENTARY-- "no semantic corroboration available to
#                             confirm non-duplicative content" (the
#                             deriver's own docstring) -- explicitly
#                             HEDGED, never-confirmed evidence. Per this
#                             task's own "ambiguous relation must not
#                             bridge... prefer bounded abstention" rule:
#                             BOUNDARY (never a join).
#   RELATION_NEW_AUDIENCE_BEAT -- prior span cleanly completed, no
#                             restart, fresh delivery start after a real
#                             gap: an explicit new-beat marker. Per this
#                             task's own "EXPLICIT BOUNDARY RELATIONS"
#                             section: BOUNDARY.
#   RELATION_DISTINCT_PROPOSITION -- never emitted by any live deriver in
#                             this codebase today (confirmed: absent from
#                             ``_relation_for_pair``'s own body) -- but by
#                             name it asserts the two propositions are
#                             distinct, which is structurally a boundary,
#                             not a join. BOUNDARY.
#   RELATION_UNCERTAIN / no # "insufficient evidence to support any bounded
#   relation at all         relation hypothesis" (the deriver's own
#                             docstring) or no predecessor relation
#                             resolved at all: the genuinely ambiguous/
#                             absent case. Per this task's own "NO
#                             RELATION CASE" firewall: BOUNDARY (never
#                             fabricate a join from chronological
#                             adjacency alone).
#
# This yields exactly the three JOIN relations the directive's own
# "EXPLICIT POSITIVE RELATIONS" section names as audit candidates.
# ---------------------------------------------------------------------------
_JOIN_RELATIONS: frozenset[str] = frozenset({RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION})

GROUP_REASON_SOLE_MOMENT_IN_SOURCE = "SOLE_MOMENT_IN_SOURCE"
GROUP_REASON_ISOLATED_NO_RELATION_EVIDENCE = "ISOLATED_NO_RELATION_EVIDENCE"
GROUP_REASON_RELATION_LINKED_CHAIN = "RELATION_LINKED_CHAIN"
ALLOWED_GROUPING_REASONS: frozenset[str] = frozenset({
    GROUP_REASON_SOLE_MOMENT_IN_SOURCE, GROUP_REASON_ISOLATED_NO_RELATION_EVIDENCE,
    GROUP_REASON_RELATION_LINKED_CHAIN,
})


def _local_group_id(source_asset_id: str, moment_ids: Sequence[str]) -> str:
    """Deterministic, MEMBERSHIP-anchored id -- same shape as D-194's own
    ``_editorial_sequence_id`` (sorted member id set, never order- or
    timestamp-anchored, never a clip/family/dict-order dependency) under a
    distinct ``elgrp_`` prefix."""
    raw = "|".join((source_asset_id, "|".join(sorted(str(v) for v in moment_ids if v)))).encode("utf-8")
    return "elgrp_" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class EditorialLocalGroup:
    """D-197: one bounded, deterministic LOCAL group of ``EditorialMoment``
    indices from the SAME per-source ordered moment list -- structural
    evidence only, never a semantic/authority decision. Grouping decides
    ONLY which moments are locally considered together; it never decides
    ``sequence_kind``, winner, deletion, family, or global (cross-source)
    redundancy -- see ``build_editorial_local_groups``'s own docstring."""
    source_asset_id: str
    group_id: str
    moment_indices: Tuple[int, ...]
    moment_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    grouping_reason: str
    relation_support: Tuple[str, ...]
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _group_confidence(group_moments: Sequence[EditorialMoment]) -> str:
    """Categorical only, never averaged -- mirrors ``_aggregate_confidence``
    below's own contract at the group level."""
    if any(m.conflict_flags for m in group_moments):
        return CONFIDENCE_MIXED
    confidences = {m.confidence for m in group_moments}
    if confidences == {CONFIDENCE_SUPPORTED}:
        return CONFIDENCE_SUPPORTED
    if CONFIDENCE_MIXED in confidences:
        return CONFIDENCE_MIXED
    return CONFIDENCE_WEAK


def _emit_local_group(
    groups: list[EditorialLocalGroup],
    moments: Tuple[EditorialMoment, ...],
    indices: Sequence[int],
    relations: Sequence[str],
) -> None:
    group_moments = [moments[i] for i in indices]
    moment_ids = tuple(m.editorial_moment_id for m in group_moments)
    source_asset_id = group_moments[0].source_asset_id
    if len(indices) == 1:
        reason = (
            GROUP_REASON_SOLE_MOMENT_IN_SOURCE if len(moments) == 1
            else GROUP_REASON_ISOLATED_NO_RELATION_EVIDENCE
        )
        confidence = CONFIDENCE_UNKNOWN
    else:
        reason = GROUP_REASON_RELATION_LINKED_CHAIN
        confidence = _group_confidence(group_moments)
    conflict_flags = tuple(sorted({f for m in group_moments for f in m.conflict_flags}))
    provenance = ["EDITORIAL_MOMENT_CLASSIFICATION"]
    if relations:
        provenance.append("RELATION_EVIDENCE")
    groups.append(EditorialLocalGroup(
        source_asset_id=source_asset_id,
        group_id=_local_group_id(source_asset_id, moment_ids),
        moment_indices=tuple(indices),
        moment_ids=moment_ids,
        source_start=min(m.source_start for m in group_moments),
        source_end=max(m.source_end for m in group_moments),
        grouping_reason=reason,
        relation_support=tuple(sorted(set(relations))),
        confidence=confidence,
        conflict_flags=conflict_flags,
        provenance=tuple(provenance),
    ))


def build_editorial_local_groups(
    moments: Tuple[EditorialMoment, ...],
    *,
    relation_candidates_by_position: Mapping[int, str] | None = None,
) -> Tuple[EditorialLocalGroup, ...]:
    """D-197's one canonical local-group builder. Pure; no I/O, no
    provider/media/network call, no global (cross-source) search, no
    Family Formation, no numeric adjacency threshold anywhere.

    Chains CONSECUTIVE moments (already caller-ordered, one source) into
    one local group exactly when the ALREADY-COMPUTED dominant D-157
    relation from a moment to its immediate predecessor
    (``relation_candidates_by_position`` -- the exact map ``build_
    editorial_moments_for_source`` already returns, never recomputed
    here) is one of RETRY/CORRECTION/CONTINUATION (see this section's own
    module-level audit comment above). Every other relation value, or a
    missing/unresolved one, is a BOUNDARY -- chronological adjacency
    alone never joins two moments.

    Each position's relation refers only to its own immediate
    predecessor (D-157's own per-pair design), so this is already an
    ordered linear chain, never a general graph -- no transitive-closure
    search is performed or needed (this task's own "connected component /
    chain design" audit conclusion).

    A standalone moment (no join to predecessor or successor) is returned
    as its own ``EditorialLocalGroup`` of size 1 -- singletons are valid
    P1 output, never forced into a fabricated 2-moment group."""
    relation_candidates_by_position = relation_candidates_by_position or {}
    if not moments:
        return ()

    groups: list[EditorialLocalGroup] = []
    current_indices: list[int] = [0]
    current_relations: list[str] = []
    for i in range(1, len(moments)):
        relation = relation_candidates_by_position.get(i)
        if relation in _JOIN_RELATIONS:
            current_indices.append(i)
            current_relations.append(relation)
        else:
            _emit_local_group(groups, moments, current_indices, current_relations)
            current_indices = [i]
            current_relations = []
    _emit_local_group(groups, moments, current_indices, current_relations)
    return tuple(groups)


def editorial_local_group_diagnostics(group: EditorialLocalGroup) -> dict:
    """Bounded, JSON-safe projection -- same verbatim-field-only contract
    as ``editorial_moment_diagnostics``/``editorial_sequence_diagnostics``.
    ``moment_ids`` are stable-id references only, never the underlying
    ``EditorialMoment`` objects; no transcript."""
    return {
        "group_id": group.group_id,
        "source_asset_id": group.source_asset_id,
        "source_start": group.source_start,
        "source_end": group.source_end,
        "moment_ids": list(group.moment_ids),
        "moment_count": len(group.moment_ids),
        "grouping_reason": group.grouping_reason,
        "relation_support": list(group.relation_support),
        "confidence": group.confidence,
        "conflict": list(group.conflict_flags),
        "provenance": list(group.provenance),
    }


def build_editorial_sequences_for_moments(
    moments: Tuple[EditorialMoment, ...],
    *,
    relation_candidates_by_position: Mapping[int, str] | None = None,
    local_groups: Sequence[Sequence[int]] | None = None,
) -> Tuple[EditorialSequenceHypothesis, ...]:
    """Forms bounded LOCAL sequences only. With no ``local_groups``
    supplied, this treats the WHOLE per-source moment list (already
    bounded to one source_asset_id by the caller) as ONE local window --
    the ORIGINAL D-195 default, preserved here unchanged for callers that
    still rely on it directly. The live pipeline no longer relies on this
    default: ``build_editorial_moment_understanding_for_source`` below
    always supplies REAL, structurally-computed groups (D-197's own
    ``build_editorial_local_groups``) instead of leaving this ``None`` --
    see D-196/D-197 (``docs/CUTSELL_DECISIONS.md``) for why the
    whole-source default was never safe as a live default. A caller MAY
    still supply explicit ``local_groups`` (index tuples into ``moments``)
    for a finer partition; this module still performs no search of its
    own."""
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
    live_language_spine: LiveLanguageSpineEvidence | None = None,
) -> EditorialMomentUnderstanding:
    """The one canonical per-source P1 Phase B builder. Pure; no I/O, no
    provider call, no perception recomputation (see module docstring).
    ``proposition_candidates``/``relation_evidence`` are accepted as
    OPTIONAL real D-169 collections for future callers that already have
    them -- when given, their ids are folded in as reference evidence
    (``proposition_candidate_ids_by_attempt_id``); when absent (the
    current live-pipeline default), moments simply carry no proposition
    references, honestly reported via ``missing_evidence``.

    D-199 (docs/CUTSELL_DECISIONS.md D-199): ``live_language_spine``, when
    supplied, is the SINGLE canonical source for all three of
    ``language_attempts_by_span_id``/``proposition_candidates``/
    ``relation_evidence`` -- it OVERRIDES those three parameters (never
    mixes canonical and caller-supplied values for the same source,
    preventing two competing linguistic truths). Real ``LanguageAttempt``
    objects are bridged onto this source's own ``UnderstandingSpan``s via
    ``language_spine_live_integration.language_attempts_by_span_id_for_
    source``'s deterministic maximum-overlap match -- a span with no
    overlapping real attempt simply falls through to the existing D-157
    fallback (``_derive_language_attempt``, unchanged), exactly the
    FALLBACK CONTRACT that module's own docstring specifies. When
    ``live_language_spine`` is ``None`` (the default -- the D-199 live-
    language-spine flag OFF, or construction unavailable/failed for this
    source), behavior is BYTE-IDENTICAL to pre-D-199."""
    missing_evidence: list[str] = []

    if live_language_spine is not None and watch_listen_understanding is not None:
        language_attempts_by_span_id = language_attempts_by_span_id_for_source(
            watch_listen_understanding.understanding_spans, live_language_spine.attempts,
        )
        proposition_candidates = live_language_spine.proposition_candidates
        relation_evidence = live_language_spine.relation_evidence

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

    relation_evidence_by_pair = (
        relation_evidence_by_proposition_pair(tuple(relation_evidence)) if relation_evidence is not None else {}
    )

    provenance_out: dict = {}
    moments, unresolved_count, fallback_count, relation_by_position = build_editorial_moments_for_source(
        source_asset_id=source_asset_id,
        takes_for_source=takes_tuple,
        understanding_spans_by_id=understanding_spans_by_id,
        language_attempts_by_span_id=language_attempts_by_span_id,
        proposition_candidate_ids_by_attempt_id=proposition_candidate_ids_by_attempt_id,
        prosodic_evidence_by_span_id=prosodic_evidence_by_span_id,
        relation_evidence_by_pair=relation_evidence_by_pair,
        provenance_out=provenance_out,
    )
    attempt_source_by_position = provenance_out.get("attempt_source_by_position", {})
    relation_evidence_source_by_position = provenance_out.get("relation_evidence_source_by_position", {})
    structured_relation_by_position = provenance_out.get("structured_relation_by_position", {})
    grouping_relation_by_position = provenance_out.get("grouping_relation_by_position", {})
    grouping_action_by_position = provenance_out.get("grouping_action_by_position", {})
    grouping_reason_by_position = provenance_out.get("grouping_reason_by_position", {})

    # D-200.3 (docs/CUTSELL_DECISIONS.md D-200.3): the live D-199 path
    # (both the P1 diagnostics flag AND the live Language-Spine diagnostics
    # flag on -- the SAME condition `live_language_spine is not None`
    # already gates every other D-199 canonical-vs-fallback override above)
    # feeds D-197's grouper the NEW dimension-aware `grouping_relation_by_
    # position` map instead of the flat-fused `relation_by_position` map --
    # this changes WHICH relation value reaches `build_editorial_local_
    # groups`'s own UNCHANGED join/split rule set, never that rule set
    # itself (same posture D-199's own docstring already established for
    # the flat-fusion case). `relation_by_position` (flat-fused) keeps
    # feeding `classify_editorial_moment`'s own `relation_to_predecessor`
    # argument (moment-role classification, D-194, byte-identical) and
    # `build_editorial_sequences_for_moments` (sequence classification,
    # D-194, byte-identical) UNCHANGED in both cases -- only P1's own
    # LOCAL GROUPING input differs. When `live_language_spine` is `None`
    # (the D-198 legacy default, or the live-spine flag off), grouping
    # uses `relation_by_position` exactly as before D-200.3 -- BYTE-
    # IDENTICAL to pre-D-200.3 behavior.
    grouping_relation_source = (
        grouping_relation_by_position if live_language_spine is not None else relation_by_position
    )

    if not moments:
        capability_status = CAPABILITY_NOT_EVALUABLE
    elif unresolved_count > 0:
        capability_status = CAPABILITY_PARTIAL
    else:
        capability_status = CAPABILITY_AVAILABLE

    # D-197: when the caller leaves ``local_groups`` unset (the live
    # pipeline's own default -- see ``build_editorial_moment_understanding_
    # for_sources`` below), compute REAL structural local groups instead of
    # ever falling through to ``build_editorial_sequences_for_moments``'s
    # own whole-source default. A caller that explicitly supplies
    # ``local_groups`` keeps that override verbatim (unchanged contract).
    if local_groups is None:
        computed_local_groups = build_editorial_local_groups(
            moments, relation_candidates_by_position=grouping_relation_source,
        )
        effective_local_groups: list[list[int]] = [
            list(g.moment_indices) for g in computed_local_groups if len(g.moment_indices) >= 2
        ]
    else:
        computed_local_groups = ()
        effective_local_groups = list(local_groups)

    sequences = build_editorial_sequences_for_moments(
        moments, relation_candidates_by_position=relation_by_position, local_groups=effective_local_groups,
    )

    # D-198: index-aligned pass-through of the SAME relation_by_position map
    # build_editorial_local_groups already consumed above -- no re-derivation,
    # no re-ranking. ``None`` for the source's first moment or any position
    # with no resolved dominant relation (the exact same semantics D-197's
    # own grouper already treats as a boundary).
    moment_relation_to_predecessor = tuple(
        relation_by_position.get(i) for i in range(len(moments))
    )
    # D-199: index-aligned pass-through of the two new per-position fusion/
    # source dicts build_editorial_moments_for_source now returns -- same
    # pure-serialization pattern as moment_relation_to_predecessor (D-198)
    # above, no re-derivation.
    moment_language_evidence_source = tuple(
        attempt_source_by_position.get(i) for i in range(len(moments))
    )
    moment_relation_evidence_source = tuple(
        relation_evidence_source_by_position.get(i, RELATION_SOURCE_MISSING) for i in range(len(moments))
    )
    # D-200.3: index-aligned pass-through of the four new per-position
    # dicts build_editorial_moments_for_source now returns -- same pure-
    # serialization pattern as D-198/D-199 above, no re-derivation.
    moment_structured_relation = tuple(
        structured_relation_by_position.get(i) for i in range(len(moments))
    )
    moment_grouping_effective_relation = tuple(
        grouping_relation_by_position.get(i) for i in range(len(moments))
    )
    moment_grouping_action = tuple(
        grouping_action_by_position.get(i, GROUPING_ACTION_SPLIT) for i in range(len(moments))
    )
    moment_grouping_reason = tuple(
        grouping_reason_by_position.get(i, GROUPING_REASON_NO_PREDECESSOR) for i in range(len(moments))
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
        local_groups=computed_local_groups,
        moment_relation_to_predecessor=moment_relation_to_predecessor,
        moment_language_evidence_source=moment_language_evidence_source,
        moment_relation_evidence_source=moment_relation_evidence_source,
        moment_structured_relation=moment_structured_relation,
        moment_grouping_effective_relation=moment_grouping_effective_relation,
        moment_grouping_action=moment_grouping_action,
        moment_grouping_reason=moment_grouping_reason,
    )


def build_editorial_moment_understanding_for_sources(
    *,
    sources: Iterable[str],
    takes: Iterable[CandidateTake],
    watch_listen_understandings: Iterable[WatchListenUnderstanding],
    raw_understanding_maps: Iterable[RawUnderstandingMap] = (),
    live_language_spine_by_source: Mapping[str, LiveLanguageSpineEvidence] | None = None,
    **kwargs,
) -> Tuple[EditorialMomentUnderstanding, ...]:
    """Batch form, one per source, in ``sources`` order -- deterministic
    regardless of ``takes``/``watch_listen_understandings`` input order.
    ``live_language_spine_by_source`` (D-199): optional ``source_asset_id
    -> LiveLanguageSpineEvidence`` map, looked up per source and passed
    straight through as ``build_editorial_moment_understanding_for_
    source``'s own ``live_language_spine`` parameter -- absent for a given
    source is identical to ``None`` (byte-identical pre-D-199 behavior for
    that source)."""
    takes = tuple(takes)
    wlu_by_source = {u.source_asset_id: u for u in watch_listen_understandings}
    raw_by_source = {m.source_asset_id: m for m in raw_understanding_maps}
    live_language_spine_by_source = live_language_spine_by_source or {}
    out = []
    for source_asset_id in sources:
        out.append(build_editorial_moment_understanding_for_source(
            source_asset_id=source_asset_id,
            takes_for_source=takes,
            watch_listen_understanding=wlu_by_source.get(source_asset_id),
            raw_understanding_map=raw_by_source.get(source_asset_id),
            live_language_spine=live_language_spine_by_source.get(source_asset_id),
            **kwargs,
        ))
    return tuple(out)


def p1_moment_role_and_audience_status_by_clip_id_for(
    editorial_moment_understandings: Iterable[EditorialMomentUnderstanding],
) -> Tuple[dict[str, str], dict[str, str]]:
    """D-239I Seam C: trivial passthrough projection of each already-built
    ``EditorialMoment``'s own ``moment_role``/``audience_delivery_status``
    fields, keyed by ``source_span_id`` (== ``take.clip_id`` verbatim, per
    ``build_editorial_moments_for_source``'s own "reuse the REAL existing
    canonical clip identity" comment) -- never a re-derivation, never a
    new role/delivery classifier. Mirrors ``proposition_slot_evidence_by_
    id_for``'s own "trivial passthrough projection" pattern exactly.

    D-239U (docs/CUTSELL_DECISIONS.md D-239T/D-239U): a moment is included
    ONLY when its own ``role_evidence_confidence`` -- the confidence of
    the SPECIFIC evidence channel that established THIS moment's role,
    never the unrelated ``EditorialMoment.confidence`` -- is
    ``CONFIDENCE_SUPPORTED`` (never ``CONFIDENCE_MIXED``, a genuine
    conflict) AND its own ``moment_role`` is not ``MOMENT_ROLE_UNCERTAIN``.
    This is a REFINEMENT of this seam's own consumption, not a new gate:
    it still requires the ROLE itself to be sufficiently supported before
    Seam C hands it to any downstream authority; it changes only WHICH
    already-computed confidence answers that question (D-239T's own
    forensic finding: ``EditorialMoment.confidence`` can be UNKNOWN from
    an unrelated linguistic-boundary gap even when the role's OWN
    establishing evidence is strong). ``EditorialMoment.confidence``
    itself, every other seam, and every downstream consumer of these two
    maps (ownership/materiality/Freeze/RepairLoop) are unchanged -- an
    ambiguous/conflicted/role-unsupported moment is simply OMITTED from
    both maps, so a caller's plain ``.get(clip_id)`` lookup naturally
    yields ``None`` (UNKNOWN) for it, never a guessed role.
    ``source_span_id is None`` (no matching ``UnderstandingSpan`` for this
    take) is likewise omitted -- never falls back to any other id."""
    role_by_clip_id: dict[str, str] = {}
    audience_status_by_clip_id: dict[str, str] = {}
    for understanding in editorial_moment_understandings:
        for moment in understanding.moments:
            if moment.source_span_id is None:
                continue
            if moment.role_evidence_confidence != CONFIDENCE_SUPPORTED:
                continue
            if moment.moment_role == MOMENT_ROLE_UNCERTAIN:
                continue
            role_by_clip_id[moment.source_span_id] = moment.moment_role
            audience_status_by_clip_id[moment.source_span_id] = moment.audience_delivery_status
    return role_by_clip_id, audience_status_by_clip_id


# ---------------------------------------------------------------------------
# D-239O: EXACT LOST-ATOM TARGET -> P1 MOMENT OBSERVABILITY.
#
# D-239N's own forensic (docs/CUTSELL_DECISIONS.md D-239N) ruled out
# clip-id namespace mismatch, attempt-id substitution, and one-to-many
# moment construction as causes of an unresolved Seam C lookup, and
# narrowed the remaining gap to exactly two live, undistinguished
# mechanisms: (B) the target's own `CandidateTake` never received a
# matching `UnderstandingSpan` (so `build_editorial_moments_for_source`
# skipped it, and no `EditorialMoment` was ever minted), or (F) a moment
# WAS minted but its own `confidence`/`moment_role`/`audience_delivery_
# status` fails `p1_moment_role_and_audience_status_by_clip_id_for`'s own
# three-condition gate (see that function's own docstring). This section
# adds BEHAVIOR-NEUTRAL observability distinguishing exactly those two
# cases -- and the finer sub-cases within (B) -- for a BOUNDED set of
# caller-supplied target clip_ids (never a full P1 dump). It calls NO
# new classifier, recomputes NO role/confidence/audience value, and
# reuses the SAME already-computed `EditorialMomentUnderstanding`/
# `WatchListenUnderstanding` objects and the SAME already-computed
# `p1_moment_role_by_clip_id`/`p1_audience_delivery_status_by_clip_id`
# maps `p1_moment_role_and_audience_status_by_clip_id_for` already
# produces -- this function's own body contains zero classification
# logic of its own; it only READS already-classified fields and reports,
# per target, WHICH of the (mutually exclusive, ordered) reasons applies.
# ---------------------------------------------------------------------------
P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED = "MOMENT_FOUND_RESOLVED"
P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE = "MOMENT_FOUND_LOW_CONFIDENCE"
P1_TARGET_STATUS_MOMENT_FOUND_ROLE_UNCERTAIN = "MOMENT_FOUND_ROLE_UNCERTAIN"
P1_TARGET_STATUS_MOMENT_FOUND_AUDIENCE_UNCERTAIN = "MOMENT_FOUND_AUDIENCE_UNCERTAIN"
P1_TARGET_STATUS_NO_EDITORIAL_MOMENT = "NO_EDITORIAL_MOMENT"
P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN = "NO_UNDERSTANDING_SPAN"
P1_TARGET_STATUS_AMBIGUOUS = "AMBIGUOUS"
ALLOWED_P1_TARGET_LOOKUP_STATUSES: frozenset[str] = frozenset({
    P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED, P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE,
    P1_TARGET_STATUS_MOMENT_FOUND_ROLE_UNCERTAIN, P1_TARGET_STATUS_MOMENT_FOUND_AUDIENCE_UNCERTAIN,
    P1_TARGET_STATUS_NO_EDITORIAL_MOMENT, P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN,
    P1_TARGET_STATUS_AMBIGUOUS,
})

_EXACT_P1_TARGET_EVIDENCE_SCHEMA_VERSION = "cutsell.exact_p1_target_evidence.v1"


def exact_p1_target_evidence_for(
    target_source_asset_id_by_clip_id: Mapping[str, str],
    *,
    editorial_moment_understandings: Iterable[EditorialMomentUnderstanding],
    watch_listen_understandings: Iterable[WatchListenUnderstanding],
    p1_moment_role_by_clip_id: Mapping[str, str],
    p1_audience_delivery_status_by_clip_id: Mapping[str, str],
) -> dict:
    """D-239O: exact, per-target-clip_id P1 lookup observability. Pure;
    reads only already-computed objects, mints nothing, calls no
    classifier. `target_source_asset_id_by_clip_id` is the BOUNDED set of
    lost-atom-correlated clip_ids to report on (never every clip in the
    run) -- the caller (`pipeline.py`, the SAME `lost_atom_ownership_by_
    clip_id` population D-238/D-239F already build) decides scope; this
    function never discovers targets on its own.

    Correlates by exact, unmodified identity only: `target_source_asset_
    id_by_clip_id`'s own keys ARE `CandidateTake.clip_id`/`DraftClip.
    clip_id` (D-239N's own Stage 2 proof -- one unbroken identity chain),
    matched against `UnderstandingSpan.span_id` and `EditorialMoment.
    source_span_id` -- both, by construction, the SAME clip_id value
    space (D-239N's own Stage 5 proof). No alternate namespace (attempt_
    id, source_span_id-as-a-different-field, timestamp/overlap) is ever
    consulted for this correlation.

    `p1_target_lookup_status` is derived, per target, from an ordered
    sequence of already-computed facts (never re-derived, never a new
    threshold): ambiguous multi-moment match (defensive; D-239N's own
    Stage 1 proof says this should never occur) > no matching
    `UnderstandingSpan` at all (D-239N's own live candidate B) > a
    span existed but no `EditorialMoment` was built for it (a DIFFERENT,
    narrower shape than "no span at all" -- reported honestly, distinct
    from B, per this task's own "another exact structural reason" case)
    > a moment exists but its own `confidence != CONFIDENCE_SUPPORTED`
    (D-239N's own live candidate F) > a moment exists, confidence
    supported, but `moment_role == MOMENT_ROLE_UNCERTAIN` (also
    candidate F) > a moment exists, confidence supported, role resolved,
    but `audience_delivery_status` is `UNCERTAIN`/absent (a finer
    distinction than Seam C's own binary resolved/not-resolved -- Seam C
    itself does not gate on audience status, so `helper_lookup_resolved`
    can be `True` even when this exact status fires; D-239L's own
    corroboration check separately requires a resolved audience status,
    so this distinction explains a DIFFERENT downstream gate, never
    changes Seam C's own contract) > fully resolved.

    `helper_lookup_resolved`/`helper_lookup_role`/`helper_lookup_
    audience_delivery_status` are read DIRECTLY off the SAME `p1_moment_
    role_by_clip_id`/`p1_audience_delivery_status_by_clip_id` maps `p1_
    moment_role_and_audience_status_by_clip_id_for` already produced --
    never recomputed, so this diagnostic's own `helper_lookup_resolved`
    is provably consistent with that function's own real behavior (see
    this task's own consistency tests)."""
    understandings_by_source: dict[str, EditorialMomentUnderstanding] = {
        u.source_asset_id: u for u in editorial_moment_understandings
    }
    spans_by_source: dict[str, dict[str, UnderstandingSpan]] = {}
    for wlu in watch_listen_understandings:
        span_map = spans_by_source.setdefault(wlu.source_asset_id, {})
        for span in wlu.understanding_spans:
            span_map[span.span_id] = span

    rows: list[dict] = []
    for clip_id in sorted(target_source_asset_id_by_clip_id):
        source_asset_id = target_source_asset_id_by_clip_id[clip_id]
        span_map = spans_by_source.get(source_asset_id, {})
        understanding_span_present = clip_id in span_map
        understanding_span_id = span_map[clip_id].span_id if understanding_span_present else None

        understanding = understandings_by_source.get(source_asset_id)
        matching_moments = tuple(
            m for m in (understanding.moments if understanding is not None else ())
            if m.source_span_id == clip_id
        )
        editorial_moment_present = bool(matching_moments)
        ambiguous = len(matching_moments) > 1

        moment_id = source_span_id_out = local_group_id = None
        role = role_confidence = audience_status = recording_process_status = None
        role_evidence_source = role_evidence_confidence = None
        proposition_ids: Tuple[str, ...] = ()
        if matching_moments:
            moment = matching_moments[0]
            moment_id = moment.editorial_moment_id
            source_span_id_out = moment.source_span_id
            role = moment.moment_role
            # D-239U: `role_confidence` keeps its EXISTING meaning verbatim
            # -- `EditorialMoment.confidence` (the attempt/conflict
            # confidence, unchanged) -- honestly labelled, never silently
            # repurposed. `role_evidence_source`/`role_evidence_confidence`
            # are the NEW, additive, role-source-aligned fields (D-239T/
            # D-239U) answering the narrower "how strong is the evidence
            # that established THIS role" question.
            role_confidence = moment.confidence
            role_evidence_source = moment.role_evidence_source
            role_evidence_confidence = moment.role_evidence_confidence
            audience_status = moment.audience_delivery_status
            recording_process_status = moment.recording_process_status
            proposition_ids = tuple(moment.proposition_candidate_ids)
            if understanding is not None:
                for group in understanding.local_groups:
                    if moment_id in group.moment_ids:
                        local_group_id = group.group_id
                        break

        helper_lookup_resolved = clip_id in p1_moment_role_by_clip_id
        helper_lookup_role = p1_moment_role_by_clip_id.get(clip_id)
        helper_lookup_audience_delivery_status = p1_audience_delivery_status_by_clip_id.get(clip_id)

        if ambiguous:
            status = P1_TARGET_STATUS_AMBIGUOUS
            reason = "multiple_editorial_moments_share_this_exact_source_span_id"
        elif not understanding_span_present:
            status = P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
            reason = "no_understanding_span_matches_this_exact_clip_id"
        elif not editorial_moment_present:
            status = P1_TARGET_STATUS_NO_EDITORIAL_MOMENT
            reason = "understanding_span_present_but_no_editorial_moment_was_built_for_it"
        elif role_evidence_confidence != CONFIDENCE_SUPPORTED:
            # D-239U: mirrors Seam C's OWN refined gate (`role_evidence_
            # confidence`, not the unrelated `EditorialMoment.confidence`)
            # so this diagnostic's `helper_lookup_resolved` stays provably
            # consistent with the real seam's behavior -- see this
            # function's own docstring.
            status = P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE
            reason = "editorial_moment_present_but_role_evidence_confidence_is_not_CONFIDENCE_SUPPORTED"
        elif role == MOMENT_ROLE_UNCERTAIN:
            status = P1_TARGET_STATUS_MOMENT_FOUND_ROLE_UNCERTAIN
            reason = "editorial_moment_present_and_role_evidence_confidence_supported_but_role_is_UNCERTAIN"
        elif audience_status is None or audience_status == AUDIENCE_DELIVERY_UNCERTAIN:
            status = P1_TARGET_STATUS_MOMENT_FOUND_AUDIENCE_UNCERTAIN
            reason = "role_resolved_but_audience_delivery_status_uncertain_or_absent"
        else:
            status = P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
            reason = "exact_editorial_moment_resolved_for_this_clip_id"

        rows.append({
            "clip_id": clip_id,
            "source_asset_id": source_asset_id,
            "understanding_span_present": understanding_span_present,
            "understanding_span_id": understanding_span_id,
            "editorial_moment_present": editorial_moment_present,
            "moment_id": moment_id,
            "source_span_id": source_span_id_out,
            "local_group_id": local_group_id,
            "role": role,
            "role_confidence": role_confidence,
            "role_evidence_source": role_evidence_source,
            "role_evidence_confidence": role_evidence_confidence,
            "audience_delivery_status": audience_status,
            "recording_process_status": recording_process_status,
            "proposition_candidate_ids": list(proposition_ids),
            "helper_lookup_resolved": helper_lookup_resolved,
            "helper_lookup_role": helper_lookup_role,
            "helper_lookup_audience_delivery_status": helper_lookup_audience_delivery_status,
            "helper_lookup_reason": reason,
            "p1_target_lookup_status": status,
        })

    return {
        "schema_version": _EXACT_P1_TARGET_EVIDENCE_SCHEMA_VERSION,
        "target_count": len(rows),
        "targets": rows,
        "provenance": (_EXACT_P1_TARGET_EVIDENCE_SCHEMA_VERSION, "exact_p1_target_evidence_for"),
    }


# ---------------------------------------------------------------------------
# Diagnostics / run summary.
# ---------------------------------------------------------------------------
def _moment_diagnostics_with_relation(
    moment: EditorialMoment,
    relation_to_predecessor: str | None,
    language_evidence_source: str | None = None,
    relation_evidence_source: str | None = None,
    structured_relation: StructuredEditorialRelationEvidence | None = None,
    grouping_effective_relation_value: str | None = None,
    grouping_action: str | None = None,
    grouping_reason: str | None = None,
) -> dict:
    """D-198: `editorial_moment_diagnostics` (D-194, unchanged) plus ONE
    diagnostic-only key, `relation_to_predecessor` -- the exact
    already-computed value D-197's own grouper consumed for this moment's
    predecessor edge, never re-derived here. See docs/CUTSELL_DECISIONS.md
    D-198 ("SINGLE SOURCE OF TRUTH").

    D-199: two more diagnostic-only keys, `language_evidence_source`
    (CANONICAL_LANGUAGE_SPINE / D157_FALLBACK / ``None`` when the caller
    never supplied per-moment provenance) and `relation_evidence_source`
    (the ``fuse_relation_evidence`` provenance tag -- MISSING/D157_ONLY/
    CANONICAL_ONLY/AGREEMENT/CONFLICT_ABSTAINED) -- no full text, no new
    computation, pure pass-through.

    D-200.3: `structured_relation` (the dimension-aware evidence for this
    edge, projected via `structured_editorial_relation_diagnostics` --
    ``None`` for the source's first moment) plus `grouping_effective_
    relation`/`grouping_action`/`grouping_reason` -- the exact value/JOIN
    or SPLIT/reason `build_editorial_local_groups` actually consumed for
    THIS run (which may be the flat-fused value when the live Language-
    Spine flag is off -- see `build_editorial_moment_understanding_for_
    source`'s own gate). `relation_to_predecessor` above is UNCHANGED --
    it remains the value fed to D-194's own moment-role classification,
    never the grouping-effective one."""
    return {
        **editorial_moment_diagnostics(moment),
        "relation_to_predecessor": relation_to_predecessor,
        "language_evidence_source": language_evidence_source,
        "relation_evidence_source": relation_evidence_source,
        "structured_relation": structured_editorial_relation_diagnostics(structured_relation),
        "grouping_effective_relation": grouping_effective_relation_value,
        "grouping_action": grouping_action,
        "grouping_reason": grouping_reason,
    }


def editorial_moment_understanding_diagnostics(understanding: EditorialMomentUnderstanding) -> dict:
    relations = understanding.moment_relation_to_predecessor
    language_sources = understanding.moment_language_evidence_source
    relation_sources = understanding.moment_relation_evidence_source
    structured_relations = understanding.moment_structured_relation
    grouping_relations = understanding.moment_grouping_effective_relation
    grouping_actions = understanding.moment_grouping_action
    grouping_reasons = understanding.moment_grouping_reason
    moment_rows = [
        _moment_diagnostics_with_relation(
            m,
            relations[i] if i < len(relations) else None,
            language_sources[i] if i < len(language_sources) else None,
            relation_sources[i] if i < len(relation_sources) else None,
            structured_relations[i] if i < len(structured_relations) else None,
            grouping_relations[i] if i < len(grouping_relations) else None,
            grouping_actions[i] if i < len(grouping_actions) else None,
            grouping_reasons[i] if i < len(grouping_reasons) else None,
        )
        for i, m in enumerate(understanding.moments)
    ]
    return {
        "source_asset_id": understanding.source_asset_id,
        "moment_count": understanding.moment_count,
        "sequence_count": understanding.sequence_count,
        "capability_status": understanding.capability_status,
        "missing_evidence": list(understanding.missing_evidence),
        "confidence": understanding.confidence,
        "conflict": list(understanding.conflict_flags),
        "moments": moment_rows,
        "sequences": [editorial_sequence_diagnostics(s) for s in understanding.sequence_hypotheses],
        # D-197: real structural local groups (empty when the caller
        # bypassed the auto-grouper with its own explicit local_groups).
        "local_groups": [editorial_local_group_diagnostics(g) for g in understanding.local_groups],
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

    all_local_groups = tuple(g for u in understandings for g in u.local_groups)
    singleton_groups = tuple(g for g in all_local_groups if len(g.moment_ids) == 1)
    multi_moment_groups = tuple(g for g in all_local_groups if len(g.moment_ids) >= 2)
    sequenced_moment_ids = {mid for s in all_sequences for mid in s.moment_ids}
    unsequenced_moment_count = sum(1 for m in all_moments if m.editorial_moment_id not in sequenced_moment_ids)

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
        # D-197: local-group formation diagnostics. ``sequence_count`` is
        # the same value already reported as ``editorial_sequence_count``
        # (from the base D-194 summary) -- both names kept for readability
        # of this specific field group, never a second source of truth.
        "local_group_count": len(all_local_groups),
        "singleton_group_count": len(singleton_groups),
        "multi_moment_group_count": len(multi_moment_groups),
        "max_group_moment_count": max((len(g.moment_ids) for g in all_local_groups), default=0),
        "sequence_count": len(all_sequences),
        "sequence_from_supported_group_count": sum(
            1 for g in multi_moment_groups if g.confidence == CONFIDENCE_SUPPORTED
        ),
        "unsequenced_moment_count": unsequenced_moment_count,
        **_structured_relation_run_summary_counts(understandings),
    }


def _structured_relation_run_summary_counts(
    understandings: Tuple[EditorialMomentUnderstanding, ...],
) -> dict:
    """D-200.3 (docs/CUTSELL_DECISIONS.md D-200.3): bounded, counts-only
    aggregate over every source's already-built `moment_structured_
    relation`/`moment_grouping_action` tuples -- no re-derivation, no
    transcript. `proposition_same_count`/`proposition_progression_count`/
    `same_editorial_beat_count` may legitimately be `0` for every real run
    today -- see `structured_editorial_relation.py`'s own "no new evidence
    source" section; this is never "fixed" by inventing evidence here
    (this task's own "NO SAME_EDITORIAL_BEAT FICTION" instruction)."""
    edges: list[StructuredEditorialRelationEvidence] = [
        r for u in understandings for r in u.moment_structured_relation if r is not None
    ]
    actions: list[str] = [
        a for u in understandings for a in u.moment_grouping_action
    ]
    reasons: list[str] = [
        r for u in understandings for r in u.moment_grouping_reason
    ]

    def _cross_dimension_compatible(edge: StructuredEditorialRelationEvidence) -> bool:
        # At least two dimensions carry a genuinely resolved (non-UNKNOWN)
        # value AND no same-dimension conflict flag was raised for this
        # edge -- i.e. this edge is an example of "different supported
        # truths in different dimensions", never a global conflict
        # (D-200.2's own core claim, validated per-edge here).
        if edge.conflict_flags:
            return False
        resolved = sum((
            edge.attempt_relation != ATTEMPT_UNKNOWN,
            edge.proposition_relation != PROPOSITION_UNKNOWN,
            edge.editorial_beat_relation != BEAT_UNKNOWN,
        ))
        return resolved >= 2

    return {
        "structured_relation_edge_count": len(edges),
        "attempt_retry_count": sum(1 for e in edges if e.attempt_relation == ATTEMPT_RETRY),
        "attempt_correction_count": sum(1 for e in edges if e.attempt_relation == ATTEMPT_CORRECTION),
        "attempt_continuation_count": sum(1 for e in edges if e.attempt_relation == ATTEMPT_CONTINUATION),
        "attempt_unknown_count": sum(1 for e in edges if e.attempt_relation == ATTEMPT_UNKNOWN),
        "proposition_same_count": sum(1 for e in edges if e.proposition_relation == PROPOSITION_SAME),
        "proposition_distinct_count": sum(1 for e in edges if e.proposition_relation == PROPOSITION_DISTINCT),
        "proposition_complementary_count": sum(1 for e in edges if e.proposition_relation == PROPOSITION_COMPLEMENTARY),
        "proposition_progression_count": sum(1 for e in edges if e.proposition_relation == PROPOSITION_PROGRESSION),
        "proposition_unknown_count": sum(1 for e in edges if e.proposition_relation == PROPOSITION_UNKNOWN),
        "new_audience_beat_count": sum(1 for e in edges if e.editorial_beat_relation == BEAT_NEW_AUDIENCE),
        "same_editorial_beat_count": sum(1 for e in edges if e.editorial_beat_relation == BEAT_SAME),
        "editorial_beat_unknown_count": sum(1 for e in edges if e.editorial_beat_relation == BEAT_UNKNOWN),
        "cross_dimension_compatible_count": sum(1 for e in edges if _cross_dimension_compatible(e)),
        "same_dimension_conflict_count": sum(1 for e in edges if e.conflict_flags),
        "grouping_join_count": sum(1 for a in actions if a == GROUPING_ACTION_JOIN),
        "grouping_split_count": sum(1 for a in actions if a == GROUPING_ACTION_SPLIT),
        "grouping_split_new_beat_count": sum(1 for r in reasons if r == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY),
        "grouping_split_no_positive_join_count": sum(1 for r in reasons if r == GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE),
    }


def live_language_spine_source_diagnostics_for_p1(
    evidence: "LiveLanguageSpineEvidence", understanding: EditorialMomentUnderstanding,
) -> dict:
    """D-199 (docs/CUTSELL_DECISIONS.md D-199): the mandated per-source
    "diagnostics required per source" block -- ``live_language_spine_
    diagnostics`` (construction-only, D-199's own module) merged with
    FOUR P1-CONSUMPTION counts derived purely from this source's already-
    built ``EditorialMomentUnderstanding`` (no re-derivation, no new
    computation): how many of THIS source's moments actually used a real
    canonical attempt/proposition/relation versus the D-157 fallback.
    Pure re-projection, same pattern as every other D-19x diagnostics
    function in this module."""
    base = live_language_spine_diagnostics(evidence)
    canonical_attempt_used_by_p1_count = sum(
        1 for s in understanding.moment_language_evidence_source if s == LANGUAGE_EVIDENCE_CANONICAL
    )
    fallback_attempt_used_by_p1_count = sum(
        1 for s in understanding.moment_language_evidence_source if s == LANGUAGE_EVIDENCE_D157_FALLBACK
    )
    canonical_proposition_coverage_count = sum(
        1 for m in understanding.moments if m.proposition_candidate_ids
    )
    canonical_relation_coverage_count = sum(
        1 for s in understanding.moment_relation_evidence_source
        if s in (RELATION_SOURCE_CANONICAL_ONLY, RELATION_SOURCE_AGREEMENT, RELATION_SOURCE_CONFLICT_ABSTAINED)
    )
    return {
        **base,
        "canonical_attempt_used_by_p1_count": canonical_attempt_used_by_p1_count,
        "fallback_attempt_used_by_p1_count": fallback_attempt_used_by_p1_count,
        "canonical_proposition_coverage_count": canonical_proposition_coverage_count,
        "canonical_relation_coverage_count": canonical_relation_coverage_count,
    }
