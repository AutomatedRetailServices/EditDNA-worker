"""D-199: P1 LIVE LANGUAGE-SPINE EVIDENCE INTEGRATION.

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 14/15
and ``docs/CUTSELL_DECISIONS.md`` D-166/D-168/D-169/D-193/D-194/D-195/
D-197/D-198/D-199. This module is the ONE adapter layer between the
canonical Language Spine's already-existing, unchanged builders --

    LanguageWord (D-166) -> LanguagePhrase (D-166) ->
    LanguageUtterance (D-168) -> LanguageAttempt (D-168) ->
    PropositionCandidate (D-169) -> RelationEvidence (D-169)

-- and P1's own per-source integration layer
(``editorial_moment_sequence_integration.py``, D-195/D-197/D-198). It
mints NOTHING new: every object this module produces is built by a
DIRECT CALL into D-166/D-168/D-169's own already-vetted functions,
never a re-implementation, never a competing parser.

## THE TRANSCRIPT REMAINS THE LINGUISTIC SPINE (this task's own core
## principle, restated)

Watch+Listen (D-157) remains multimodal CORROBORATION; it is never the
source of linguistic structure. P1 previously had no live linguistic
structure to consume at all (D-195's own forensic finding: real
``LanguageAttempt``/``PropositionCandidate``/``RelationEvidence``
collections are not constructed source-wide anywhere in the live
pipeline path) and fell back to an honest, labelled APPROXIMATION
derived from D-157 evidence (``editorial_moment_sequence_integration.
_derive_language_attempt``, unchanged by this task). This module closes
that gap by constructing the REAL canonical objects, once per source,
directly from already-computed ASR word timings -- the fallback stays
available, unchanged, as the documented last resort (see "FALLBACK
CONTRACT" below).

## NO ASR RERUN (structurally true, not merely claimed)

This module imports NOTHING from ``asr.py`` and calls no transcription
provider/endpoint. Its only external input is ``RawUnderstandingMap.
word_timings`` (D-155, unchanged) -- the SAME ``contracts.Word`` tuple
the rest of the codebase already computed once. Confirmed by module-
leaf import-absence tests (``tests/test_cutsell_d199_language_spine_
live_integration.py``).

## Construction seam (once per source, never per clip/family/finalist)

``build_live_language_spine_for_source`` is the ONE per-source
construction entry point. It is a pure function of ``RawUnderstanding
Map.word_timings`` -- callers (``pipeline.py``) are responsible for
invoking it exactly once per real ``source_asset_id``; this module adds
no caching of its own (nothing here needs global mutable state) but the
test suite proves the SAME determinstic output on repeated calls with
identical input, and pipeline.py's own orchestration calls it once per
source in a dict comprehension (never per clip/family/finalist).

## Source identity (binding, this task's own requirement)

Every object below traces back to real RAW timing via the D-166/D-168/
D-169 builders' own existing id-minting (word_index/phrase_id/
utterance_id/attempt_id/proposition_candidate_id, all timestamp- or
membership-anchored, never a rendered-timeline or BestTake/family id).
This module mints NO id of its own for any Language Spine object --
only ``LiveLanguageSpineEvidence`` itself (a diagnostics container, not
a Language Spine rung) needs no id at all (it is keyed by
``source_asset_id`` at the call site).

## Bridging real Language Spine objects onto P1's take-keyed spans

P1's existing per-take/per-span architecture
(``editorial_moment_sequence_integration.build_editorial_moments_for_
source``) keys evidence by ``take.clip_id``/``UnderstandingSpan.
span_id`` -- boundaries D-155/D-157's own independent segmentation
produced. The live Language Spine's own ``LanguageAttempt`` boundaries
come from an INDEPENDENT segmentation (D-166/D-168's own word-timing-gap
+ structural-boundary algorithm) and will not, in general, align
exactly with those spans. ``language_attempts_by_span_id_for_source``
bridges the two via a deterministic MAXIMUM-OVERLAP match (ownership by
the single largest temporal overlap, ties broken by earliest attempt
start then attempt id) -- never a rendered-timeline or BestTake/family
identity, and never a partial/fuzzy text match. A span with no
temporally-overlapping real attempt at all gets no canonical mapping for
that span -- it falls through to the existing D-157 fallback
untouched (see "FALLBACK CONTRACT").

## FALLBACK CONTRACT (binding, this task's own required precedence)

    1. canonical live Language Spine (this module), when construction
       succeeded AND a real attempt overlaps the span;
    2. the existing D-157 fallback adapter (``editorial_moment_
       sequence_integration._derive_language_attempt``, unchanged),
       only when (1) is unavailable for that span;
    3. UNKNOWN/missing, when neither exists.

Never a silent mix of two competing linguistic truths for the SAME
span -- exactly one of the three applies per span, and which one is
recorded per moment (``attempt_language_evidence_source``, see
``editorial_moment_sequence_integration.py``'s own D-199 extension).

## RELATION CONSISTENCY (binding, this task's own required contract)

There are now potentially TWO independent relation-evidence sources for
the same predecessor edge: D-157's own Watch+Listen
``AttemptRelationHypothesis`` (multimodal corroboration) and this
module's own D-169 canonical ``RelationEvidence`` (linguistic structural
evidence). This module NEVER majority-votes them and NEVER silently
prefers one over the other -- ``fuse_relation_evidence`` (consumed by
``editorial_moment_sequence_integration.build_editorial_moments_for_
source``, D-199's own small, authorized extension there) implements the
explicit contract:

    - only D-157 resolved  -> D157_ONLY, D-157's own value;
    - only canonical resolved -> CANONICAL_ONLY, canonical's own value;
    - both resolved, AGREE -> AGREEMENT, the shared value;
    - both resolved, CONFLICT -> CONFLICT_ABSTAINED, forced to
      ``RELATION_UNCERTAIN`` (D-197's own grouper already treats
      UNCERTAIN as a boundary -- never joins on unresolved disagreement,
      per this task's own "preserve conflict... abstain from claiming
      certainty" instruction);
    - neither resolved -> MISSING, ``None``.

This is a genuinely NEW small piece of logic this task is authorized to
add (a value-selection/fusion policy over two ALREADY-COMPUTED evidence
sources for P1's own diagnostic consumption) -- it is NOT a change to
D-197's ``build_editorial_local_groups`` join/split RULES (RETRY/
CORRECTION/CONTINUATION join; NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION/
COMPLEMENTARY/UNCERTAIN/missing split), which remain byte-identical --
only WHICH relation value reaches that unchanged rule set can differ.

## Proposition/Retry identity firewall (binding, restated)

This module reads ``PropositionCandidate.proposition_candidate_id``
only as a stable reference key for matching ``RelationEvidence`` pairs.
It never merges propositions, never splits families, and never mints
or reads ``retry_family_id``/``take_group_id``/``semantic_idea_id`` --
confirmed by module-leaf string-absence tests, mirroring D-197's own
"no family/BestTake signal" firewall.

## Bilingual / code-switching safety (this task's own explicit
## requirement)

This module imports no language-detection logic and adds none: every
function it calls (D-166's ``normalize_language_text``/
``segment_language_phrases``, D-168's ``segment_language_utterances``/
``build_language_attempts``, D-169's ``build_claim_signature``/
``classify_relation_candidate``) is already Unicode-token/timing/
punctuation/structural-evidence-driven, with zero ``if language ==
"es"``/``"en"`` branches anywhere in this codebase (grep-verified). A
language switch mid-utterance changes no boundary decision by itself --
only real timing/structural evidence (word gaps, punctuation, restart
lexical evidence, audio pauses) can. No new phrase dictionary (e.g. a
"retry" or "correction" marker-word list) is added anywhere in this
module -- this task's own explicit "do not phrase-hardcode from the
fixture language" instruction.

## No authority (restated)

This module never touches Family Formation, BestTake, D-191, Ordering,
Boundary, Pacing, Renderer, or any commercial/sales-funnel authority. It
produces evidence only; P1 itself (unchanged, D-193/D-194) remains the
only consumer, and P1 has zero authority (D-193's own boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Mapping, Tuple

from .audio_silence import AUDIO_SILENCE_EVENT_KIND
from .contracts import Word
from .language_proposition_relation import (
    PropositionCandidate,
    RelationEvidence,
    build_proposition_candidates,
    build_relation_evidence,
)
from .language_spine import (
    BOUNDARY_PAUSE,
    BOUNDARY_RESTART_BOUNDARY,
    LanguagePhrase,
    LanguageWord,
    adapt_words_to_language_words,
    segment_language_phrases,
)
from .language_utterance_attempt import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguageAttempt,
    LanguageUtterance,
    build_language_attempts,
    segment_language_utterances,
)
from .raw_understanding_map import RawUnderstandingMap
from .watch_listen_understanding import UnderstandingSpan

SCHEMA_VERSION = "cutsell.language_spine_live_integration.v1"

_DIAGNOSTICS_ENV = "CUTSELL_LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def live_language_spine_diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Default OFF. Narrow diagnostic-only flag, independent of P1's own
    ``CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED`` (D-196) --
    this task's own directive requires the two never be auto-linked.
    When OFF, nothing in this module is ever called by ``pipeline.py``."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


# ---------------------------------------------------------------------------
# Capability-status vocabulary (same spelling as D-195's own).
# ---------------------------------------------------------------------------
CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"
CAPABILITY_DISABLED = "DISABLED"

# ---------------------------------------------------------------------------
# Relation-evidence-source vocabulary (this task's own required set).
# ---------------------------------------------------------------------------
RELATION_SOURCE_MISSING = "MISSING"
RELATION_SOURCE_D157_ONLY = "D157_ONLY"
RELATION_SOURCE_CANONICAL_ONLY = "CANONICAL_ONLY"
RELATION_SOURCE_AGREEMENT = "AGREEMENT"
RELATION_SOURCE_CONFLICT_ABSTAINED = "CONFLICT_ABSTAINED"

# ---------------------------------------------------------------------------
# Language-evidence-source vocabulary (this task's own required set, reused
# verbatim as the "language_evidence_source" moment diagnostic in
# editorial_moment_sequence_integration.py).
# ---------------------------------------------------------------------------
LANGUAGE_EVIDENCE_CANONICAL = "CANONICAL_LANGUAGE_SPINE"
LANGUAGE_EVIDENCE_D157_FALLBACK = "D157_FALLBACK"
LANGUAGE_EVIDENCE_MISSING = "MISSING"

# Relation value D-197's own grouper already treats as "insufficient
# evidence to join" -- reused verbatim (never redefined) as the abstention
# target on a genuine cross-source conflict.
_RELATION_UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class LiveLanguageSpineEvidence:
    """Per-source aggregate of the six canonical Language Spine rungs,
    built ONCE from ``RawUnderstandingMap.word_timings`` -- diagnostics-
    only container, never an authority object. References child objects
    by value (all six rungs are already-bounded, small, frozen types --
    never a giant upstream object copied in); no transcript blob beyond
    what the Language Spine's own typed objects already carry (their own
    ``text_raw``/``text_normalized`` fields, same as every other
    Language Spine diagnostic in this codebase)."""
    source_asset_id: str
    words: Tuple[LanguageWord, ...]
    phrases: Tuple[LanguagePhrase, ...]
    utterances: Tuple[LanguageUtterance, ...]
    attempts: Tuple[LanguageAttempt, ...]
    proposition_candidates: Tuple[PropositionCandidate, ...]
    relation_evidence: Tuple[RelationEvidence, ...]
    capability_status: str
    missing_evidence: Tuple[str, ...]
    conflicts: Tuple[str, ...]
    provenance: Tuple[str, ...]
    # D-236: additive, defaulted so every pre-existing direct construction
    # site (this module's own two return sites below, plus existing test
    # fixtures) stays valid without modification -- the real count of
    # audio-silence intervals actually consumed for this source (0 when
    # none were available or supplied, never a re-derivation of
    # `missing_evidence`'s own AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED flag).
    audio_silence_interval_count: int = 0


def _audio_silence_intervals_from_raw_understanding_map(
    raw_understanding_map: RawUnderstandingMap | None,
) -> Tuple[Tuple[float, float], ...]:
    """D-236: the ONE extraction point for real, already-computed audio-
    silence-interval evidence into the plain ``(start, end)`` tuple shape
    ``language_spine.segment_language_phrases`` already accepts. Never
    recomputes anything -- ``RawUnderstandingMap.audio_events`` (D-155)
    is itself already filtered to ``audio_silence_interval``-kind events
    at construction time (``raw_understanding_map.build_raw_
    understanding_map``), built from ``audio_silence.py``'s own real
    ffmpeg ``silencedetect`` pass and merged onto the whole-video context
    once per source in ``flow_b.py`` -- no ffmpeg call, no ASR call, no
    local-performance recomputation happens here or anywhere downstream
    of this function. Defensively re-filters by ``kind`` and by this
    map's OWN ``source_asset_id`` anyway (belt-and-braces, matching this
    module's own existing "never compared across two different
    source_asset_id values" posture elsewhere) so a Source A interval can
    never leak into Source B's phrase segmentation even if some future
    caller passed a differently-scoped event list. Returns ``()``
    (fail-open, byte-identical to pre-D-236 behavior) when the map is
    absent or carries no such evidence -- never raises."""
    if raw_understanding_map is None:
        return ()
    return tuple(
        (float(event.start), float(event.end))
        for event in raw_understanding_map.audio_events
        if event.kind == AUDIO_SILENCE_EVENT_KIND
        and event.source_asset_id == raw_understanding_map.source_asset_id
    )


def build_live_language_spine_for_source(
    *, source_asset_id: str, raw_understanding_map: RawUnderstandingMap | None,
) -> LiveLanguageSpineEvidence:
    """The one canonical per-source live Language Spine builder. Pure;
    no I/O, no provider/network call, no ASR re-invocation -- consumes
    ONLY ``raw_understanding_map.word_timings`` (already-computed
    ``contracts.Word`` tuple). Calls D-166/D-168/D-169's own builders
    verbatim, once each, in the canonical order. Never raises on empty/
    absent input -- returns an honest ``NOT_EVALUABLE`` result instead
    (this module's own half of the "no crash, no dropped source"
    fallback-failure contract; the other half, falling back to D-157,
    lives in ``editorial_moment_sequence_integration.py``)."""
    if raw_understanding_map is None or not raw_understanding_map.word_timings:
        return LiveLanguageSpineEvidence(
            source_asset_id=source_asset_id, words=(), phrases=(), utterances=(), attempts=(),
            proposition_candidates=(), relation_evidence=(), capability_status=CAPABILITY_NOT_EVALUABLE,
            missing_evidence=("RAW_UNDERSTANDING_MAP_WORD_TIMINGS_ABSENT",), conflicts=(), provenance=(),
        )

    words = adapt_words_to_language_words(source_asset_id, raw_understanding_map.word_timings)
    # D-236 (docs/CUTSELL_DECISIONS.md D-235Z/D-236): the real, already-
    # computed per-source audio-silence-interval evidence
    # (RawUnderstandingMap.audio_events, D-155, itself sourced from
    # audio_silence.py's own real ffmpeg silencedetect pass, merged in
    # flow_b.py -- never recomputed here) is now threaded into phrase
    # segmentation, closing D-235Z's own confirmed missing-evidence
    # finding. No restart-marker evidence exists ANYWHERE upstream in
    # this codebase in a source-wide, pre-computed, timestamp form
    # (D-236's own audit of clean_cut.py/post_selection_internal_
    # retake_trim.py/internal_repeat_trim.py found only per-clip lexical
    # or ad hoc local-variable restart heuristics, never a reusable
    # per-source marker-time list) -- reported honestly below via
    # RESTART_MARKER_EVIDENCE_NOT_AVAILABLE rather than invented, per
    # this task's own "do NOT invent a new detector" instruction.
    # segment_language_phrases's OWN documented fail-open contract
    # (D-166) still applies whenever no real interval is available for a
    # source (empty/no-audio/failed ffmpeg pass): timing + punctuation
    # only, byte-identical to pre-D-236 behavior in that case.
    audio_silence_intervals = _audio_silence_intervals_from_raw_understanding_map(raw_understanding_map)
    phrases = segment_language_phrases(words, audio_silence_intervals=audio_silence_intervals)
    utterances = segment_language_utterances(phrases)
    attempts = build_language_attempts(utterances)
    proposition_candidates = build_proposition_candidates(attempts)
    attempts_by_id = {a.attempt_id: a for a in attempts}
    relation_evidence = build_relation_evidence(proposition_candidates, attempts_by_id)

    missing_evidence: list[str] = ["RESTART_MARKER_EVIDENCE_NOT_AVAILABLE"]
    if not audio_silence_intervals:
        missing_evidence.append("AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED")
    if not attempts:
        capability_status = CAPABILITY_NOT_EVALUABLE
        missing_evidence.append("NO_LANGUAGE_ATTEMPT_CONSTRUCTED")
    else:
        capability_status = CAPABILITY_AVAILABLE

    conflicts = tuple(
        r.left_proposition_candidate_id + "|" + r.right_proposition_candidate_id
        for r in relation_evidence if r.proposition_conflict
    )

    return LiveLanguageSpineEvidence(
        source_asset_id=source_asset_id,
        words=words,
        phrases=phrases,
        utterances=utterances,
        attempts=attempts,
        proposition_candidates=proposition_candidates,
        relation_evidence=relation_evidence,
        capability_status=capability_status,
        missing_evidence=tuple(missing_evidence),
        conflicts=conflicts,
        provenance=("CANONICAL_LANGUAGE_SPINE",),
        audio_silence_interval_count=len(audio_silence_intervals),
    )


# ---------------------------------------------------------------------------
# Bridging: real LanguageAttempt -> P1's take/span-keyed architecture.
# ---------------------------------------------------------------------------
def _overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def language_attempts_by_span_id_for_source(
    understanding_spans: Iterable[UnderstandingSpan], attempts: Tuple[LanguageAttempt, ...],
) -> dict[str, LanguageAttempt]:
    """Deterministic MAXIMUM-OVERLAP bridge from real ``LanguageAttempt``
    objects (D-166/D-168's own independent segmentation) onto P1's
    existing ``UnderstandingSpan.span_id`` keys. A span with zero
    temporal overlap against every real attempt is simply absent from
    the returned mapping -- callers fall through to the existing D-157
    fallback for that span (see module docstring's FALLBACK CONTRACT).
    Never uses BestTake/family/rendered-timeline identity; never a text
    match -- timing overlap only."""
    result: dict[str, LanguageAttempt] = {}
    for span in understanding_spans:
        best: LanguageAttempt | None = None
        best_overlap = 0.0
        for attempt in attempts:
            overlap = _overlap_duration(span.source_start, span.source_end, attempt.source_start, attempt.source_end)
            if overlap <= 0.0:
                continue
            if best is None or overlap > best_overlap or (
                overlap == best_overlap and (attempt.source_start, attempt.attempt_id) < (best.source_start, best.attempt_id)
            ):
                best, best_overlap = attempt, overlap
        if best is not None:
            result[span.span_id] = best
    return result


def proposition_candidate_ids_by_attempt_id_for(
    proposition_candidates: Tuple[PropositionCandidate, ...],
) -> dict[str, Tuple[str, ...]]:
    """Trivial passthrough grouping -- D-169's own V1 builds exactly one
    ``PropositionCandidate`` per ``LanguageAttempt`` (module docstring's
    own "bounded and simple" scope), so this is a direct id-to-id(s) map,
    never a re-derivation."""
    result: dict[str, Tuple[str, ...]] = {}
    for prop in proposition_candidates:
        for attempt_id in prop.attempt_ids:
            result.setdefault(attempt_id, ())
            result[attempt_id] += (prop.proposition_candidate_id,)
    return result


def proposition_slot_evidence_by_id_for(
    proposition_candidates: Tuple[PropositionCandidate, ...],
) -> dict[str, str]:
    """D-235X Part A: trivial passthrough projection of each
    ``PropositionCandidate``'s own already-computed ``editorial_slot_
    evidence`` field (D-169) -- never a re-derivation, never a new slot
    classifier. The exact ``proposition_slot_evidence_by_id`` lookup
    ``complete_lost_semantic_atom_materiality.assess_complete_lost_
    semantic_atom_materiality`` already accepts as an optional parameter
    (D-235Q), now given a real live producer."""
    return {prop.proposition_candidate_id: prop.editorial_slot_evidence for prop in proposition_candidates}


def relation_evidence_by_proposition_pair(
    relation_evidence: Tuple[RelationEvidence, ...],
) -> dict[Tuple[str, str], RelationEvidence]:
    """Keys real ``RelationEvidence`` rows by their own
    ``(left_proposition_candidate_id, right_proposition_candidate_id)``
    pair -- the exact pair ``build_editorial_moments_for_source``'s own
    D-199 extension looks up per predecessor edge."""
    return {(r.left_proposition_candidate_id, r.right_proposition_candidate_id): r for r in relation_evidence}


# ---------------------------------------------------------------------------
# Relation consistency (binding contract -- see module docstring).
# ---------------------------------------------------------------------------
def fuse_relation_evidence(
    d157_relation: str | None, canonical_relation: str | None,
) -> Tuple[str | None, str]:
    """The one canonical relation-fusion function this task requires.
    Never majority-votes; never silently prefers one source. Returns
    ``(final_relation_or_none, relation_evidence_source)``. See module
    docstring's "RELATION CONSISTENCY" section for the exact contract."""
    if d157_relation is None and canonical_relation is None:
        return None, RELATION_SOURCE_MISSING
    if d157_relation is not None and canonical_relation is None:
        return d157_relation, RELATION_SOURCE_D157_ONLY
    if d157_relation is None and canonical_relation is not None:
        return canonical_relation, RELATION_SOURCE_CANONICAL_ONLY
    if d157_relation == canonical_relation:
        return d157_relation, RELATION_SOURCE_AGREEMENT
    return _RELATION_UNCERTAIN, RELATION_SOURCE_CONFLICT_ABSTAINED


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts-only -- same pattern as every other D-19x
# compact summary in this codebase). No transcript dump.
# ---------------------------------------------------------------------------
def live_language_spine_diagnostics(evidence: LiveLanguageSpineEvidence) -> dict:
    # D-236: compact, counts-only boundary-kind tally over the already-
    # built phrases -- no transcript, no re-segmentation, pure re-
    # projection of what segment_language_phrases already decided.
    pause_boundary_count = sum(1 for p in evidence.phrases if p.boundary_kind == BOUNDARY_PAUSE)
    restart_boundary_count = sum(1 for p in evidence.phrases if p.boundary_kind == BOUNDARY_RESTART_BOUNDARY)
    return {
        "source_asset_id": evidence.source_asset_id,
        "capability_status": evidence.capability_status,
        "missing_evidence": list(evidence.missing_evidence),
        "conflict_count": len(evidence.conflicts),
        "language_word_count": len(evidence.words),
        "language_phrase_count": len(evidence.phrases),
        "language_utterance_count": len(evidence.utterances),
        "language_attempt_count": len(evidence.attempts),
        "proposition_candidate_count": len(evidence.proposition_candidates),
        "relation_evidence_count": len(evidence.relation_evidence),
        "provenance": list(evidence.provenance),
        # D-236 additions -- see docs/CUTSELL_DECISIONS.md D-236.
        "language_spine_audio_silence_evidence_status": (
            "SUPPLIED" if "AUDIO_SILENCE_EVIDENCE_NOT_SUPPLIED" not in evidence.missing_evidence
            else "NOT_SUPPLIED"
        ),
        "audio_silence_interval_count": evidence.audio_silence_interval_count,
        "restart_marker_evidence_status": (
            "NOT_AVAILABLE" if "RESTART_MARKER_EVIDENCE_NOT_AVAILABLE" in evidence.missing_evidence
            else "AVAILABLE"
        ),
        "phrase_count": len(evidence.phrases),
        "utterance_count": len(evidence.utterances),
        "attempt_count": len(evidence.attempts),
        "pause_boundary_count": pause_boundary_count,
        "restart_boundary_count": restart_boundary_count,
    }


def live_language_spine_run_summary(evidences: Iterable[LiveLanguageSpineEvidence]) -> dict:
    evidences = tuple(evidences)
    statuses = {e.capability_status for e in evidences}
    if not evidences:
        status = CAPABILITY_NOT_EVALUABLE
    elif statuses == {CAPABILITY_AVAILABLE}:
        status = CAPABILITY_AVAILABLE
    elif statuses == {CAPABILITY_NOT_EVALUABLE}:
        status = CAPABILITY_NOT_EVALUABLE
    else:
        status = CAPABILITY_PARTIAL

    return {
        "schema_version": SCHEMA_VERSION,
        "live_language_spine_status": status,
        "source_construction_count": len(evidences),
        "language_word_count": sum(len(e.words) for e in evidences),
        "language_phrase_count": sum(len(e.phrases) for e in evidences),
        "language_utterance_count": sum(len(e.utterances) for e in evidences),
        "language_attempt_count": sum(len(e.attempts) for e in evidences),
        "proposition_candidate_count": sum(len(e.proposition_candidates) for e in evidences),
        "relation_evidence_count": sum(len(e.relation_evidence) for e in evidences),
        "language_conflict_count": sum(len(e.conflicts) for e in evidences),
        # D-236 addition -- see docs/CUTSELL_DECISIONS.md D-236.
        "audio_silence_interval_count": sum(e.audio_silence_interval_count for e in evidences),
    }
