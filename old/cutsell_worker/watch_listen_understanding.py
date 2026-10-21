"""D-157 Phase B -- Watch+Listen Multimodal Understanding V1.

Per docs/CUTSELL_DECISIONS.md D-148/D-154/D-155/D-156/D-157. This module
is the FIRST real Watch+Listen UNDERSTANDING layer: it consumes ONE
`raw_understanding_map.RawUnderstandingMap` (D-155, unchanged) and forms
bounded, categorical HYPOTHESES about behavior state, attempt boundaries,
attempt relations, meaning completion, and performance usability -- never
a final editorial decision.

## The core principle this module enforces structurally

    PERCEPTION PROPOSES EVIDENCE.       (Tracks A-D, `raw_understanding_map.py`)
    UNDERSTANDING FORMS HYPOTHESES.     (this module)
    STRUCTURED EDITORIAL AUTHORITIES DECIDE.  (D-145 Family Formation,
                                          D-150 semantic authority, D-123/
                                          D-128 BestTake, Boundary, Pacing)

This module NEVER decides final Proposition Identity, final Retry
Identity, final family topology, a BestTake winner, a Boundary trim, or a
Pacing transition. It is imported by nothing in those authorities as of
this task (structurally asserted by the module-leaf tests in
`tests/test_cutsell_d157_watch_listen_understanding.py`), and it imports
nothing from them either.

## What this module recomputes vs. reuses

Recomputes NOTHING about ASR, audio, or visual content. Every atomic
signal this module's hypotheses are built from is either:
- copied verbatim from the already-computed `RawUnderstandingMap`/
  `RawUnderstandingSpan` (D-155, unchanged) -- transcript, word timings,
  positioned ENTRY/DELIVERY/EXIT events, behavior hypotheses, provenance;
- read directly off the original `CandidateTake` objects the map was
  itself built from (`complete_idea` -- a real field D-046/attempt_
  reconstruction.py already computes upstream, never re-derived here);
- or REUSED, not reimplemented, from `attempt_reconstruction.py`'s own
  already-vetted pure helpers (`_restart_evidence`, `_measured_pause_at_
  transition`, `_TERMINAL_PUNCT_RE`) -- the exact same precedent
  `case_b_performance_evidence.py` already set for importing a sibling
  module's private helpers directly rather than writing a fourth,
  drifting copy of the same logic (D-154's own duplicate-compute
  inventory names this exact class of risk).

The one genuinely NEW thing this module does is COMBINE those existing,
real signals into bounded categorical hypotheses -- it invents no new
detector, no new numeric threshold, and no new provider call.

## Confidence vocabulary (no invented scores)

Per this task's own instruction ("no new 0.73/0.84 thresholds"), every
hypothesis carries a confidence from the fixed set `SUPPORTED`/`MIXED`/
`WEAK`/`UNKNOWN` -- never a numeric probability. `MIXED` is reserved for
genuine multimodal disagreement (`conflict_flags` non-empty); `SUPPORTED`
requires at least one DIRECTLY OBSERVED (non-absence-derived) real event;
`WEAK` covers absence-derived or single-signal inferences; `UNKNOWN`
covers no evidence at all.

## Attempt-relation vocabulary (hypotheses, never authority)

`RETRY`/`CORRECTION`/`CONTINUATION`/`COMPLEMENTARY`/`NEW_AUDIENCE_BEAT`/
`DISTINCT_PROPOSITION`/`UNCERTAIN` mirror this task's own directive
vocabulary and D-111/D-145's existing Attempt Relationship naming, but
these are HYPOTHESES about a span pair, never the final Attempt
Relationship D-145's own 5-way vocabulary decides. `DISTINCT_PROPOSITION`
is deliberately NEVER emitted above `UNCERTAIN` in V1 -- asserting it
honestly requires semantic/topical judgment this module has no provider
evidence for (D-111's own "same topic/opener is NOT enough" invariant,
restated here as a hard ceiling on this module's own confidence, not
merely on RETRY's).

## Duplicate-consumer consolidation (deferred, not performed this task)

D-154 named three independent readers of `local_performance.py`'s dense
events with their own private kind-logic: `attempt_reconstruction.py`
(`_RESET_KINDS`/`_CAMERA_KINDS`/`_FACE_KINDS`), `take_judge.py`'s
`delivery_cleanliness_evidence`, and `case_b_performance_evidence.py`
(which already migrated onto `take_judge.py`'s own private helpers to
avoid a fourth copy). This task's own directive explicitly allows, but
does not require, migrating those onto this module or the map
("Do NOT force every downstream consumer to migrate in this task").
**Deliberately NOT done here**: all three of those readers sit directly
on the live Selection/BestTake path (`attempt_reconstruction.py` decides
real attempt-merge boundaries; `take_judge.py`'s cleanliness evidence
feeds real DeliveryScorer signals) -- refactoring them carries real
regression risk to CLOSED authorities (D-123/D-128/the D-145-D-153
thread) for a task whose own strict scope forbids any BestTake/
DeliveryScorer/family change. Migrating them is left as a well-scoped,
separately-authorized future step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple

from .attempt_reconstruction import (
    _TERMINAL_PUNCT_RE,
    _measured_pause_at_transition,
    _restart_evidence,
)
from .contracts import CandidateTake
from .positioned_performance_evidence import ZONE_DELIVERY, ZONE_ENTRY, ZONE_EXIT
from .providers import ProviderStatus
from .raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_AUDIENCE_DELIVERY,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    PROVENANCE_DETERMINISTIC_RULE,
    PROVENANCE_MULTIMODAL_FUSION,
    PROVENANCE_VISUAL_SIGNAL,
    RawUnderstandingMap,
    RawUnderstandingSpan,
    _ABANDONED_KINDS,
    _BREAKING_CHARACTER_KINDS,
    _FALSE_START_KINDS,
    _RECORDING_PROCESS_KINDS,
    _RESET_FAMILY_KINDS,
)
from .whole_video_analysis import SourceVideoContext, WholeVideoContext

SCHEMA_VERSION = "cutsell.watch_listen_understanding.v1"

# ---------------------------------------------------------------------------
# Confidence vocabulary -- categorical only, never a numeric score.
# ---------------------------------------------------------------------------
CONFIDENCE_SUPPORTED = "SUPPORTED"
CONFIDENCE_MIXED = "MIXED"
CONFIDENCE_WEAK = "WEAK"
CONFIDENCE_UNKNOWN = "UNKNOWN"
ALLOWED_CONFIDENCE_LEVELS: frozenset[str] = frozenset({
    CONFIDENCE_SUPPORTED, CONFIDENCE_MIXED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Meaning-completion vocabulary.
# ---------------------------------------------------------------------------
MEANING_COMPLETE = "COMPLETE"
MEANING_INCOMPLETE = "INCOMPLETE"
MEANING_UNCERTAIN = "UNCERTAIN"
ALLOWED_MEANING_STATES: frozenset[str] = frozenset({
    MEANING_COMPLETE, MEANING_INCOMPLETE, MEANING_UNCERTAIN,
})

# ---------------------------------------------------------------------------
# Performance-usability vocabulary.
# ---------------------------------------------------------------------------
USABILITY_USABLE = "USABLE"
USABILITY_QUESTIONABLE = "QUESTIONABLE"
USABILITY_UNUSABLE = "UNUSABLE"
USABILITY_UNKNOWN = "UNKNOWN"
ALLOWED_USABILITY_STATES: frozenset[str] = frozenset({
    USABILITY_USABLE, USABILITY_QUESTIONABLE, USABILITY_UNUSABLE, USABILITY_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Attempt-relation vocabulary -- HYPOTHESES only (D-145's own 5-way vocabulary
# decides the final Attempt Relationship; this is never it).
# ---------------------------------------------------------------------------
RELATION_RETRY = "RETRY"
RELATION_CORRECTION = "CORRECTION"
RELATION_CONTINUATION = "CONTINUATION"
RELATION_COMPLEMENTARY = "COMPLEMENTARY"
RELATION_NEW_AUDIENCE_BEAT = "NEW_AUDIENCE_BEAT"
RELATION_DISTINCT_PROPOSITION = "DISTINCT_PROPOSITION"
RELATION_UNCERTAIN = "UNCERTAIN"
ALLOWED_ATTEMPT_RELATIONS: frozenset[str] = frozenset({
    RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION,
    RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT,
    RELATION_DISTINCT_PROPOSITION, RELATION_UNCERTAIN,
})

# ---------------------------------------------------------------------------
# Attempt-boundary vocabulary.
# ---------------------------------------------------------------------------
BOUNDARY_ATTEMPT_BEGINS = "ATTEMPT_BEGINS"
BOUNDARY_ATTEMPT_COMPLETES = "ATTEMPT_COMPLETES"
BOUNDARY_ATTEMPT_ABANDONED = "ATTEMPT_ABANDONED"
BOUNDARY_POST_TAKE_RESET = "POST_TAKE_RESET"
BOUNDARY_NEW_DELIVERY_BEGINS = "NEW_DELIVERY_BEGINS"
ALLOWED_BOUNDARY_KINDS: frozenset[str] = frozenset({
    BOUNDARY_ATTEMPT_BEGINS, BOUNDARY_ATTEMPT_COMPLETES, BOUNDARY_ATTEMPT_ABANDONED,
    BOUNDARY_POST_TAKE_RESET, BOUNDARY_NEW_DELIVERY_BEGINS,
})

# Real behavior-defect kinds (imported from raw_understanding_map.py's own
# already-verified kind-sets -- never a new vocabulary) used to derive
# per-zone performance usability. Wrong_take/retry_setup are D-100's
# deterministic-rule confirmations; the rest are real Track C event kinds.
_DEFECT_KINDS: frozenset[str] = (
    _RESET_FAMILY_KINDS | _ABANDONED_KINDS | _RECORDING_PROCESS_KINDS
    | _FALSE_START_KINDS | _BREAKING_CHARACTER_KINDS
)

# Mirrors attempt_reconstruction.reconstruct_delivery_attempts's own
# already-vetted default (1.20s) -- the SAME production constant, not a
# new number invented for this module.
_DEFAULT_MAX_CONTINUATION_GAP_SEC = 1.20


@dataclass(frozen=True)
class AttemptBoundaryHypothesis:
    """Evidence for a specific boundary event at one edge ('start'/'end')
    of a span -- never a physical cut. See module docstring."""
    boundary_kind: str
    edge: str
    confidence: str
    basis: str
    provenance: str


@dataclass(frozen=True)
class AttemptRelationHypothesis:
    """One hypothesis about this span's relation to its immediate
    predecessor in the same source (`left_span_id=None` when this span is
    the source's first -- no predecessor to relate to)."""
    relation: str
    confidence: str
    basis: str
    left_span_id: str | None
    provenance: Tuple[str, ...] = ()


@dataclass(frozen=True)
class UnderstandingSpan:
    """One bounded span's fused Understanding V1 hypotheses -- evidence,
    never an authority verdict. Mirrors `RawUnderstandingSpan`'s own
    identity fields exactly (same `span_id`/timing, source-relative)."""
    span_id: str
    source_asset_id: str
    source_start: float
    source_end: float
    behavior_state_hypotheses: Tuple[object, ...]  # BehaviorHypothesis, reused verbatim from D-155
    behavior_confidence: str
    attempt_boundary_hypotheses: Tuple[AttemptBoundaryHypothesis, ...]
    attempt_relation_hypotheses: Tuple[AttemptRelationHypothesis, ...]
    relation_confidence: str
    meaning_completion_hypothesis: str
    performance_usability_hypothesis: str
    entry_usability: str
    delivery_usability: str
    exit_usability: str
    conflict_flags: Tuple[str, ...]
    evidence_provenance: Mapping[str, str]


@dataclass(frozen=True)
class WatchListenUnderstanding:
    """The per-source Watch+Listen Understanding V1 container -- read by
    nothing yet (V1 is a hypothesis foundation only; see module
    docstring)."""
    source_asset_id: str
    understanding_spans: Tuple[UnderstandingSpan, ...] = ()
    track_status: Mapping[str, str] = field(default_factory=dict)


def _context_for_pause_lookup(raw_map: RawUnderstandingMap) -> WholeVideoContext:
    """Wraps the map's OWN already-filtered `audio_events` in the shape
    `attempt_reconstruction._measured_pause_at_transition` expects, so
    that function's exact existing tolerance/confidence logic can be
    reused verbatim -- no re-derivation of silence-evidence rules here."""
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id=raw_map.source_asset_id, summary="", dominant_style="",
            creator_intent="", events=raw_map.audio_events,
        ),),
        status=ProviderStatus("watch_listen_understanding_internal", True, True, "reused_from_map"),
    )


def _has_label(span: RawUnderstandingSpan, label: str) -> bool:
    return any(h.label == label for h in span.behavior_hypotheses)


def _zone_events(span: RawUnderstandingSpan, zone: str) -> tuple:
    return tuple(e for e in span.positioned_evidence.positioned_events if e.zone == zone)


def _has_exit_reset(span: RawUnderstandingSpan) -> bool:
    return any(e.kind in _RESET_FAMILY_KINDS for e in _zone_events(span, ZONE_EXIT))


def _usability_for_zone(span: RawUnderstandingSpan, zone: str) -> str:
    events = _zone_events(span, zone)
    if zone == ZONE_DELIVERY and not span.positioned_evidence.delivery_span.available:
        return USABILITY_UNKNOWN
    if any(e.kind in _DEFECT_KINDS for e in events):
        return USABILITY_UNUSABLE if zone == ZONE_DELIVERY else USABILITY_QUESTIONABLE
    return USABILITY_USABLE


def _overall_usability(entry: str, delivery: str, exit_: str) -> str:
    if delivery == USABILITY_UNUSABLE:
        return USABILITY_UNUSABLE
    if delivery == USABILITY_UNKNOWN:
        return USABILITY_UNKNOWN
    if entry == USABILITY_QUESTIONABLE or exit_ == USABILITY_QUESTIONABLE:
        return USABILITY_QUESTIONABLE
    return USABILITY_USABLE


def _meaning_completion(candidate: CandidateTake) -> str:
    if not str(candidate.text or "").strip():
        return MEANING_UNCERTAIN
    return MEANING_COMPLETE if candidate.complete_idea else MEANING_INCOMPLETE


def _behavior_confidence(span: RawUnderstandingSpan, conflicts: Tuple[str, ...]) -> str:
    if conflicts:
        return CONFIDENCE_MIXED
    provenances = {h.provenance for h in span.behavior_hypotheses}
    if provenances & {PROVENANCE_VISUAL_SIGNAL, PROVENANCE_DETERMINISTIC_RULE}:
        return CONFIDENCE_SUPPORTED
    if provenances & {PROVENANCE_MULTIMODAL_FUSION}:
        return CONFIDENCE_WEAK
    return CONFIDENCE_UNKNOWN


def _conflict_flags(span: RawUnderstandingSpan, meaning: str) -> Tuple[str, ...]:
    flags: list[str] = []
    if meaning == MEANING_COMPLETE and _has_label(span, BEHAVIOR_ABANDONED_ATTEMPT):
        flags.append("MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE")
    if meaning == MEANING_COMPLETE and _has_label(span, BEHAVIOR_POST_TAKE_RESET) and _has_exit_reset(span):
        flags.append("EXIT_RESET_VS_MEANING_COMPLETE")
    return tuple(flags)


def _relation_for_pair(
    left_take: CandidateTake | None,
    right_take: CandidateTake,
    left_span: RawUnderstandingSpan | None,
    right_span: RawUnderstandingSpan,
    context: WholeVideoContext,
) -> Tuple[AttemptRelationHypothesis, ...]:
    if left_take is None or left_span is None:
        return (AttemptRelationHypothesis(
            relation=RELATION_UNCERTAIN, confidence=CONFIDENCE_UNKNOWN,
            basis="first span in source; no predecessor to relate to",
            left_span_id=None, provenance=(PROVENANCE_MULTIMODAL_FUSION,),
        ),)

    restart = _restart_evidence(left_take.text, right_take.text)
    pause = _measured_pause_at_transition(context, left_take, right_take)
    gap = max(0.0, right_take.start - left_take.end)
    p_abandoned = _has_label(left_span, BEHAVIOR_ABANDONED_ATTEMPT) or _has_label(left_span, BEHAVIOR_FALSE_START)
    p_reset_after = _has_label(left_span, BEHAVIOR_POST_TAKE_RESET) and _has_exit_reset(left_span)
    p_broken = p_abandoned or p_reset_after or pause > 0.0
    p_complete = bool(left_take.complete_idea)
    p_terminal = bool(_TERMINAL_PUNCT_RE.search(str(left_take.text or "").rstrip()))
    s_fresh = _has_label(right_span, BEHAVIOR_CLEAN_ATTEMPT) or _has_label(right_span, BEHAVIOR_AUDIENCE_DELIVERY)

    entries: list[AttemptRelationHypothesis] = []
    prov = (PROVENANCE_DETERMINISTIC_RULE,)

    if restart and p_broken and s_fresh:
        entries.append(AttemptRelationHypothesis(
            RELATION_RETRY, CONFIDENCE_SUPPORTED,
            "lexical restart + prior-attempt abandonment/reset/pause evidence + fresh delivery start",
            left_span.span_id, prov,
        ))
    elif restart and p_broken:
        entries.append(AttemptRelationHypothesis(
            RELATION_RETRY, CONFIDENCE_WEAK,
            "lexical restart + prior-attempt abandonment/reset/pause evidence",
            left_span.span_id, prov,
        ))
    elif restart:
        entries.append(AttemptRelationHypothesis(
            RELATION_RETRY, CONFIDENCE_WEAK, "lexical restart evidence alone",
            left_span.span_id, prov,
        ))

    if restart and p_complete and not p_broken:
        entries.append(AttemptRelationHypothesis(
            RELATION_CORRECTION, CONFIDENCE_SUPPORTED,
            "lexical restart immediately after an apparently complete prior statement, no abandonment/reset evidence",
            left_span.span_id, prov,
        ))

    if not restart and (not p_complete or not p_terminal) and gap <= _DEFAULT_MAX_CONTINUATION_GAP_SEC and pause == 0.0:
        entries.append(AttemptRelationHypothesis(
            RELATION_CONTINUATION, CONFIDENCE_SUPPORTED,
            "no restart; prior span incomplete/non-terminal; tight gap; no measured pause",
            left_span.span_id, prov,
        ))
    elif not restart and not p_complete:
        entries.append(AttemptRelationHypothesis(
            RELATION_CONTINUATION, CONFIDENCE_WEAK,
            "no restart; prior span incomplete, but gap/pause evidence is ambiguous",
            left_span.span_id, prov,
        ))

    if not restart and p_complete and gap > _DEFAULT_MAX_CONTINUATION_GAP_SEC:
        entries.append(AttemptRelationHypothesis(
            RELATION_COMPLEMENTARY, CONFIDENCE_WEAK,
            "both spans appear complete, no restart, moderate gap -- no semantic corroboration available to confirm non-duplicative content",
            left_span.span_id, (PROVENANCE_MULTIMODAL_FUSION,),
        ))

    if not restart and p_complete and p_terminal and s_fresh and gap > _DEFAULT_MAX_CONTINUATION_GAP_SEC:
        entries.append(AttemptRelationHypothesis(
            RELATION_NEW_AUDIENCE_BEAT, CONFIDENCE_SUPPORTED,
            "prior span cleanly completed; no restart; fresh delivery start after a real gap",
            left_span.span_id, prov,
        ))
    elif not restart and p_complete and p_terminal:
        entries.append(AttemptRelationHypothesis(
            RELATION_NEW_AUDIENCE_BEAT, CONFIDENCE_WEAK,
            "prior span cleanly completed; no restart; fresh-start evidence inconclusive",
            left_span.span_id, prov,
        ))

    if not entries:
        entries.append(AttemptRelationHypothesis(
            RELATION_UNCERTAIN, CONFIDENCE_UNKNOWN,
            "insufficient evidence to support any bounded relation hypothesis",
            left_span.span_id, (PROVENANCE_MULTIMODAL_FUSION,),
        ))
    return tuple(entries)


_RELATION_RANK = {CONFIDENCE_SUPPORTED: 3, CONFIDENCE_MIXED: 2, CONFIDENCE_WEAK: 1, CONFIDENCE_UNKNOWN: 0}


def _best_relation_confidence(relations: Tuple[AttemptRelationHypothesis, ...]) -> str:
    if not relations:
        return CONFIDENCE_UNKNOWN
    return max((r.confidence for r in relations), key=lambda c: _RELATION_RANK.get(c, 0))


def _boundary_hypotheses(
    left_take: CandidateTake | None,
    right_take: CandidateTake,
    left_span: RawUnderstandingSpan | None,
    right_span: RawUnderstandingSpan,
    meaning: str,
    relations: Tuple[AttemptRelationHypothesis, ...],
) -> Tuple[AttemptBoundaryHypothesis, ...]:
    out: list[AttemptBoundaryHypothesis] = []
    s_fresh = _has_label(right_span, BEHAVIOR_CLEAN_ATTEMPT) or _has_label(right_span, BEHAVIOR_AUDIENCE_DELIVERY)
    p_abandoned = left_span is not None and (
        _has_label(left_span, BEHAVIOR_ABANDONED_ATTEMPT) or _has_label(left_span, BEHAVIOR_FALSE_START)
    )
    p_reset_after = left_span is not None and _has_label(left_span, BEHAVIOR_POST_TAKE_RESET) and _has_exit_reset(left_span)

    if s_fresh:
        if left_take is None or p_abandoned or p_reset_after:
            out.append(AttemptBoundaryHypothesis(
                BOUNDARY_ATTEMPT_BEGINS, "start", CONFIDENCE_SUPPORTED,
                "fresh delivery start, first span or following a broken/reset prior attempt",
                PROVENANCE_MULTIMODAL_FUSION,
            ))
        else:
            out.append(AttemptBoundaryHypothesis(
                BOUNDARY_ATTEMPT_BEGINS, "start", CONFIDENCE_WEAK,
                "fresh delivery start evidence alone", PROVENANCE_MULTIMODAL_FUSION,
            ))

    if meaning == MEANING_COMPLETE:
        out.append(AttemptBoundaryHypothesis(
            BOUNDARY_ATTEMPT_COMPLETES, "end", CONFIDENCE_SUPPORTED,
            "complete_idea true and terminal punctuation", PROVENANCE_DETERMINISTIC_RULE,
        ))

    if _has_label(right_span, BEHAVIOR_ABANDONED_ATTEMPT) or _has_label(right_span, BEHAVIOR_FALSE_START):
        out.append(AttemptBoundaryHypothesis(
            BOUNDARY_ATTEMPT_ABANDONED, "end", CONFIDENCE_SUPPORTED,
            "confirmed wrong_take/retry_setup or false_start evidence", PROVENANCE_DETERMINISTIC_RULE,
        ))

    if _has_label(right_span, BEHAVIOR_POST_TAKE_RESET) and _has_exit_reset(right_span):
        out.append(AttemptBoundaryHypothesis(
            BOUNDARY_POST_TAKE_RESET, "end", CONFIDENCE_SUPPORTED,
            "reset-family event measured in this span's own EXIT zone", PROVENANCE_VISUAL_SIGNAL,
        ))

    if any(r.relation == RELATION_RETRY and r.confidence == CONFIDENCE_SUPPORTED for r in relations):
        out.append(AttemptBoundaryHypothesis(
            BOUNDARY_NEW_DELIVERY_BEGINS, "start", CONFIDENCE_SUPPORTED,
            "supported RETRY relation to predecessor", PROVENANCE_MULTIMODAL_FUSION,
        ))

    return tuple(out)


def build_understanding_span(
    left_take: CandidateTake | None,
    right_take: CandidateTake,
    left_span: RawUnderstandingSpan | None,
    right_span: RawUnderstandingSpan,
    context: WholeVideoContext,
) -> UnderstandingSpan:
    """Pure projection: forms one span's bounded hypotheses from ALREADY-
    computed evidence (the map's own span, the original candidate's
    `complete_idea`, and its immediate predecessor for pairwise relation/
    boundary evidence). Recomputes no perception."""
    meaning = _meaning_completion(right_take)
    entry_use = _usability_for_zone(right_span, ZONE_ENTRY)
    delivery_use = _usability_for_zone(right_span, ZONE_DELIVERY)
    exit_use = _usability_for_zone(right_span, ZONE_EXIT)
    conflicts = _conflict_flags(right_span, meaning)
    relations = _relation_for_pair(left_take, right_take, left_span, right_span, context)
    boundaries = _boundary_hypotheses(left_take, right_take, left_span, right_span, meaning, relations)

    evidence_provenance = {
        "behavior_state_hypotheses": PROVENANCE_MULTIMODAL_FUSION,
        "meaning_completion_hypothesis": PROVENANCE_DETERMINISTIC_RULE,
        "performance_usability_hypothesis": PROVENANCE_DETERMINISTIC_RULE,
        "attempt_relation_hypotheses": PROVENANCE_MULTIMODAL_FUSION,
        "attempt_boundary_hypotheses": PROVENANCE_MULTIMODAL_FUSION,
    }

    return UnderstandingSpan(
        span_id=right_span.span_id,
        source_asset_id=right_span.source_asset_id,
        source_start=right_span.source_start,
        source_end=right_span.source_end,
        behavior_state_hypotheses=right_span.behavior_hypotheses,
        behavior_confidence=_behavior_confidence(right_span, conflicts),
        attempt_boundary_hypotheses=boundaries,
        attempt_relation_hypotheses=relations,
        relation_confidence=_best_relation_confidence(relations),
        meaning_completion_hypothesis=meaning,
        performance_usability_hypothesis=_overall_usability(entry_use, delivery_use, exit_use),
        entry_usability=entry_use,
        delivery_usability=delivery_use,
        exit_usability=exit_use,
        conflict_flags=conflicts,
        evidence_provenance=evidence_provenance,
    )


def build_watch_listen_understanding(
    raw_map: RawUnderstandingMap,
    takes_for_source: Iterable[CandidateTake],
) -> WatchListenUnderstanding:
    """Pure builder: one `WatchListenUnderstanding` per source, from an
    ALREADY-built `RawUnderstandingMap` (D-155, unchanged) plus the
    original `CandidateTake`s it was built from (needed only for
    `complete_idea`/pairwise pause-and-restart evidence -- text/timing are
    already duplicated onto the map's own spans and are read from there).
    Deterministic: spans are always processed in `(source_start,
    source_end, span_id)` order regardless of input order."""
    takes_by_id = {t.clip_id: t for t in takes_for_source}
    spans = tuple(sorted(
        raw_map.candidate_span_evidence,
        key=lambda s: (s.source_start, s.source_end, s.span_id),
    ))
    context = _context_for_pause_lookup(raw_map)

    understanding_spans = []
    left_take: CandidateTake | None = None
    left_span: RawUnderstandingSpan | None = None
    for span in spans:
        right_take = takes_by_id.get(span.span_id)
        if right_take is None:
            # Defensive: a span with no matching original candidate carries
            # no `complete_idea` truth -- skip it rather than guess. This
            # should not happen in the live pipeline (raw_understanding_map
            # builds spans FROM these same candidates), so this branch is
            # untested-by-construction but never silently wrong.
            continue
        understanding_spans.append(build_understanding_span(left_take, right_take, left_span, span, context))
        left_take, left_span = right_take, span

    return WatchListenUnderstanding(
        source_asset_id=raw_map.source_asset_id,
        understanding_spans=tuple(understanding_spans),
        track_status=raw_map.track_status,
    )


def build_watch_listen_understanding_for_sources(
    raw_maps: Iterable[RawUnderstandingMap],
    takes: Iterable[CandidateTake],
) -> Tuple[WatchListenUnderstanding, ...]:
    """Batch form, one `WatchListenUnderstanding` per `RawUnderstandingMap`,
    in `raw_maps` order -- deterministic regardless of `takes`' own order."""
    takes = tuple(takes)
    out = []
    for raw_map in raw_maps:
        source_takes = tuple(t for t in takes if t.source_asset_id == raw_map.source_asset_id)
        out.append(build_watch_listen_understanding(raw_map, source_takes))
    return tuple(out)


def watch_listen_understanding_diagnostics(
    understandings: Iterable[WatchListenUnderstanding],
) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-119/D-125/
    D-152/D-155's own compact summaries) -- never dumps a transcript,
    word timing, or per-hypothesis basis string."""
    understandings = tuple(understandings)
    spans = tuple(span for u in understandings for span in u.understanding_spans)

    behavior_count = sum(len(s.behavior_state_hypotheses) for s in spans)
    boundary_count = sum(len(s.attempt_boundary_hypotheses) for s in spans)
    relation_count = sum(len(s.attempt_relation_hypotheses) for s in spans)
    uncertain_relation_count = sum(
        1 for s in spans for r in s.attempt_relation_hypotheses if r.relation == RELATION_UNCERTAIN
    )
    conflict_count = sum(len(s.conflict_flags) for s in spans)

    provenance_counts: dict[str, int] = {}
    for s in spans:
        for h in s.behavior_state_hypotheses:
            provenance_counts[h.provenance] = provenance_counts.get(h.provenance, 0) + 1

    return {
        "watch_listen_understanding_created": bool(understandings),
        "understanding_span_count": len(spans),
        "behavior_hypothesis_count": behavior_count,
        "attempt_boundary_hypothesis_count": boundary_count,
        "attempt_relation_hypothesis_count": relation_count,
        "uncertain_relation_count": uncertain_relation_count,
        "meaning_complete_count": sum(1 for s in spans if s.meaning_completion_hypothesis == MEANING_COMPLETE),
        "meaning_incomplete_count": sum(1 for s in spans if s.meaning_completion_hypothesis == MEANING_INCOMPLETE),
        "meaning_uncertain_count": sum(1 for s in spans if s.meaning_completion_hypothesis == MEANING_UNCERTAIN),
        "performance_usable_count": sum(1 for s in spans if s.performance_usability_hypothesis == USABILITY_USABLE),
        "performance_questionable_count": sum(1 for s in spans if s.performance_usability_hypothesis == USABILITY_QUESTIONABLE),
        "performance_unusable_count": sum(1 for s in spans if s.performance_usability_hypothesis == USABILITY_UNUSABLE),
        "conflict_count": conflict_count,
        "provenance_counts": provenance_counts,
        "source_count": len(understandings),
    }
