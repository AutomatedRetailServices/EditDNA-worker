"""D-115 -- the ONE canonical position-aware performance evidence layer.

D-114 (`docs/CUTSELL_FORENSIC_POSITION_AWARE_PERCEPTION_D114.md`) proved real,
RAW-absolute-timestamped visual/motion events already exist
(`local_performance.py::detect_candidate_events`) and survive, unaggregated,
in `WholeVideoContext.sources[].events`; the loss is that `local_performance.
apply_local_performance_to_takes` (and, a second time, `attempt_
reconstruction._merge_signals`) collapses them into duration-weighted
`MediaSignals` scalars before `take_judge.score_take` ever sees them --
destroying position, never the events themselves.

This module is the ONE canonical temporal interpretation of a
take/attempt's ENTRY/DELIVERY/EXIT structure, so Boundary, BestTake and
Watch+Listen never each independently reinvent it (D-115's core
architectural rule). It is EVIDENCE ONLY:

- it derives the measured DELIVERY span from the take/attempt's own
  already-aligned word timestamps (`CandidateTake.words`) -- no fixed
  lead-in/tail seconds, no semantic lookahead, no provider call;
- it classifies already-real, already-timestamped performance events
  (D-114's four local-performance candidate kinds, plus `audio_silence_
  interval` -- the only other event already sharing the exact same
  `TemporalEvent` schema and already relied on elsewhere for boundary
  reasoning) relative to that span into ENTRY / DELIVERY / EXIT, recording
  straddling explicitly rather than forcing a single zone;
- it changes NOTHING about `CandidateTake`, `MediaSignals`, selection
  membership, DeliveryScorer/BestTake scoring, BoundaryEngine trimming, the
  render plan, or Watch+Listen verdicts. Every function here is a pure,
  read-only projection of data that already exists, called for
  DIAGNOSTICS ONLY (see `flow_b.py`'s wiring, which stores the result
  under `attempt_reconstruction_diagnostics["positioned_performance_
  evidence"]` and never feeds it back into `takes` or `whole_context`).

D-116 (`docs/CUTSELL_DECISIONS.md`) is the first real consumer: `boundary_
engine_pass.py`'s CASE A visual edge trimming calls `compute_delivery_span`
and `classify_event_zone` directly on its own selected `DraftClip`s rather
than recomputing ENTRY/DELIVERY/EXIT itself -- the shared-temporal-
authority rule this module exists to enforce. D-116 never trims a
DELIVERY-zone (or any straddling) event; that remains reserved for a
future, separately-authorized BestTake/DeliveryScorer consumer (CASE B).

Attempt-merge preservation (D-114's second loss point): this module is
deliberately called AFTER `attempt_reconstruction.reconstruct_delivery_
attempts` has already produced the final fused attempts. A fused attempt's
`words` are already the verbatim concatenation of every member's words
(`_merge_attempt`), and its `start`/`end` already span the full source-
absolute range of its members -- so computing delivery span and positioned
events directly off the FINAL attempt object automatically covers every
member's evidence with no averaging, no re-derivation, and no change to
`_merge_signals` itself. There is nothing to "carry through" the merge
because nothing here is computed before it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .audio_silence import AUDIO_SILENCE_EVENT_KIND
from .contracts import CandidateTake, Word
from .whole_video_analysis import TemporalEvent, WholeVideoContext

SCHEMA_VERSION = "cutsell.positioned_performance_evidence.v1"

ZONE_ENTRY = "ENTRY"
ZONE_DELIVERY = "DELIVERY"
ZONE_EXIT = "EXIT"
ZONE_UNKNOWN = "UNKNOWN"

DELIVERY_SPAN_SOURCE_WORD_ENVELOPE = "word_envelope"
DELIVERY_SPAN_SOURCE_UNAVAILABLE = "unavailable_no_words"

# D-114's four proven, real, RAW-timestamped local-performance candidate
# kinds -- the minimum event set this task authorizes.
LOCAL_PERFORMANCE_EVENT_KINDS: frozenset[str] = frozenset({
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
    "body_reset_candidate",
    "hand_motion_reset_candidate",
})

# `audio_silence_interval` shares the exact same `TemporalEvent` schema, is
# already real/timestamped (ffmpeg silencedetect, `audio_silence.py`), and
# is already consulted elsewhere for entry/exit/interior reasoning
# (`boundary_engine_pass.py`, `attempt_reconstruction._measured_pause_at_
# transition`) -- included per this task's own "any other already-real
# timestamped event that fits the same existing event schema" allowance,
# not a new signal class.
POSITIONED_EVENT_KINDS: frozenset[str] = LOCAL_PERFORMANCE_EVENT_KINDS | {AUDIO_SILENCE_EVENT_KIND}


@dataclass(frozen=True)
class DeliverySpan:
    """The measured spoken-delivery envelope for a take/attempt.

    `available=False` (start/end both None) whenever the candidate carries
    no aligned words -- this module never manufactures a delivery span."""
    start: float | None
    end: float | None
    available: bool
    source: str


@dataclass(frozen=True)
class PositionedEvent:
    """One already-real, already-timestamped performance event, classified
    relative to a `DeliverySpan`. `zone`/`overlaps_delivery`/`starts_before_
    delivery`/`ends_after_delivery` are all `None` when the delivery span
    itself is unavailable -- never a fabricated single-zone guess."""
    kind: str
    start: float
    end: float
    confidence: float
    zone: str
    overlaps_delivery: bool | None
    starts_before_delivery: bool | None
    ends_after_delivery: bool | None
    evidence_source: str


@dataclass(frozen=True)
class PositionAwarePerformanceEvidence:
    """The full positioned-evidence record for one take/attempt. Additive
    and optional -- nothing in the active pipeline reads this to make an
    editorial, scoring, or trimming decision (D-115 scope)."""
    candidate_id: str
    source_asset_id: str
    source_start: float
    source_end: float
    delivery_span: DeliverySpan
    positioned_events: Tuple[PositionedEvent, ...] = ()


def compute_delivery_span(words: Iterable[Word]) -> DeliverySpan:
    """DELIVERY start/end = first spoken word start / last spoken word end,
    using only the already-existing aligned word timestamps. No fixed
    lead-in/tail, no semantic lookahead, no threshold. Absent any word,
    the span is explicitly unavailable rather than guessed."""
    word_tuple = tuple(words)
    if not word_tuple:
        return DeliverySpan(start=None, end=None, available=False,
                             source=DELIVERY_SPAN_SOURCE_UNAVAILABLE)
    start = min(float(word.start) for word in word_tuple)
    end = max(float(word.end) for word in word_tuple)
    return DeliverySpan(start=start, end=end, available=True,
                         source=DELIVERY_SPAN_SOURCE_WORD_ENVELOPE)


def classify_event_zone(
    event_start: float, event_end: float, delivery_span: DeliverySpan,
) -> tuple[str, bool | None, bool | None, bool | None]:
    """The ONE canonical ENTRY/DELIVERY/EXIT zone classification (D-115/
    D-116's shared temporal authority -- no other module may recompute this
    itself). An event that overlaps the delivery span at all is classified
    DELIVERY (per this task's conceptual semantics: "DELIVERY = event
    overlaps DELIVERY") even when it also extends into ENTRY or EXIT
    territory -- `starts_before_delivery`/`ends_after_delivery` record that
    straddle explicitly rather than losing it to a single zone label."""
    if not delivery_span.available:
        return ZONE_UNKNOWN, None, None, None
    d_start = float(delivery_span.start)  # type: ignore[arg-type]
    d_end = float(delivery_span.end)  # type: ignore[arg-type]
    overlaps = event_start < d_end and event_end > d_start
    starts_before = event_start < d_start
    ends_after = event_end > d_end
    if overlaps:
        zone = ZONE_DELIVERY
    elif event_end <= d_start:
        zone = ZONE_ENTRY
    else:
        zone = ZONE_EXIT
    return zone, overlaps, starts_before, ends_after


def _events_for_source(
    context: WholeVideoContext | None, source_asset_id: str,
) -> Tuple[TemporalEvent, ...]:
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return source.events
    return ()


def build_positioned_performance_evidence(
    candidate: CandidateTake,
    context: WholeVideoContext | None,
    *,
    event_kinds: frozenset[str] = POSITIONED_EVENT_KINDS,
) -> PositionAwarePerformanceEvidence:
    """Build the one canonical positioned-evidence record for `candidate`.

    Event selection reuses the EXACT same source-window overlap test
    `local_performance.apply_local_performance_to_takes` already uses
    (`event.end > candidate.start and event.start < candidate.end`) --
    no new windowing rule, no new threshold. Restricted to `event_kinds`
    (default: the four D-114 local-performance kinds + `audio_silence_
    interval`) so default-only/aggregate-only `MediaSignals` fields never
    masquerade as positioned evidence."""
    delivery_span = compute_delivery_span(candidate.words)
    events = _events_for_source(context, candidate.source_asset_id)
    positioned: list[PositionedEvent] = []
    for event in events:
        kind = str(event.kind)
        if kind not in event_kinds:
            continue
        if not (event.end > candidate.start and event.start < candidate.end):
            continue
        zone, overlaps, before, after = classify_event_zone(event.start, event.end, delivery_span)
        positioned.append(PositionedEvent(
            kind=kind,
            start=float(event.start),
            end=float(event.end),
            confidence=float(event.confidence),
            zone=zone,
            overlaps_delivery=overlaps,
            starts_before_delivery=before,
            ends_after_delivery=after,
            evidence_source="local_performance" if kind in LOCAL_PERFORMANCE_EVENT_KINDS else "audio_silence",
        ))
    positioned.sort(key=lambda item: (item.start, item.end, item.kind))
    return PositionAwarePerformanceEvidence(
        candidate_id=candidate.clip_id,
        source_asset_id=candidate.source_asset_id,
        source_start=float(candidate.start),
        source_end=float(candidate.end),
        delivery_span=delivery_span,
        positioned_events=tuple(positioned),
    )


def build_positioned_performance_evidence_for_takes(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    event_kinds: frozenset[str] = POSITIONED_EVENT_KINDS,
) -> Tuple[PositionAwarePerformanceEvidence, ...]:
    """Batch form of `build_positioned_performance_evidence`, one record per
    take/attempt, in input order. Pure and read-only -- callers should
    invoke this once per run on the FINAL post-attempt-reconstruction
    candidate pool (see module docstring) and treat the result as
    diagnostics only."""
    return tuple(
        build_positioned_performance_evidence(take, context, event_kinds=event_kinds)
        for take in takes
    )


def positioned_performance_evidence_diagnostics(
    evidence_items: Iterable[PositionAwarePerformanceEvidence],
    *,
    limit: int = 300,
) -> list[dict]:
    """JSON-safe diagnostic rows, one per take/attempt, capped at `limit`
    (mirrors the `[:300]` cap `attempt_reconstruction.py`'s own diagnostics
    already use). Included even when a candidate has no words or no
    matching events, so an UNKNOWN delivery span or an empty event list is
    directly observable rather than silently omitted."""
    rows: list[dict] = []
    for item in tuple(evidence_items)[:limit]:
        rows.append({
            "candidate_id": item.candidate_id,
            "source_asset_id": item.source_asset_id,
            "source_start": round(item.source_start, 3),
            "source_end": round(item.source_end, 3),
            "delivery_span": {
                "available": item.delivery_span.available,
                "start": round(item.delivery_span.start, 3) if item.delivery_span.start is not None else None,
                "end": round(item.delivery_span.end, 3) if item.delivery_span.end is not None else None,
                "source": item.delivery_span.source,
            },
            "positioned_event_count": len(item.positioned_events),
            "positioned_events": [
                {
                    "kind": event.kind,
                    "start": round(event.start, 3),
                    "end": round(event.end, 3),
                    "confidence": round(event.confidence, 4),
                    "zone": event.zone,
                    "overlaps_delivery": event.overlaps_delivery,
                    "starts_before_delivery": event.starts_before_delivery,
                    "ends_after_delivery": event.ends_after_delivery,
                    "evidence_source": event.evidence_source,
                }
                for event in item.positioned_events
            ],
        })
    return rows
