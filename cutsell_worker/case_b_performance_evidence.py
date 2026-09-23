"""D-122 -- BestTake CASE B performance-evidence projection.

ADVISORY / DIAGNOSTICS ONLY (see docs/CUTSELL_DECISIONS.md D-122 and
docs/CUTSELL_BESTTAKE_CASE_B_FORENSIC_D121.md, the forensic this module
implements the evidence-infrastructure recommendation of). This module:

- computes NOTHING new about a take's performance -- it is a pure
  re-projection of D-115's already-computed, already-tested
  `positioned_performance_evidence.py` output, filtered to events whose
  `zone == DELIVERY` (D-115's own canonical classification: an event that
  overlaps the measured DELIVERY span, including one that also straddles
  into ENTRY/EXIT territory -- D-115 never leaves that to a second zone);
- introduces NO score, weight, threshold, or confidence cutoff of its own.
  `count_by_kind`/`duration_by_kind`/`event_density` are factual
  aggregates (a count, a sum, a ratio) -- never a GOOD/BAD/PASS/FAIL
  verdict;
- never changes `CandidateTake`, `MediaSignals`, `RankedTake`,
  `take_judge.rank_takes`/`score_take` ordering, `_semantic_best_take`'s
  decision, `deterministic_best_take_authority`'s moves, selection
  membership, grouping, or BoundaryEngine. Every function here is called
  strictly for diagnostics, exactly like `positioned_performance_
  evidence.py` itself (D-115's own module docstring rule, reused
  verbatim for this layer);
- makes the D-121 double-counting risk INSPECTABLE rather than resolving
  it: for every projected DELIVERY event it reports (a) which existing
  `MediaSignals` scalar(s) that event KIND already structurally
  contributes to whenever `local_performance.apply_local_performance_to_
  takes` runs (a static, code-derived fact -- see
  `MEDIASIGNALS_PROVENANCE` below, read directly off that function's own
  `body_n`/`face_n`/`disengage_n` bucket assignment, never guessed from
  field names) and (b) whether THIS SPECIFIC event, for THIS SPECIFIC
  take, would also be counted by D-097's independent `take_judge.
  delivery_cleanliness_evidence` -- computed by calling that module's own
  private interior-window/threshold helpers directly (`_interior_events`,
  `_RESET_KINDS`, `_BREAK_KINDS`, `_CLEANLINESS_EDGE_MARGIN_SEC`), never a
  re-typed copy of its margin or confidence constants, so this can never
  silently drift from D-097's real behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .contracts import CandidateTake
from .positioned_performance_evidence import (
    LOCAL_PERFORMANCE_EVENT_KINDS,
    ZONE_DELIVERY,
    build_positioned_performance_evidence,
)
from .take_judge import (
    _BREAK_KINDS,
    _CLEANLINESS_EDGE_MARGIN_SEC,
    _RESET_KINDS,
    _interior_events,
)
from .whole_video_analysis import WholeVideoContext

SCHEMA_VERSION = "cutsell.case_b_performance_evidence.v1"

# D-121 Section 9 / D-096 duplication register: the ONE existing code path
# that already folds these four D-114/D-115 event kinds into a MediaSignals
# scalar is `local_performance.py::apply_local_performance_to_takes`. This
# mapping is read directly off that function's own bucket assignment:
#   body_n     = count of {body_reset_candidate, hand_motion_reset_candidate}
#   face_n     = count of facial_expression_shift_candidate
#   disengage_n = count of camera_disengagement_candidate
#   visual_fumble          = f(body_n, face_n)
#   expression_naturalness = f(face_n)
#   gesture_naturalness    = f(body_n)
#   distraction_risk       = f(disengage_n)
# body_reset_candidate and hand_motion_reset_candidate share the SAME
# body_n bucket -- both structurally feed visual_fumble AND gesture_
# naturalness, never gesture_naturalness alone for one and fumble alone
# for the other. This is a STATIC, code-derived fact about what each event
# KIND structurally contributes to whenever that pipeline stage runs -- it
# is not a per-run confirmation that the stage actually ran for a given
# take (this module never fabricates that confirmation; see the module
# docstring's "no default-signal fabrication" rule).
MEDIASIGNALS_PROVENANCE: Mapping[str, Tuple[str, ...]] = {
    "body_reset_candidate": ("visual_fumble", "gesture_naturalness"),
    "hand_motion_reset_candidate": ("visual_fumble", "gesture_naturalness"),
    "facial_expression_shift_candidate": ("visual_fumble", "expression_naturalness"),
    "camera_disengagement_candidate": ("distraction_risk",),
}
MEDIASIGNALS_PROVENANCE_PRODUCER = "local_performance.apply_local_performance_to_takes"


@dataclass(frozen=True)
class CaseBEvent:
    """One D-115 DELIVERY-zone event, re-projected with provenance/overlap
    metadata. Never a score -- every field is a fact traceable back to the
    original `PositionedEvent`/`TemporalEvent` timestamp."""
    kind: str
    start: float
    end: float
    confidence: float
    duration: float
    evidence_source: str
    straddle: bool
    mediasignal_fields: Tuple[str, ...]
    d097_geometrically_inside: bool
    d097_meets_confidence_floor: bool
    d097_would_be_counted: bool


@dataclass(frozen=True)
class CaseBPerformanceEvidence:
    """The full CASE B evidence record for one BestTake competitor.
    Additive and advisory -- nothing in the active pipeline reads this to
    make a scoring, ranking, or winner decision (this module's own scope
    rule, mirroring D-115's identical guarantee for its own output)."""
    candidate_id: str
    source_asset_id: str
    delivery_available: bool
    delivery_start: float | None
    delivery_end: float | None
    delivery_span_duration: float | None
    delivery_events: Tuple[CaseBEvent, ...]
    delivery_event_count: int
    delivery_event_duration_total: float
    count_by_kind: Mapping[str, int]
    duration_by_kind: Mapping[str, float]
    event_density: float | None


def _d097_checks(
    take: CandidateTake, kind: str, start: float, end: float, confidence: float,
) -> tuple[bool, bool]:
    """Replay D-097's OWN interior-window geometry and confidence floor
    for this exact event, by calling its real private helpers directly
    (never a re-typed copy of `_CLEANLINESS_EDGE_MARGIN_SEC`/0.88/0.76)."""
    fake_event = {
        "start": start, "end": end, "kind": kind, "confidence": confidence,
        "source_asset_id": take.source_asset_id,
    }
    inside = bool(_interior_events(take, (fake_event,), margin_sec=_CLEANLINESS_EDGE_MARGIN_SEC))
    if kind in _RESET_KINDS:
        meets_floor = confidence >= 0.88
    elif kind in _BREAK_KINDS:
        meets_floor = confidence >= 0.76
    else:
        meets_floor = False
    return inside, meets_floor


def build_case_b_performance_evidence(
    candidate: CandidateTake,
    context: WholeVideoContext | None,
) -> CaseBPerformanceEvidence:
    """Build the CASE B evidence record for `candidate`, filtered to D-115's
    own DELIVERY-zone classification -- never ENTRY/EXIT-only events (those
    remain D-116 Boundary's territory, per this task's explicit scope), and
    never a new event window or threshold of this module's own invention."""
    evidence = build_positioned_performance_evidence(
        candidate, context, event_kinds=LOCAL_PERFORMANCE_EVENT_KINDS,
    )
    delivery = evidence.delivery_span
    delivery_events: list[CaseBEvent] = []
    count_by_kind: dict[str, int] = {}
    duration_by_kind: dict[str, float] = {}
    total_duration = 0.0

    for event in evidence.positioned_events:
        if event.zone != ZONE_DELIVERY:
            continue
        duration = max(0.0, event.end - event.start)
        inside, meets_floor = _d097_checks(candidate, event.kind, event.start, event.end, event.confidence)
        delivery_events.append(CaseBEvent(
            kind=event.kind,
            start=round(event.start, 3),
            end=round(event.end, 3),
            confidence=round(event.confidence, 4),
            duration=round(duration, 3),
            evidence_source=event.evidence_source,
            straddle=bool(event.starts_before_delivery or event.ends_after_delivery),
            mediasignal_fields=MEDIASIGNALS_PROVENANCE.get(event.kind, ()),
            d097_geometrically_inside=inside,
            d097_meets_confidence_floor=meets_floor,
            d097_would_be_counted=bool(inside and meets_floor),
        ))
        count_by_kind[event.kind] = count_by_kind.get(event.kind, 0) + 1
        duration_by_kind[event.kind] = round(duration_by_kind.get(event.kind, 0.0) + duration, 3)
        total_duration += duration

    delivery_events.sort(key=lambda item: (item.start, item.end, item.kind))
    span_duration = (delivery.end - delivery.start) if delivery.available else None
    density = (
        round(total_duration / span_duration, 4)
        if (span_duration is not None and span_duration > 0)
        else None
    )

    return CaseBPerformanceEvidence(
        candidate_id=candidate.clip_id,
        source_asset_id=candidate.source_asset_id,
        delivery_available=delivery.available,
        delivery_start=round(delivery.start, 3) if delivery.available else None,
        delivery_end=round(delivery.end, 3) if delivery.available else None,
        delivery_span_duration=round(span_duration, 3) if span_duration is not None else None,
        delivery_events=tuple(delivery_events),
        delivery_event_count=len(delivery_events),
        delivery_event_duration_total=round(total_duration, 3),
        count_by_kind=dict(sorted(count_by_kind.items())),
        duration_by_kind=dict(sorted(duration_by_kind.items())),
        event_density=density,
    )


def case_b_performance_evidence_diagnostics(evidence: CaseBPerformanceEvidence) -> dict:
    """JSON-safe diagnostic row for one competitor. Bounded: no unbounded
    duplication of raw events beyond what `positioned_performance_
    evidence.py` itself already caps per take."""
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": evidence.candidate_id,
        "source_asset_id": evidence.source_asset_id,
        "delivery_available": evidence.delivery_available,
        "delivery_start": evidence.delivery_start,
        "delivery_end": evidence.delivery_end,
        "delivery_span_duration": evidence.delivery_span_duration,
        "delivery_event_count": evidence.delivery_event_count,
        "delivery_event_duration_total": evidence.delivery_event_duration_total,
        "count_by_kind": dict(evidence.count_by_kind),
        "duration_by_kind": dict(evidence.duration_by_kind),
        "event_density": evidence.event_density,
        "mediasignals_provenance_producer": MEDIASIGNALS_PROVENANCE_PRODUCER,
        "delivery_events": [
            {
                "kind": e.kind,
                "start": e.start,
                "end": e.end,
                "confidence": e.confidence,
                "duration": e.duration,
                "evidence_source": e.evidence_source,
                "straddle": e.straddle,
                "mediasignal_fields": list(e.mediasignal_fields),
                "d097_geometrically_inside": e.d097_geometrically_inside,
                "d097_meets_confidence_floor": e.d097_meets_confidence_floor,
                "d097_would_be_counted": e.d097_would_be_counted,
            }
            for e in evidence.delivery_events
        ],
    }
