"""D-155 Phase A -- Structured RAW Understanding Map V1.

Per docs/CUTSELL_DECISIONS.md D-148/D-154/D-155. This module is the FIRST
concrete implementation of D-148 Section 13.3.1's "Structured RAW
Understanding Map" and D-154's own design, restricted exactly to that
task's own V1 boundary: a PURE, ADDITIVE PROJECTION of evidence Tracks
A-D ALREADY compute today (ASR, `audio_silence.py`, `local_performance.py`
+ `positioned_performance_evidence.py`, `media_probe.py`) into one
reusable, bounded container per source. It computes NOTHING new about
speech, audio, or visual content -- every field here is copied or
trivially derived (set-membership, presence/absence, counting) from
values `flow_b.py`'s existing pipeline already produced.

## What this module is NOT (D-154's own authority boundary, unchanged)

This module NEVER decides:
- final Proposition Identity (Milestone-1 text/structure-led stage, D-098
  13.5, untouched);
- final retry family / Attempt Relationship (D-145's 5-way vocabulary,
  untouched -- `attempt_relation`/`proposition_relation` fields on
  `RawUnderstandingSpan` are deliberately left `None` in V1: this module
  supplies EVIDENCE only, never an authority verdict);
- BestTake, Boundary, Dialogue/Pacing, or the Renderer (all untouched --
  nothing in `cutsell_worker` imports this module from any of those
  authorities as of this task; confirmed via the module-leaf tests in
  `tests/test_cutsell_d155_raw_understanding_map.py`);
- D-146/D-149/D-150's semantic-authority gate (D-145-D-153 thread,
  PRESERVED CLOSED -- this module never reads or writes
  `family_complete_context`/`complete_context_conflict`/`semantic_
  authority_gate_status`, and is never imported by `semantic_authority_
  observability.py` or `pipeline.py`'s D-150 call site).

## Behavior hypotheses (bounded, per D-154/D-155's own directive)

`BehaviorHypothesis` may only take one of the 8 labels in
`ALLOWED_BEHAVIOR_HYPOTHESES` below -- RETRY/CONTINUATION/CORRECTION/
COMPLEMENTARY are deliberately excluded from that set (those describe a
RELATIONSHIP between two spans, decided by D-145's own Attempt
Relationship vocabulary, never inferred here from a single span's local
performance evidence alone). Per the real, verified event-kind inventory
this task confirmed by direct code read:

- `local_performance.py` (the only live Track C producer) emits exactly
  four kinds: `camera_disengagement_candidate`, `facial_expression_
  shift_candidate`, `body_reset_candidate`, `hand_motion_reset_
  candidate` -- all reset/disengagement-family evidence, mapped to
  `POST_TAKE_RESET`.
- `performance_confirmation.confirm_local_performance_events` (D-100's
  multimodal-corroboration stage, a real, live, DETERMINISTIC_RULE
  producer that runs AFTER local_performance/segmentation) additionally
  emits `wrong_take`/`retry_setup` kind events -- mapped to `ABANDONED_
  ATTEMPT` (a statement about THIS span's own completeness, never a
  cross-span RETRY relationship).
- `false_start`/`breaking_character` are valid labels in the allowed
  vocabulary (kept for schema completeness and forward compatibility --
  a future producer may emit them) but are NEVER PRODUCED by any live
  component this task found; they are honestly expected to almost never
  fire against today's real event inventory. This is reported plainly,
  not smoothed over.
- `AUDIENCE_DELIVERY`/`CLEAN_ATTEMPT` are DERIVED (absence-of-evidence)
  inferences, tagged `MULTIMODAL_FUSION` provenance, never claimed as a
  directly observed signal.
- `PRE_TAKE_SETUP` is in the allowed vocabulary but this deriver never
  emits it -- D-154 classified it `MISSING_EVIDENCE` (no existing signal
  justifies it) and this task does not invent one.

`conflict_flags` is honestly empty in this V1 deriver -- no two
independent evidence sources are cross-validated per span yet (that is
future Watch+Listen Fusion work, D-154 Phase B), so no genuine conflict
can be detected here; the field exists in the schema (never silently
dropped) for that future use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals, Word
from .media_probe import MediaProbe
from .positioned_performance_evidence import (
    LOCAL_PERFORMANCE_EVENT_KINDS,
    POSITIONED_EVENT_KINDS,
    ZONE_DELIVERY,
    PositionAwarePerformanceEvidence,
    PositionedEvent,
    build_positioned_performance_evidence_for_takes,
)
from .whole_video_analysis import TemporalEvent, WholeVideoContext

SCHEMA_VERSION = "cutsell.raw_understanding_map.v1"

# ---------------------------------------------------------------------------
# Evidence provenance tags (D-154's own vocabulary, verbatim).
# ---------------------------------------------------------------------------
PROVENANCE_ASR = "ASR"
PROVENANCE_AUDIO_SIGNAL = "AUDIO_SIGNAL"
PROVENANCE_VISUAL_SIGNAL = "VISUAL_SIGNAL"
PROVENANCE_MEDIA_TIMING = "MEDIA_TIMING"
PROVENANCE_DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
PROVENANCE_SEMANTIC_PROVIDER = "SEMANTIC_PROVIDER"
PROVENANCE_MULTIMODAL_FUSION = "MULTIMODAL_FUSION"
PROVENANCE_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Per-track status (D-155's own directive vocabulary).
# ---------------------------------------------------------------------------
TRACK_STATUS_PASS = "PASS"
TRACK_STATUS_PARTIAL = "PARTIAL"
TRACK_STATUS_FAILED = "FAILED"
TRACK_STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"

# ---------------------------------------------------------------------------
# Overall map status.
# ---------------------------------------------------------------------------
MAP_STATUS_COMPLETE_EXISTING_EVIDENCE = "COMPLETE_EXISTING_EVIDENCE"
MAP_STATUS_PARTIAL_EXISTING_EVIDENCE = "PARTIAL_EXISTING_EVIDENCE"
MAP_STATUS_FAILED = "FAILED"

# ---------------------------------------------------------------------------
# Behavior hypothesis vocabulary -- bounded to 8 labels; RETRY/CONTINUATION/
# CORRECTION/COMPLEMENTARY are cross-span RELATIONSHIPS and are deliberately
# excluded (D-145's own Attempt Relationship authority, untouched).
# ---------------------------------------------------------------------------
BEHAVIOR_AUDIENCE_DELIVERY = "AUDIENCE_DELIVERY"
BEHAVIOR_PRE_TAKE_SETUP = "PRE_TAKE_SETUP"
BEHAVIOR_FALSE_START = "FALSE_START"
BEHAVIOR_ABANDONED_ATTEMPT = "ABANDONED_ATTEMPT"
BEHAVIOR_CLEAN_ATTEMPT = "CLEAN_ATTEMPT"
BEHAVIOR_POST_TAKE_RESET = "POST_TAKE_RESET"
BEHAVIOR_RECORDING_PROCESS = "RECORDING_PROCESS"
BEHAVIOR_BREAKING_CHARACTER = "BREAKING_CHARACTER"

ALLOWED_BEHAVIOR_HYPOTHESES: frozenset[str] = frozenset({
    BEHAVIOR_AUDIENCE_DELIVERY, BEHAVIOR_PRE_TAKE_SETUP, BEHAVIOR_FALSE_START,
    BEHAVIOR_ABANDONED_ATTEMPT, BEHAVIOR_CLEAN_ATTEMPT, BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_RECORDING_PROCESS, BEHAVIOR_BREAKING_CHARACTER,
})

# Explicitly-forbidden cross-span relationship labels -- never emitted by
# this module; asserted against in tests as a structural guarantee.
FORBIDDEN_RELATIONSHIP_LABELS: frozenset[str] = frozenset({
    "RETRY", "CONTINUATION", "CORRECTION", "COMPLEMENTARY",
})

_RESET_FAMILY_KINDS: frozenset[str] = LOCAL_PERFORMANCE_EVENT_KINDS  # the 4 real, live Track C kinds
_ABANDONED_KINDS: frozenset[str] = frozenset({"wrong_take", "retry_setup"})
_RECORDING_PROCESS_KINDS: frozenset[str] = frozenset({
    "recording_joke", "verbal_fumble", "product_handling_mistake",
    "accidental_laughter", "searching_for_words",
})
_FALSE_START_KINDS: frozenset[str] = frozenset({"false_start"})
_BREAKING_CHARACTER_KINDS: frozenset[str] = frozenset({"breaking_character"})

# `positioned_performance_evidence.py`'s own default `event_kinds` filter
# (`POSITIONED_EVENT_KINDS`) is intentionally narrow -- the 4 D-114 local-
# performance kinds + `audio_silence_interval` (D-115's own scope). This
# module's behavior-hypothesis vocabulary is wider (D-100's `wrong_take`/
# `retry_setup`, the recording-process family, `false_start`/
# `breaking_character`), so it passes this WIDER, explicit `event_kinds`
# set into D-115's own already-exposed `event_kinds` parameter -- reusing
# the exact same shared `classify_event_zone`/windowing logic, never
# reimplementing zoning. Without this, every one of those kinds would be
# silently filtered out before ever reaching `_behavior_hypotheses_for_span`,
# and ABANDONED_ATTEMPT/RECORDING_PROCESS/FALSE_START/BREAKING_CHARACTER
# would be permanently unreachable dead code.
_BEHAVIOR_RELEVANT_EVENT_KINDS: frozenset[str] = (
    POSITIONED_EVENT_KINDS
    | _ABANDONED_KINDS
    | _RECORDING_PROCESS_KINDS
    | _FALSE_START_KINDS
    | _BREAKING_CHARACTER_KINDS
)


def _provenance_for_kind(kind: str) -> str:
    if kind in _RESET_FAMILY_KINDS or kind in _FALSE_START_KINDS or kind in _BREAKING_CHARACTER_KINDS:
        return PROVENANCE_VISUAL_SIGNAL
    if kind in _ABANDONED_KINDS or kind in _RECORDING_PROCESS_KINDS:
        return PROVENANCE_DETERMINISTIC_RULE
    return PROVENANCE_UNKNOWN


@dataclass(frozen=True)
class BehaviorHypothesis:
    """One bounded behavior-state hypothesis for a span -- evidence, never
    an authority verdict. `confidence` is copied/derived from the
    underlying event(s), never invented."""
    label: str
    confidence: float
    provenance: str
    basis: str


@dataclass(frozen=True)
class RawUnderstandingSpan:
    """One bounded span's fused evidence (today: one per `CandidateTake`,
    the existing span identity -- no new identity minted)."""
    span_id: str
    source_asset_id: str
    source_start: float
    source_end: float
    transcript: str
    word_timings: Tuple[Word, ...]
    positioned_evidence: PositionAwarePerformanceEvidence
    behavior_hypotheses: Tuple[BehaviorHypothesis, ...]
    conflict_flags: Tuple[str, ...]
    evidence_provenance: Mapping[str, str]


@dataclass(frozen=True)
class RawUnderstandingMap:
    """The per-source Structured RAW Understanding Map V1 -- one reusable
    container assembled from existing Track A-D evidence, read by nothing
    yet (V1 is diagnostics-only; see module docstring)."""
    source_asset_id: str
    source_duration: float
    source_timeline_origin: str
    transcript: str
    word_timings: Tuple[Word, ...]
    speech_spans: Tuple = ()  # Tuple[TranscriptSegment, ...] -- ASR's own segment objects, reused verbatim
    audio_events: Tuple[TemporalEvent, ...] = ()
    audio_signal_status: str = TRACK_STATUS_NOT_AVAILABLE
    visual_performance_events: Tuple[TemporalEvent, ...] = ()
    positioned_performance_evidence: Tuple[PositionAwarePerformanceEvidence, ...] = ()
    media_facts: Mapping = field(default_factory=dict)
    candidate_span_evidence: Tuple[RawUnderstandingSpan, ...] = ()
    behavior_hypotheses: Tuple[BehaviorHypothesis, ...] = ()
    conflict_flags: Tuple[str, ...] = ()
    evidence_provenance: Mapping[str, str] = field(default_factory=dict)
    track_status: Mapping[str, str] = field(default_factory=dict)


SOURCE_TIMELINE_ORIGIN = "source_relative_seconds"


def _behavior_hypotheses_for_span(
    positioned_events: Tuple[PositionedEvent, ...],
    delivery_available: bool,
) -> Tuple[BehaviorHypothesis, ...]:
    """Pure, bounded deriver -- see module docstring for the exact,
    verified event-kind -> label mapping. Never emits a label outside
    `ALLOWED_BEHAVIOR_HYPOTHESES`."""
    kinds_present: dict[str, PositionedEvent] = {}
    for event in positioned_events:
        existing = kinds_present.get(event.kind)
        if existing is None or event.confidence > existing.confidence:
            kinds_present[event.kind] = event

    hypotheses: list[BehaviorHypothesis] = []

    if _BREAKING_CHARACTER_KINDS & kinds_present.keys():
        event = kinds_present[next(iter(_BREAKING_CHARACTER_KINDS & kinds_present.keys()))]
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_BREAKING_CHARACTER, float(event.confidence), PROVENANCE_VISUAL_SIGNAL,
            "breaking_character event present",
        ))
    if _FALSE_START_KINDS & kinds_present.keys():
        event = kinds_present[next(iter(_FALSE_START_KINDS & kinds_present.keys()))]
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_FALSE_START, float(event.confidence), PROVENANCE_VISUAL_SIGNAL,
            "false_start event present",
        ))
    abandoned_hit = _ABANDONED_KINDS & kinds_present.keys()
    if abandoned_hit:
        best = max((kinds_present[k] for k in abandoned_hit), key=lambda e: e.confidence)
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_ABANDONED_ATTEMPT, float(best.confidence), PROVENANCE_DETERMINISTIC_RULE,
            f"confirmed {sorted(abandoned_hit)} event (D-100 multimodal corroboration)",
        ))
    reset_hit = _RESET_FAMILY_KINDS & kinds_present.keys()
    if reset_hit:
        best = max((kinds_present[k] for k in reset_hit), key=lambda e: e.confidence)
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_POST_TAKE_RESET, float(best.confidence), PROVENANCE_VISUAL_SIGNAL,
            f"reset-family event(s) present: {sorted(reset_hit)}",
        ))
    recording_hit = _RECORDING_PROCESS_KINDS & kinds_present.keys()
    if recording_hit:
        best = max((kinds_present[k] for k in recording_hit), key=lambda e: e.confidence)
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_RECORDING_PROCESS, float(best.confidence), PROVENANCE_DETERMINISTIC_RULE,
            f"recording-process event(s) present: {sorted(recording_hit)}",
        ))
    if not kinds_present:
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_CLEAN_ATTEMPT, 0.5, PROVENANCE_MULTIMODAL_FUSION,
            "no break/reset/camera/face event observed for this span",
        ))
    delivery_zone_kinds = {event.kind for event in positioned_events if event.zone == ZONE_DELIVERY}
    if delivery_available and not delivery_zone_kinds:
        hypotheses.append(BehaviorHypothesis(
            BEHAVIOR_AUDIENCE_DELIVERY, 0.5, PROVENANCE_MULTIMODAL_FUSION,
            "delivery span measured; no break/reset event overlaps it",
        ))
    return tuple(hypotheses)


def _span_evidence_provenance(positioned_events: Tuple[PositionedEvent, ...]) -> Mapping[str, str]:
    provenance: dict[str, str] = {
        "transcript": PROVENANCE_ASR,
        "word_timings": PROVENANCE_ASR,
    }
    for event in positioned_events:
        provenance[f"event:{event.kind}"] = _provenance_for_kind(event.kind)
    return provenance


def build_raw_understanding_span(
    candidate: CandidateTake,
    positioned: PositionAwarePerformanceEvidence,
) -> RawUnderstandingSpan:
    """Pure projection: wraps one already-computed `PositionAwarePerformance
    Evidence` record (D-115, unchanged) plus the candidate's own transcript/
    word timings, and derives bounded behavior hypotheses from its existing
    positioned events. Recomputes nothing about ASR/audio/visual content."""
    hypotheses = _behavior_hypotheses_for_span(
        positioned.positioned_events, positioned.delivery_span.available,
    )
    return RawUnderstandingSpan(
        span_id=candidate.clip_id,
        source_asset_id=candidate.source_asset_id,
        source_start=float(candidate.start),
        source_end=float(candidate.end),
        transcript=str(candidate.text or ""),
        word_timings=tuple(candidate.words),
        positioned_evidence=positioned,
        behavior_hypotheses=hypotheses,
        conflict_flags=(),
        evidence_provenance=_span_evidence_provenance(positioned.positioned_events),
    )


def _events_for_source(context: WholeVideoContext | None, source_asset_id: str) -> Tuple[TemporalEvent, ...]:
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def build_raw_understanding_map(
    *,
    source_asset_id: str,
    source_duration: float,
    transcript_segments: Iterable,
    whole_context: WholeVideoContext | None,
    takes_for_source: Iterable[CandidateTake],
    media_probe: MediaProbe | None,
    speech_track_status: str,
    audio_track_status: str,
    visual_track_status: str,
    media_track_status: str,
) -> RawUnderstandingMap:
    """Pure builder: assembles ONE `RawUnderstandingMap` for `source_asset_id`
    from evidence Tracks A-D have ALREADY computed (passed in verbatim --
    this function reads no file, calls no provider, calls no subprocess).

    Per-span `candidate_span_evidence` reuses `positioned_performance_
    evidence.py`'s own D-115 builder directly -- no second computation of
    ENTRY/DELIVERY/EXIT zoning."""
    segments = tuple(transcript_segments)
    takes = tuple(t for t in takes_for_source if t.source_asset_id == source_asset_id)
    transcript = " ".join(str(seg.text or "").strip() for seg in segments if seg.text).strip()
    word_timings = tuple(word for seg in segments for word in seg.words)

    audio_events = tuple(
        event for event in _events_for_source(whole_context, source_asset_id)
        if event.kind == "audio_silence_interval"
    )
    visual_events = tuple(
        event for event in _events_for_source(whole_context, source_asset_id)
        if event.kind != "audio_silence_interval"
    )

    positioned_items = build_positioned_performance_evidence_for_takes(
        takes, whole_context, event_kinds=_BEHAVIOR_RELEVANT_EVENT_KINDS,
    )
    span_evidence = tuple(
        build_raw_understanding_span(take, positioned)
        for take, positioned in zip(takes, positioned_items)
    )

    all_hypotheses = tuple(h for span in span_evidence for h in span.behavior_hypotheses)
    all_conflicts = tuple(flag for span in span_evidence for flag in span.conflict_flags)

    evidence_provenance: dict[str, str] = {
        "transcript": PROVENANCE_ASR,
        "word_timings": PROVENANCE_ASR,
        "audio_events": PROVENANCE_AUDIO_SIGNAL,
        "visual_performance_events": PROVENANCE_VISUAL_SIGNAL,
        "media_facts": PROVENANCE_MEDIA_TIMING,
        "positioned_performance_evidence": PROVENANCE_DETERMINISTIC_RULE,
        "behavior_hypotheses": PROVENANCE_MULTIMODAL_FUSION,
    }

    media_facts = (
        {
            "duration_sec": media_probe.duration_sec,
            "width": media_probe.width,
            "height": media_probe.height,
            "fps": media_probe.fps,
            "has_audio": media_probe.has_audio,
        }
        if media_probe is not None else {}
    )

    statuses = (speech_track_status, audio_track_status, visual_track_status, media_track_status)
    if media_track_status == TRACK_STATUS_FAILED or speech_track_status == TRACK_STATUS_FAILED:
        overall = MAP_STATUS_FAILED
    elif all(status == TRACK_STATUS_PASS for status in statuses):
        overall = MAP_STATUS_COMPLETE_EXISTING_EVIDENCE
    else:
        overall = MAP_STATUS_PARTIAL_EXISTING_EVIDENCE

    track_status = {
        "speech_track_status": speech_track_status,
        "audio_track_status": audio_track_status,
        "visual_track_status": visual_track_status,
        "media_track_status": media_track_status,
        "raw_understanding_map_status": overall,
    }

    return RawUnderstandingMap(
        source_asset_id=source_asset_id,
        source_duration=float(source_duration),
        source_timeline_origin=SOURCE_TIMELINE_ORIGIN,
        transcript=transcript,
        word_timings=word_timings,
        speech_spans=segments,
        audio_events=audio_events,
        audio_signal_status=audio_track_status,
        visual_performance_events=visual_events,
        positioned_performance_evidence=positioned_items,
        media_facts=media_facts,
        candidate_span_evidence=span_evidence,
        behavior_hypotheses=all_hypotheses,
        conflict_flags=all_conflicts,
        evidence_provenance=evidence_provenance,
        track_status=track_status,
    )


def build_raw_understanding_maps_for_sources(
    *,
    sources,
    transcript_tuple,
    whole_context: WholeVideoContext | None,
    takes,
    media_probes_by_source: Mapping[str, MediaProbe],
    speech_track_status: str,
    audio_track_status: str,
    visual_track_status: str,
    media_track_status: str,
) -> Tuple[RawUnderstandingMap, ...]:
    """Batch form, one map per hydrated source, in `sources` order --
    deterministic regardless of any concurrent task's completion order
    (this function itself performs no concurrency; see
    `parallel_perception.py` for the orchestration layer that feeds it)."""
    segments_by_source: dict[str, list] = {}
    for segment in transcript_tuple:
        segments_by_source.setdefault(segment.source_asset_id, []).append(segment)

    maps = []
    for source in sources:
        maps.append(build_raw_understanding_map(
            source_asset_id=source.source_asset_id,
            source_duration=source.duration_sec,
            transcript_segments=segments_by_source.get(source.source_asset_id, ()),
            whole_context=whole_context,
            takes_for_source=takes,
            media_probe=media_probes_by_source.get(source.source_asset_id),
            speech_track_status=speech_track_status,
            audio_track_status=audio_track_status,
            visual_track_status=visual_track_status,
            media_track_status=media_track_status,
        ))
    return tuple(maps)


def raw_understanding_map_diagnostics(maps: Iterable[RawUnderstandingMap]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-119/D-125/D-152's
    own compact summaries) -- never dumps a transcript, word timing, or
    per-event payload."""
    maps = tuple(maps)
    span_count = sum(len(m.candidate_span_evidence) for m in maps)
    event_count = sum(
        len(m.audio_events) + len(m.visual_performance_events) for m in maps
    )
    conflict_count = sum(len(m.conflict_flags) for m in maps)
    overall_statuses = {m.track_status.get("raw_understanding_map_status") for m in maps}
    if not maps:
        overall = MAP_STATUS_FAILED
    elif overall_statuses == {MAP_STATUS_COMPLETE_EXISTING_EVIDENCE}:
        overall = MAP_STATUS_COMPLETE_EXISTING_EVIDENCE
    elif MAP_STATUS_FAILED in overall_statuses:
        overall = MAP_STATUS_FAILED
    else:
        overall = MAP_STATUS_PARTIAL_EXISTING_EVIDENCE
    return {
        "raw_understanding_map_created": bool(maps),
        "raw_understanding_span_count": span_count,
        "raw_understanding_event_count": event_count,
        "raw_understanding_conflict_count": conflict_count,
        "raw_understanding_map_status": overall,
        "source_count": len(maps),
    }
