"""D-115 -- the ONE canonical position-aware performance evidence layer.

Proves `positioned_performance_evidence.py` derives DELIVERY from existing
ASR word timestamps, classifies existing real visual/performance events into
ENTRY/DELIVERY/EXIT (recording straddling explicitly), preserves that
evidence through attempt merging, and changes NOTHING about scalar
MediaSignals, DeliveryScorer/BestTake scoring, BoundaryEngine trimming, or
the render plan. Generic fixtures only -- no Video00 text/ids/timestamps.
"""
from dataclasses import replace

import pytest

from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.audio_silence import AUDIO_SILENCE_EVENT_KIND
from cutsell_worker.boundary_engine_pass import apply_post_freeze_boundary_pass
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, JobState,
    MediaSignals, ProcessingResult, SCHEMA_VERSION, SemanticRole, Word,
)
from cutsell_worker.positioned_performance_evidence import (
    DELIVERY_SPAN_SOURCE_UNAVAILABLE,
    DELIVERY_SPAN_SOURCE_WORD_ENVELOPE,
    LOCAL_PERFORMANCE_EVENT_KINDS,
    POSITIONED_EVENT_KINDS,
    ZONE_DELIVERY,
    ZONE_ENTRY,
    ZONE_EXIT,
    ZONE_UNKNOWN,
    build_positioned_performance_evidence,
    build_positioned_performance_evidence_for_takes,
    compute_delivery_span,
    positioned_performance_evidence_diagnostics,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_judge import rank_takes, score_take
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _words(text: str, start: float, end: float):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _take(clip_id, start, end, text, *, source="src", words=None, signals=None):
    return CandidateTake(
        clip_id, source, 0, start, end, text,
        words=words if words is not None else _words(text, start, end),
        signals=signals,
    )


def _event(kind, start, end, *, source="src", confidence=0.9, description="candidate"):
    return TemporalEvent(source, start, end, kind, confidence, description)


def _context(source_id, events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id=source_id, summary="", dominant_style="", creator_intent="",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


# --- delivery span derivation ------------------------------------------------

def test_delivery_span_derived_from_word_envelope():
    take = _take("c1", 2.0, 5.0, "one two three")
    span = compute_delivery_span(take.words)
    assert span.available is True
    assert span.source == DELIVERY_SPAN_SOURCE_WORD_ENVELOPE
    assert span.start == pytest.approx(2.0)
    assert span.end == pytest.approx(5.0)


def test_no_words_delivery_span_unknown_no_fabricated_zone():
    take = _take("c1", 2.0, 5.0, "unused", words=())
    context = _context("src", [_event("body_reset_candidate", 4.5, 4.9)])
    evidence = build_positioned_performance_evidence(take, context)
    assert evidence.delivery_span.available is False
    assert evidence.delivery_span.start is None and evidence.delivery_span.end is None
    assert evidence.delivery_span.source == DELIVERY_SPAN_SOURCE_UNAVAILABLE
    assert len(evidence.positioned_events) == 1
    event = evidence.positioned_events[0]
    assert event.zone == ZONE_UNKNOWN
    assert event.overlaps_delivery is None
    assert event.starts_before_delivery is None
    assert event.ends_after_delivery is None


# --- ENTRY / DELIVERY / EXIT classification ----------------------------------

def test_event_fully_before_speech_is_entry():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("camera_disengagement_candidate", 0.2, 0.9)])
    evidence = build_positioned_performance_evidence(take, context)
    event = evidence.positioned_events[0]
    assert event.zone == ZONE_ENTRY
    assert event.overlaps_delivery is False
    assert event.starts_before_delivery is True
    assert event.ends_after_delivery is False


def test_event_fully_inside_speech_is_delivery_overlap():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("hand_motion_reset_candidate", 3.0, 3.5)])
    evidence = build_positioned_performance_evidence(take, context)
    event = evidence.positioned_events[0]
    assert event.zone == ZONE_DELIVERY
    assert event.overlaps_delivery is True
    assert event.starts_before_delivery is False
    assert event.ends_after_delivery is False


def test_event_fully_after_speech_is_exit():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("body_reset_candidate", 5.4, 5.9)])
    evidence = build_positioned_performance_evidence(take, context)
    event = evidence.positioned_events[0]
    assert event.zone == ZONE_EXIT
    assert event.overlaps_delivery is False
    assert event.starts_before_delivery is False
    assert event.ends_after_delivery is True


def test_event_crossing_delivery_start_records_straddle():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("facial_expression_shift_candidate", 1.5, 2.5)])
    evidence = build_positioned_performance_evidence(take, context)
    event = evidence.positioned_events[0]
    # Any overlap with DELIVERY is classified DELIVERY, but the straddle into
    # ENTRY territory must still be recorded explicitly, never silently lost.
    assert event.zone == ZONE_DELIVERY
    assert event.overlaps_delivery is True
    assert event.starts_before_delivery is True
    assert event.ends_after_delivery is False


def test_event_crossing_delivery_end_records_straddle():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("body_reset_candidate", 4.5, 5.5)])
    evidence = build_positioned_performance_evidence(take, context)
    event = evidence.positioned_events[0]
    assert event.zone == ZONE_DELIVERY
    assert event.overlaps_delivery is True
    assert event.starts_before_delivery is False
    assert event.ends_after_delivery is True


def test_multiple_event_kinds_preserve_independent_timing():
    take = _take("c1", 0.0, 8.0, "one two three four", words=_words("one two three four", 2.0, 6.0))
    context = _context("src", [
        _event("camera_disengagement_candidate", 0.1, 0.4),
        _event("hand_motion_reset_candidate", 3.0, 3.2),
        _event("body_reset_candidate", 6.5, 6.9),
    ])
    evidence = build_positioned_performance_evidence(take, context)
    assert len(evidence.positioned_events) == 3
    by_kind = {event.kind: event for event in evidence.positioned_events}
    assert by_kind["camera_disengagement_candidate"].zone == ZONE_ENTRY
    assert by_kind["camera_disengagement_candidate"].start == pytest.approx(0.1)
    assert by_kind["hand_motion_reset_candidate"].zone == ZONE_DELIVERY
    assert by_kind["hand_motion_reset_candidate"].start == pytest.approx(3.0)
    assert by_kind["body_reset_candidate"].zone == ZONE_EXIT
    assert by_kind["body_reset_candidate"].start == pytest.approx(6.5)


def test_default_only_signal_kinds_never_gain_fake_positioned_evidence():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    # A kind that is NOT one of D-114's proven real event kinds (e.g. a
    # hypothetical framing/audio-quality-style candidate) must be excluded
    # even though it temporally overlaps the take window.
    context = _context("src", [_event("framing_quality_candidate", 3.0, 3.5)])
    evidence = build_positioned_performance_evidence(take, context)
    assert evidence.positioned_events == ()
    assert "framing_quality_candidate" not in POSITIONED_EVENT_KINDS


def test_audio_silence_interval_included_as_real_positioned_event():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event(AUDIO_SILENCE_EVENT_KIND, 5.2, 5.9, confidence=1.0)])
    evidence = build_positioned_performance_evidence(take, context)
    assert len(evidence.positioned_events) == 1
    assert evidence.positioned_events[0].evidence_source == "audio_silence"
    assert evidence.positioned_events[0].zone == ZONE_EXIT


def test_events_outside_take_window_are_excluded():
    take = _take("c1", 10.0, 16.0, "one two three", words=_words("one two three", 11.0, 15.0))
    context = _context("src", [_event("body_reset_candidate", 0.0, 0.5)])
    evidence = build_positioned_performance_evidence(take, context)
    assert evidence.positioned_events == ()


def test_events_from_a_different_source_are_excluded():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), source="src_a")
    context = _context("src_b", [_event("body_reset_candidate", 3.0, 3.5, source="src_b")])
    evidence = build_positioned_performance_evidence(take, context)
    assert evidence.positioned_events == ()


def test_context_none_is_safe():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    evidence = build_positioned_performance_evidence(take, None)
    assert evidence.positioned_events == ()
    assert evidence.delivery_span.available is True


# --- attempt-merge preservation ----------------------------------------------

def test_fused_attempt_preserves_positioned_events_without_averaging():
    # Two raw takes close enough to be fused into one delivery attempt by
    # reconstruct_delivery_attempts (default max_continuation_gap_sec=1.20).
    member_a = _take("a", 0.0, 3.0, "one two", words=_words("one two", 0.2, 2.8))
    member_b = _take("b", 3.5, 6.0, "three four", words=_words("three four", 3.7, 5.8))
    context = _context("src", [
        _event("body_reset_candidate", 5.85, 5.95),   # after member_b's last word (5.8) -> EXIT
        _event("hand_motion_reset_candidate", 1.0, 1.3),  # inside member_a -> DELIVERY
    ])
    attempts, _diag = reconstruct_delivery_attempts((member_a, member_b), context)
    assert len(attempts) == 1  # confirms the fixture actually fused
    fused = attempts[0]
    assert fused.start == pytest.approx(0.0) and fused.end == pytest.approx(6.0)

    evidence = build_positioned_performance_evidence(fused, context)
    assert len(evidence.positioned_events) == 2  # both members' events preserved, not averaged away
    kinds = {event.kind: event for event in evidence.positioned_events}
    assert kinds["body_reset_candidate"].start == pytest.approx(5.85)  # exact timing, untouched
    assert kinds["body_reset_candidate"].zone == ZONE_EXIT
    assert kinds["hand_motion_reset_candidate"].start == pytest.approx(1.0)
    assert kinds["hand_motion_reset_candidate"].zone == ZONE_DELIVERY


# --- diagnostics shape --------------------------------------------------------

def test_diagnostics_contain_provenance_and_source_timing():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("body_reset_candidate", 5.4, 5.9, confidence=0.77)])
    evidence = build_positioned_performance_evidence_for_takes((take,), context)
    rows = positioned_performance_evidence_diagnostics(evidence)
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "c1"
    assert row["source_asset_id"] == "src"
    assert row["source_start"] == pytest.approx(0.0)
    assert row["source_end"] == pytest.approx(6.0)
    assert row["delivery_span"]["available"] is True
    assert row["delivery_span"]["start"] == pytest.approx(2.0)
    assert row["delivery_span"]["end"] == pytest.approx(5.0)
    assert row["positioned_event_count"] == 1
    event_row = row["positioned_events"][0]
    assert event_row["kind"] == "body_reset_candidate"
    assert event_row["zone"] == ZONE_EXIT
    assert event_row["evidence_source"] == "local_performance"
    assert event_row["confidence"] == pytest.approx(0.77)


def test_diagnostics_include_rows_with_no_events_or_unavailable_span():
    take_no_words = _take("c1", 0.0, 3.0, "unused", words=())
    take_no_events = _take("c2", 3.0, 6.0, "one two", words=_words("one two", 3.2, 5.8))
    evidence = build_positioned_performance_evidence_for_takes((take_no_words, take_no_events), None)
    rows = positioned_performance_evidence_diagnostics(evidence)
    assert len(rows) == 2
    assert rows[0]["delivery_span"]["available"] is False
    assert rows[0]["positioned_event_count"] == 0
    assert rows[1]["delivery_span"]["available"] is True
    assert rows[1]["positioned_event_count"] == 0


# --- no-editorial-authority / no-scoring-change / no-boundary-change --------

def test_scalar_mediasignals_unchanged_by_evidence_computation():
    signals = MediaSignals(source_asset_id="src", start=0.0, end=6.0, visual_fumble=0.2)
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), signals=signals)
    context = _context("src", [_event("body_reset_candidate", 5.4, 5.9)])
    build_positioned_performance_evidence(take, context)  # call for side effects only
    assert take.signals is signals  # identity-unchanged: nothing was mutated or replaced


def test_deliveryscore_besttake_ranking_unchanged_before_after():
    signals = MediaSignals(source_asset_id="src", start=0.0, end=6.0, visual_fumble=0.2)
    takes = (
        _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), signals=signals),
        _take("c2", 7.0, 12.0, "four five six", words=_words("four five six", 7.5, 11.5)),
    )
    context = _context("src", [
        _event("body_reset_candidate", 5.4, 5.9),
        _event(AUDIO_SILENCE_EVENT_KIND, 11.6, 11.9),
    ])

    before_scores = [score_take(take) for take in takes]
    before_ranked = rank_takes(takes)

    build_positioned_performance_evidence_for_takes(takes, context)  # D-115 layer, diagnostics only

    after_scores = [score_take(take) for take in takes]
    after_ranked = rank_takes(takes)

    assert before_scores == after_scores
    assert before_ranked == after_ranked


def test_boundary_pass_unaffected_by_new_diagnostics_key():
    words = _words("one two three four", 10.5, 15.5)
    clip = DraftClip(
        clip_id="c1", source_asset_id="src", source_order=0, start=10.0, end=16.0,
        text="one two three four", caption_text="one two three four", words=words,
        semantic_role=SemanticRole.STORY, selected=True,
    )
    base_diagnostics = {
        "whole_video_context": {"sources": [{
            "source_asset_id": "src",
            "events": [{"kind": AUDIO_SILENCE_EVENT_KIND, "start": 9.4, "end": 10.9, "confidence": 1.0}],
        }]},
    }
    context = _context("src", [_event(AUDIO_SILENCE_EVENT_KIND, 9.4, 10.9, confidence=1.0)])
    evidence_rows = positioned_performance_evidence_diagnostics(
        build_positioned_performance_evidence_for_takes(
            (_take("c1", 10.0, 16.0, "one two three four", words=words),), context,
        )
    )

    def _draft(diagnostics):
        return DraftTimeline(
            schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
            selected=(clip,), alternates=(), discarded=(), diagnostics=diagnostics,
        )

    def _result(draft):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )

    result_without = apply_post_freeze_boundary_pass(_result(_draft(dict(base_diagnostics))))
    result_with = apply_post_freeze_boundary_pass(_result(_draft({
        **base_diagnostics,
        "positioned_performance_evidence": evidence_rows,
    })))

    assert result_without.draft.selected == result_with.draft.selected
