"""D-155 Phase A -- Structured RAW Understanding Map V1.

Per docs/CUTSELL_DECISIONS.md D-154/D-155. `raw_understanding_map.py` is a
PURE, ADDITIVE projection of evidence Tracks A-D already compute (ASR,
`audio_silence.py`, `local_performance.py` + `positioned_performance_
evidence.py`, `media_probe.py`) onto one reusable per-source container. This
suite proves: (1) fields are copied/derived, never invented; (2) provenance
tagging is correct per real, verified event-kind producers; (3) behavior
hypotheses stay bounded to the 8-label vocabulary and never emit a
cross-span relationship label (RETRY/CONTINUATION/CORRECTION/COMPLEMENTARY
-- D-145's own authority, untouched); (4) the map never decides final
Proposition Identity/retry family/BestTake/DeliveryScorer/Boundary/pacing;
(5) deterministic batch ordering and tail-safe diagnostics.

Generic synthetic fixtures only -- no Video00-specific transcript text,
timestamps, or clip ids (CLAUDE.md: never hardcode those).
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.media_probe import MediaProbe
from cutsell_worker.positioned_performance_evidence import (
    ZONE_DELIVERY,
    ZONE_ENTRY,
    build_positioned_performance_evidence_for_takes,
)
from cutsell_worker.raw_understanding_map import (
    ALLOWED_BEHAVIOR_HYPOTHESES,
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_AUDIENCE_DELIVERY,
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_RECORDING_PROCESS,
    FORBIDDEN_RELATIONSHIP_LABELS,
    _BEHAVIOR_RELEVANT_EVENT_KINDS,
    MAP_STATUS_COMPLETE_EXISTING_EVIDENCE,
    MAP_STATUS_FAILED,
    MAP_STATUS_PARTIAL_EXISTING_EVIDENCE,
    PROVENANCE_ASR,
    PROVENANCE_AUDIO_SIGNAL,
    PROVENANCE_DETERMINISTIC_RULE,
    PROVENANCE_MEDIA_TIMING,
    PROVENANCE_MULTIMODAL_FUSION,
    PROVENANCE_VISUAL_SIGNAL,
    SCHEMA_VERSION,
    TRACK_STATUS_FAILED,
    TRACK_STATUS_NOT_AVAILABLE,
    TRACK_STATUS_PASS,
    build_raw_understanding_map,
    build_raw_understanding_maps_for_sources,
    build_raw_understanding_span,
    raw_understanding_map_diagnostics,
)
from cutsell_worker.whole_video_analysis import (
    SourceVideoContext,
    TemporalEvent,
    WholeVideoContext,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.contracts import TranscriptSegment


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


def _take(clip_id: str, source_asset_id: str = "src-1", start: float = 0.0, end: float = 2.0, text: str = "hello there") -> CandidateTake:
    words = (_word("hello", start, start + 0.5), _word("there", start + 0.6, end))
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source_asset_id,
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        signals=MediaSignals(source_asset_id=source_asset_id, start=start, end=end),
    )


def _context(source_asset_id: str, events: tuple[TemporalEvent, ...]) -> WholeVideoContext:
    return WholeVideoContext(
        sources=(SourceVideoContext(source_asset_id=source_asset_id, summary="", dominant_style="", creator_intent="", events=events),),
        status=ProviderStatus("whole_video", True, True, "applied"),
    )


def _positioned_for(take: CandidateTake, context: WholeVideoContext | None):
    # Mirrors `build_raw_understanding_map`'s own widened `event_kinds`
    # (D-100's wrong_take/retry_setup + recording-process + false_start/
    # breaking_character are outside D-115's own narrower default) -- see
    # raw_understanding_map.py's `_BEHAVIOR_RELEVANT_EVENT_KINDS` docstring.
    return build_positioned_performance_evidence_for_takes(
        (take,), context, event_kinds=_BEHAVIOR_RELEVANT_EVENT_KINDS,
    )[0]


# ---------------------------------------------------------------------------
# 1-4: pure projection -- fields copied/derived, never invented.
# ---------------------------------------------------------------------------

def test_span_preserves_source_identity_and_timing():
    take = _take("clip-1", source_asset_id="src-9", start=10.0, end=12.0)
    positioned = _positioned_for(take, None)
    span = build_raw_understanding_span(take, positioned)
    assert span.span_id == "clip-1"
    assert span.source_asset_id == "src-9"
    assert span.source_start == 10.0
    assert span.source_end == 12.0


def test_span_preserves_transcript_and_word_timings_verbatim():
    take = _take("clip-1", text="hello there")
    positioned = _positioned_for(take, None)
    span = build_raw_understanding_span(take, positioned)
    assert span.transcript == "hello there"
    assert span.word_timings == take.words


def test_span_never_recomputes_positioned_evidence_reuses_same_object():
    take = _take("clip-1")
    positioned = _positioned_for(take, None)
    span = build_raw_understanding_span(take, positioned)
    assert span.positioned_evidence is positioned


def test_span_attempt_relation_fields_do_not_exist_on_dataclass():
    # D-145's Attempt Relationship authority is untouched -- this module
    # supplies evidence only, never a relationship verdict.
    take = _take("clip-1")
    span = build_raw_understanding_span(take, _positioned_for(take, None))
    assert not hasattr(span, "attempt_relation")
    assert not hasattr(span, "proposition_relation")


# ---------------------------------------------------------------------------
# 5-13: behavior hypothesis vocabulary + provenance correctness.
# ---------------------------------------------------------------------------

def test_behavior_hypotheses_bounded_to_allowed_vocabulary_no_event_case():
    take = _take("clip-1")
    span = build_raw_understanding_span(take, _positioned_for(take, None))
    labels = {h.label for h in span.behavior_hypotheses}
    assert labels <= ALLOWED_BEHAVIOR_HYPOTHESES
    assert not (labels & FORBIDDEN_RELATIONSHIP_LABELS)


def test_no_event_span_yields_clean_attempt_with_fusion_provenance():
    take = _take("clip-1")
    span = build_raw_understanding_span(take, _positioned_for(take, None))
    clean = [h for h in span.behavior_hypotheses if h.label == BEHAVIOR_CLEAN_ATTEMPT]
    assert len(clean) == 1
    assert clean[0].provenance == PROVENANCE_MULTIMODAL_FUSION


def test_reset_family_kind_maps_to_post_take_reset_visual_provenance():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    context = _context("src-1", (TemporalEvent("src-1", 1.9, 2.1, "camera_disengagement_candidate", 0.8, ""),))
    span = build_raw_understanding_span(take, _positioned_for(take, context))
    reset = [h for h in span.behavior_hypotheses if h.label == BEHAVIOR_POST_TAKE_RESET]
    assert len(reset) == 1
    assert reset[0].provenance == PROVENANCE_VISUAL_SIGNAL
    assert reset[0].confidence == 0.8


def test_wrong_take_kind_maps_to_abandoned_attempt_deterministic_rule():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    context = _context("src-1", (TemporalEvent("src-1", 0.5, 1.5, "wrong_take", 0.97, ""),))
    span = build_raw_understanding_span(take, _positioned_for(take, context))
    abandoned = [h for h in span.behavior_hypotheses if h.label == BEHAVIOR_ABANDONED_ATTEMPT]
    assert len(abandoned) == 1
    assert abandoned[0].provenance == PROVENANCE_DETERMINISTIC_RULE
    assert abandoned[0].confidence == 0.97


def test_retry_setup_kind_also_maps_to_abandoned_attempt():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    context = _context("src-1", (TemporalEvent("src-1", 0.5, 1.5, "retry_setup", 0.6, ""),))
    span = build_raw_understanding_span(take, _positioned_for(take, context))
    labels = {h.label for h in span.behavior_hypotheses}
    assert BEHAVIOR_ABANDONED_ATTEMPT in labels


def test_recording_process_kind_maps_to_recording_process_deterministic_rule():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    context = _context("src-1", (TemporalEvent("src-1", 0.5, 1.5, "verbal_fumble", 0.7, ""),))
    span = build_raw_understanding_span(take, _positioned_for(take, context))
    recording = [h for h in span.behavior_hypotheses if h.label == BEHAVIOR_RECORDING_PROCESS]
    assert len(recording) == 1
    assert recording[0].provenance == PROVENANCE_DETERMINISTIC_RULE


def test_false_start_and_breaking_character_kinds_map_correctly_when_present():
    # Never emitted by any live producer today (see module docstring), but
    # the allowed vocabulary + deriver must still handle them correctly if a
    # future producer emits them -- schema completeness, honestly labelled.
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    fs_context = _context("src-1", (TemporalEvent("src-1", 0.0, 0.3, "false_start", 0.5, ""),))
    fs_span = build_raw_understanding_span(take, _positioned_for(take, fs_context))
    assert BEHAVIOR_FALSE_START in {h.label for h in fs_span.behavior_hypotheses}

    bc_context = _context("src-1", (TemporalEvent("src-1", 0.0, 0.3, "breaking_character", 0.5, ""),))
    bc_span = build_raw_understanding_span(take, _positioned_for(take, bc_context))
    assert BEHAVIOR_BREAKING_CHARACTER in {h.label for h in bc_span.behavior_hypotheses}


def test_audience_delivery_added_when_delivery_available_and_no_break_event():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    positioned = _positioned_for(take, None)
    assert positioned.delivery_span.available  # sanity: word timings span the take
    span = build_raw_understanding_span(take, positioned)
    assert BEHAVIOR_AUDIENCE_DELIVERY in {h.label for h in span.behavior_hypotheses}


def test_audience_delivery_absent_when_delivery_zone_has_break_event():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    # Event inside the DELIVERY zone (middle of the take) should suppress
    # the "no break overlaps delivery" AUDIENCE_DELIVERY inference.
    context = _context("src-1", (TemporalEvent("src-1", 0.9, 1.1, "body_reset_candidate", 0.6, ""),))
    positioned = _positioned_for(take, context)
    delivery_kinds = {e.kind for e in positioned.positioned_events if e.zone == ZONE_DELIVERY}
    if delivery_kinds:
        span = build_raw_understanding_span(take, positioned)
        assert BEHAVIOR_AUDIENCE_DELIVERY not in {h.label for h in span.behavior_hypotheses}


def test_evidence_provenance_map_tags_transcript_and_word_timings_as_asr():
    take = _take("clip-1")
    span = build_raw_understanding_span(take, _positioned_for(take, None))
    assert span.evidence_provenance["transcript"] == PROVENANCE_ASR
    assert span.evidence_provenance["word_timings"] == PROVENANCE_ASR


# ---------------------------------------------------------------------------
# 14-24: per-source map assembly, media facts, track-status/overall status.
# ---------------------------------------------------------------------------

def test_map_preserves_source_id_and_duration():
    take = _take("clip-1", source_asset_id="src-1")
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=42.5,
        transcript_segments=(), whole_context=None, takes_for_source=(take,),
        media_probe=None, speech_track_status=TRACK_STATUS_PASS,
        audio_track_status=TRACK_STATUS_PASS, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    assert m.source_asset_id == "src-1"
    assert m.source_duration == 42.5


def test_map_joins_transcript_and_concatenates_word_timings_from_segments():
    seg1 = TranscriptSegment(source_asset_id="src-1", start=0.0, end=1.0, text="hello", words=(_word("hello", 0.0, 1.0),))
    seg2 = TranscriptSegment(source_asset_id="src-1", start=1.0, end=2.0, text="world", words=(_word("world", 1.0, 2.0),))
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=2.0,
        transcript_segments=(seg1, seg2), whole_context=None, takes_for_source=(),
        media_probe=None, speech_track_status=TRACK_STATUS_PASS,
        audio_track_status=TRACK_STATUS_PASS, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    assert m.transcript == "hello world"
    assert len(m.word_timings) == 2
    assert m.speech_spans == (seg1, seg2)


def test_map_splits_audio_vs_visual_events_by_kind():
    context = _context("src-1", (
        TemporalEvent("src-1", 1.0, 2.0, "audio_silence_interval", 0.9, ""),
        TemporalEvent("src-1", 3.0, 3.2, "camera_disengagement_candidate", 0.7, ""),
    ))
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=5.0,
        transcript_segments=(), whole_context=context, takes_for_source=(),
        media_probe=None, speech_track_status=TRACK_STATUS_PASS,
        audio_track_status=TRACK_STATUS_PASS, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    assert len(m.audio_events) == 1 and m.audio_events[0].kind == "audio_silence_interval"
    assert len(m.visual_performance_events) == 1 and m.visual_performance_events[0].kind == "camera_disengagement_candidate"


def test_map_media_facts_populated_from_probe_and_empty_when_none():
    probe = MediaProbe(duration_sec=10.0, width=1920, height=1080, fps=30.0, has_audio=True)
    m_with = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=10.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=probe,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert m_with.media_facts == {"duration_sec": 10.0, "width": 1920, "height": 1080, "fps": 30.0, "has_audio": True}
    assert m_with.evidence_provenance["media_facts"] == PROVENANCE_MEDIA_TIMING

    m_without = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=10.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert m_without.media_facts == {}


def test_overall_status_complete_when_all_tracks_pass():
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=1.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert m.track_status["raw_understanding_map_status"] == MAP_STATUS_COMPLETE_EXISTING_EVIDENCE


def test_overall_status_partial_when_one_track_not_available():
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=1.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_NOT_AVAILABLE, media_track_status=TRACK_STATUS_PASS,
    )
    assert m.track_status["raw_understanding_map_status"] == MAP_STATUS_PARTIAL_EXISTING_EVIDENCE


def test_overall_status_failed_when_media_track_failed():
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=1.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_FAILED,
    )
    assert m.track_status["raw_understanding_map_status"] == MAP_STATUS_FAILED


def test_overall_status_failed_when_speech_track_failed():
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=1.0, transcript_segments=(),
        whole_context=None, takes_for_source=(), media_probe=None,
        speech_track_status=TRACK_STATUS_FAILED, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert m.track_status["raw_understanding_map_status"] == MAP_STATUS_FAILED


def test_map_never_recomputes_positioned_evidence_reuses_d115_builder_output():
    take = _take("clip-1", source_asset_id="src-1")
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=2.0, transcript_segments=(),
        whole_context=None, takes_for_source=(take,), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    expected = build_positioned_performance_evidence_for_takes(
        (take,), None, event_kinds=_BEHAVIOR_RELEVANT_EVENT_KINDS,
    )
    assert m.positioned_performance_evidence == expected


def test_map_filters_takes_by_source_asset_id():
    take_a = _take("clip-a", source_asset_id="src-1")
    take_b = _take("clip-b", source_asset_id="src-2")
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=2.0, transcript_segments=(),
        whole_context=None, takes_for_source=(take_a, take_b), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert [span.span_id for span in m.candidate_span_evidence] == ["clip-a"]


# ---------------------------------------------------------------------------
# 25-29: batch ordering, non-destructiveness, schema/version, tail-safety.
# ---------------------------------------------------------------------------

def test_batch_build_preserves_sources_order_regardless_of_segment_order():
    take1 = _take("clip-1", source_asset_id="src-1")
    take2 = _take("clip-2", source_asset_id="src-2")
    sources = [
        type("S", (), {"source_asset_id": "src-2", "duration_sec": 5.0})(),
        type("S", (), {"source_asset_id": "src-1", "duration_sec": 3.0})(),
    ]
    seg1 = TranscriptSegment(source_asset_id="src-1", start=0.0, end=1.0, text="a")
    seg2 = TranscriptSegment(source_asset_id="src-2", start=0.0, end=1.0, text="b")
    maps = build_raw_understanding_maps_for_sources(
        sources=sources, transcript_tuple=(seg1, seg2), whole_context=None,
        takes=(take1, take2), media_probes_by_source={},
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    assert [m.source_asset_id for m in maps] == ["src-2", "src-1"]


def test_map_does_not_mutate_input_takes_or_context():
    take = _take("clip-1", source_asset_id="src-1")
    context = _context("src-1", (TemporalEvent("src-1", 0.0, 0.2, "wrong_take", 0.5, ""),))
    context_before = context
    build_raw_understanding_map(
        source_asset_id="src-1", source_duration=2.0, transcript_segments=(),
        whole_context=context, takes_for_source=(take,), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    # Frozen dataclasses cannot be mutated in place; this asserts identity/
    # equality survive the call untouched (non-destructive by construction).
    assert context is context_before
    assert context.sources[0].events[0].kind == "wrong_take"


def test_schema_version_is_stable_and_namespaced():
    assert SCHEMA_VERSION.startswith("cutsell.raw_understanding_map.")


def test_diagnostics_are_tail_safe_no_transcript_or_event_payload():
    take = _take("clip-1", source_asset_id="src-1", text="secret transcript text")
    m = build_raw_understanding_map(
        source_asset_id="src-1", source_duration=2.0, transcript_segments=(),
        whole_context=None, takes_for_source=(take,), media_probe=None,
        speech_track_status=TRACK_STATUS_PASS, audio_track_status=TRACK_STATUS_PASS,
        visual_track_status=TRACK_STATUS_PASS, media_track_status=TRACK_STATUS_PASS,
    )
    diag = raw_understanding_map_diagnostics((m,))
    serialized = repr(diag)
    assert "secret transcript text" not in serialized
    assert diag["source_count"] == 1
    assert diag["raw_understanding_map_created"] is True


def test_diagnostics_status_aggregation_all_complete_any_failed_else_partial():
    complete_map = build_raw_understanding_map(
        source_asset_id="s1", source_duration=1.0, transcript_segments=(), whole_context=None,
        takes_for_source=(), media_probe=None, speech_track_status=TRACK_STATUS_PASS,
        audio_track_status=TRACK_STATUS_PASS, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    partial_map = build_raw_understanding_map(
        source_asset_id="s2", source_duration=1.0, transcript_segments=(), whole_context=None,
        takes_for_source=(), media_probe=None, speech_track_status=TRACK_STATUS_PASS,
        audio_track_status=TRACK_STATUS_NOT_AVAILABLE, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    failed_map = build_raw_understanding_map(
        source_asset_id="s3", source_duration=1.0, transcript_segments=(), whole_context=None,
        takes_for_source=(), media_probe=None, speech_track_status=TRACK_STATUS_FAILED,
        audio_track_status=TRACK_STATUS_PASS, visual_track_status=TRACK_STATUS_PASS,
        media_track_status=TRACK_STATUS_PASS,
    )
    assert raw_understanding_map_diagnostics((complete_map,))["raw_understanding_map_status"] == MAP_STATUS_COMPLETE_EXISTING_EVIDENCE
    assert raw_understanding_map_diagnostics((complete_map, partial_map))["raw_understanding_map_status"] == MAP_STATUS_PARTIAL_EXISTING_EVIDENCE
    assert raw_understanding_map_diagnostics((complete_map, failed_map))["raw_understanding_map_status"] == MAP_STATUS_FAILED
    assert raw_understanding_map_diagnostics(())["raw_understanding_map_status"] == MAP_STATUS_FAILED
    assert raw_understanding_map_diagnostics(())["source_count"] == 0


def test_no_forbidden_relationship_label_ever_emitted_across_event_combinations():
    take = _take("clip-1", source_asset_id="src-1", start=0.0, end=2.0)
    kind_combos = [
        (),
        ("wrong_take",),
        ("retry_setup",),
        ("camera_disengagement_candidate",),
        ("facial_expression_shift_candidate",),
        ("body_reset_candidate",),
        ("hand_motion_reset_candidate",),
        ("verbal_fumble",),
        ("false_start",),
        ("breaking_character",),
        ("wrong_take", "camera_disengagement_candidate"),
    ]
    for kinds in kind_combos:
        events = tuple(TemporalEvent("src-1", 0.1, 0.3, kind, 0.5, "") for kind in kinds)
        context = _context("src-1", events) if events else None
        span = build_raw_understanding_span(take, _positioned_for(take, context))
        labels = {h.label for h in span.behavior_hypotheses}
        assert not (labels & FORBIDDEN_RELATIONSHIP_LABELS), f"forbidden label leaked for kinds={kinds}"
        assert labels <= ALLOWED_BEHAVIOR_HYPOTHESES


def test_module_never_imports_a_downstream_authority():
    # Structural authority-boundary guarantee (D-154's own boundary,
    # restated as a test): this module must not import BestTake/Boundary/
    # Renderer/the D-150 semantic-authority gate, so it cannot accidentally
    # become a second decision authority.
    import cutsell_worker.raw_understanding_map as mod
    source = open(mod.__file__, encoding="utf-8").read()
    forbidden_imports = [
        "deterministic_best_take_authority",
        "semantic_authority_observability",
        "boundary_engine",
        "renderer",
        "composite_resolver",
    ]
    for forbidden in forbidden_imports:
        assert f"import {forbidden}" not in source and f"from .{forbidden}" not in source
