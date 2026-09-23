"""A real source pause can expose two attempts hidden in one ASR segment."""
from __future__ import annotations

from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


SRC = "src_internal_pause"


def _context(start: float, end: float, *, confidence: float = 1.0) -> WholeVideoContext:
    event = TemporalEvent(
        source_asset_id=SRC,
        start=start,
        end=end,
        kind="audio_silence_interval",
        confidence=confidence,
        description="measured source silence",
    )
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id=SRC,
            summary="",
            dominant_style="",
            creator_intent="",
            events=(event,),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _fused_take() -> CandidateTake:
    tokens = (
        ("They", 10.0, 10.3), ("tested", 10.3, 10.8), ("me", 10.8, 11.0),
        ("and", 11.0, 11.2), ("diagnosed", 11.2, 11.8), ("me", 11.8, 12.0),
        ("with...", 12.0, 12.4),
        ("I", 14.8, 14.9), ("had", 14.9, 15.1), ("stomach", 15.1, 15.5),
        ("problems", 15.5, 16.0), ("in", 16.0, 16.1), ("2023.", 16.1, 16.6),
    )
    words = tuple(Word(text, start, end) for text, start, end in tokens)
    return CandidateTake(
        clip_id="fused",
        source_asset_id=SRC,
        source_order=0,
        start=10.0,
        end=16.6,
        text="They tested me and diagnosed me with... I had stomach problems in 2023.",
        words=words,
        complete_idea=True,
        source_span_id="span_parent",
        word_indices=tuple(range(len(words))),
    )


def test_long_internal_measured_pause_splits_one_asr_candidate_at_word_boundary():
    attempts, diagnostics = reconstruct_delivery_attempts(
        (_fused_take(),), _context(12.5, 14.7),
    )

    assert [attempt.text for attempt in attempts] == [
        "They tested me and diagnosed me with...",
        "I had stomach problems in 2023.",
    ]
    assert attempts[0].complete_idea is False
    assert attempts[1].complete_idea is True
    assert attempts[0].word_indices == tuple(range(7))
    assert attempts[1].word_indices == tuple(range(7, 13))
    assert diagnostics["input_take_count"] == 1
    assert diagnostics["expanded_take_count"] == 2
    assert diagnostics["internal_measured_pause_splits"][0]["decision"] == "split_hidden_attempt_boundary"


def test_short_or_low_confidence_internal_silence_does_not_split():
    for context in (_context(12.5, 13.2), _context(12.5, 14.7, confidence=0.6)):
        attempts, diagnostics = reconstruct_delivery_attempts((_fused_take(),), context)
        assert len(attempts) == 1
        assert diagnostics["internal_measured_pause_splits"] == []


def test_internal_silence_without_word_timestamps_fails_open():
    take = CandidateTake(
        clip_id="no_words",
        source_asset_id=SRC,
        source_order=0,
        start=10.0,
        end=16.6,
        text="One delivery. Another delivery.",
    )
    attempts, diagnostics = reconstruct_delivery_attempts((take,), _context(12.5, 14.7))
    assert len(attempts) == 1
    assert attempts[0].clip_id == take.clip_id
    assert attempts[0].text == take.text
    assert diagnostics["internal_measured_pause_splits"] == []
