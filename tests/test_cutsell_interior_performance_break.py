from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.interior_performance_break import split_interior_performance_breaks
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="talking head delivery with an interior physical reset",
            dominant_style="creator_raw",
            creator_intent="record a clean take",
            events=tuple(events),
            edit_mode="natural",
            sales_intent=0.0,
            main_topic="story",
            product_or_subject="subject",
            story_logic="preserve speech while removing recording resets",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _take():
    words = (
        Word("one", 0.0, 0.4),
        Word("two", 0.5, 0.9),
        Word("three", 1.0, 1.4),
        Word("four", 1.75, 2.1),
        Word("five", 2.2, 2.6),
        Word("six", 2.7, 3.1),
    )
    return CandidateTake(
        clip_id="take",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=3.1,
        text="one two three four five six",
        words=words,
        signals=MediaSignals("src", 0.0, 3.1),
    )


def test_splits_interior_word_gap_when_physical_and_face_break_are_strong():
    take = _take()
    context = _context(
        TemporalEvent("src", 1.42, 1.49, "hand_motion_reset_candidate", 0.98, "mic drops"),
        TemporalEvent("src", 1.52, 1.60, "hand_motion_reset_candidate", 0.96, "hand resets"),
        TemporalEvent("src", 1.48, 1.56, "facial_expression_shift_candidate", 0.80, "delivery breaks"),
    )
    split, diagnostics = split_interior_performance_breaks((take,), context)
    assert len(split) == 2
    assert split[0].end == 1.4
    assert split[1].start == 1.75
    assert split[0].text == "one two three"
    assert split[1].text == "four five six"
    assert diagnostics[0]["reason"] == "interior_multimodal_performance_break_split"


def test_does_not_split_on_hand_motion_without_independent_break_signal():
    take = _take()
    context = _context(
        TemporalEvent("src", 1.42, 1.49, "hand_motion_reset_candidate", 0.99, "gesture"),
        TemporalEvent("src", 1.52, 1.60, "hand_motion_reset_candidate", 0.98, "gesture"),
    )
    split, diagnostics = split_interior_performance_breaks((take,), context)
    assert split == (take,)
    assert diagnostics == ()


def test_does_not_split_when_no_safe_word_gap_exists():
    words = (
        Word("one", 0.0, 0.4),
        Word("two", 0.45, 0.9),
        Word("three", 0.95, 1.4),
        Word("four", 1.45, 1.9),
        Word("five", 1.95, 2.4),
    )
    take = CandidateTake(
        clip_id="continuous",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=2.4,
        text="one two three four five",
        words=words,
        signals=MediaSignals("src", 0.0, 2.4),
    )
    context = _context(
        TemporalEvent("src", 1.0, 1.1, "hand_motion_reset_candidate", 1.0, "reset"),
        TemporalEvent("src", 1.1, 1.2, "hand_motion_reset_candidate", 1.0, "reset"),
        TemporalEvent("src", 1.0, 1.2, "facial_expression_shift_candidate", 0.9, "break"),
    )
    split, diagnostics = split_interior_performance_breaks((take,), context)
    assert split == (take,)
    assert diagnostics == ()
