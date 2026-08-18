from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.delivery_edge_trim import trim_delivery_edge_slack
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator delivers a clean sentence and visibly resets after it",
            dominant_style="talking_head",
            creator_intent="record a clean take",
            events=tuple(events),
            edit_mode="natural",
            sales_intent=0.0,
            main_topic="story",
            product_or_subject="subject",
            story_logic="remove pre-roll and post-roll recording beats",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _take(start=0.0, end=5.0):
    words = (
        Word("this", 0.6, 1.0),
        Word("works", 1.1, 1.6),
        Word("really", 1.7, 2.2),
        Word("well", 2.3, 2.7),
    )
    return CandidateTake(
        clip_id="take",
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text="this works really well",
        words=words,
        signals=MediaSignals("src", start, end),
    )


def test_trims_trailing_non_speech_after_visible_cut_signal():
    take = _take()
    context = _context(TemporalEvent(
        "src", 3.1, 3.2, "body_reset_candidate", 0.98,
        "creator relaxes and resets after finishing the line",
    ))
    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)
    assert trimmed[0].end == 2.7
    assert trimmed[0].text == take.text
    assert diagnostics[0]["actions"][-1]["action"] == "trim_trailing_non_speech_cut_signal"


def test_trims_leading_setup_before_first_word_when_reset_is_visible():
    take = _take()
    context = _context(TemporalEvent(
        "src", 0.1, 0.2, "body_reset_candidate", 0.99,
        "creator settles before starting",
    ))
    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)
    assert trimmed[0].start == 0.6
    assert diagnostics[0]["actions"][0]["action"] == "trim_leading_non_speech_setup"


def test_non_speech_slack_without_cut_signal_fails_open():
    take = _take()
    trimmed, diagnostics = trim_delivery_edge_slack((take,), _context())
    assert trimmed == (take,)
    assert diagnostics == ()


def test_short_natural_breathing_margin_is_preserved():
    words = (
        Word("this", 0.2, 0.5),
        Word("works", 0.6, 1.0),
    )
    take = CandidateTake(
        clip_id="short-margin",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=1.2,
        text="this works",
        words=words,
        signals=MediaSignals("src", 0.0, 1.2),
    )
    context = _context(TemporalEvent(
        "src", 0.0, 0.2, "body_reset_candidate", 1.0, "small initial movement",
    ))
    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)
    assert trimmed == (take,)
    assert diagnostics == ()


def test_talking_head_short_mic_drop_slack_is_trimmed_with_hand_reset_cluster():
    words = (
        Word("this", 0.0, 0.4),
        Word("works", 0.5, 1.0),
    )
    take = CandidateTake(
        clip_id="mic-drop",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=1.18,
        text="this works",
        words=words,
        signals=MediaSignals("src", 0.0, 1.18),
    )
    context = _context(
        TemporalEvent("src", 1.02, 1.08, "hand_motion_reset_candidate", 0.97, "mic starts dropping"),
        TemporalEvent("src", 1.09, 1.15, "hand_motion_reset_candidate", 0.99, "hand returns to rest"),
    )
    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)
    assert trimmed[0].end == 1.0
    assert diagnostics[0]["actions"][0]["talking_head_micro_edge"] is True


def test_single_short_hand_gesture_does_not_force_frame_tight_cut():
    words = (
        Word("this", 0.0, 0.4),
        Word("works", 0.5, 1.0),
    )
    take = CandidateTake(
        clip_id="gesture",
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=1.18,
        text="this works",
        words=words,
        signals=MediaSignals("src", 0.0, 1.18),
    )
    context = _context(TemporalEvent(
        "src", 1.04, 1.11, "hand_motion_reset_candidate", 0.99, "single hand movement",
    ))
    trimmed, diagnostics = trim_delivery_edge_slack((take,), context)
    assert trimmed == (take,)
    assert diagnostics == ()
