from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(text, start=10.0, end=12.0):
    return CandidateTake(
        clip_id="clip",
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def event(kind, start, end, confidence=0.9):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="creator_raw",
            creator_intent="recording_clean_cut",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_first_person_script_check_is_explicit_recording_meta():
    decision = evaluate_take(take("I need to look at the script"), context(()))

    assert decision.keep is False
    assert decision.reason == "explicit_recording_meta"
    assert decision.confidence == 0.96


def test_asr_dropped_subject_requires_multimodal_corroboration():
    candidate = take("Have to look at the word", 13.9, 15.7)
    ctx = context((
        event("body_reset_candidate", 14.1, 14.2, 1.0),
        event("facial_expression_shift_candidate", 14.16, 14.22, 0.78),
    ))

    decision = evaluate_take(candidate, ctx)

    assert decision.keep is False
    assert decision.reason == "explicit_recording_meta"


def test_asr_dropped_subject_survives_without_visual_corroboration():
    decision = evaluate_take(take("Have to look at the word"), context(()))

    assert decision.keep is True
    assert decision.reason == "valid_or_uncertain_speech"


def test_legitimate_word_reference_is_not_recording_meta():
    decision = evaluate_take(take("Look at the word SALE on the bottle"), context((
        event("body_reset_candidate", 10.5, 10.6, 1.0),
        event("facial_expression_shift_candidate", 10.7, 10.8, 0.9),
    )))

    assert decision.keep is True
