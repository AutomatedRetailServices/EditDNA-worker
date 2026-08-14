from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.recording_suffix_trim import trim_visual_self_critique_suffixes
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(text, words):
    return CandidateTake(
        clip_id="clip",
        source_asset_id="src",
        source_order=0,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=tuple(words),
    )


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="talking_head",
            creator_intent="recording",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def event(kind, start, end, confidence=0.9):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def words_for(tokens, start=0.0, step=0.4):
    return [Word(token, start + index * step, start + (index + 1) * step) for index, token in enumerate(tokens)]


def test_trims_self_critique_suffix_only_with_multimodal_physical_break():
    tokens = ["Go", "out", "on", "a", "comfortable", "run", "or", "something", "That's", "stupid"]
    item = take("Go out on a comfortable run or something. That's stupid.", words_for(tokens))
    ctx = context((
        event("facial_expression_shift_candidate", 3.15, 3.25, 0.80),
        event("hand_motion_reset_candidate", 3.20, 3.30, 0.95),
    ))

    refined, diagnostics = trim_visual_self_critique_suffixes((item,), ctx)

    assert len(refined) == 1
    assert refined[0].text == "Go out on a comfortable run or something"
    assert refined[0].end == item.words[7].end
    assert diagnostics[0]["action"] == "trim_self_critique_suffix"


def test_does_not_trim_self_critique_suffix_without_two_visual_families():
    tokens = ["Go", "out", "on", "a", "comfortable", "run", "or", "something", "That's", "stupid"]
    item = take("Go out on a comfortable run or something. That's stupid.", words_for(tokens))
    ctx = context((event("hand_motion_reset_candidate", 3.20, 3.30, 0.95),))

    refined, diagnostics = trim_visual_self_critique_suffixes((item,), ctx)

    assert refined == (item,)
    assert diagnostics == ()


def test_standalone_opinion_is_not_trimmed_without_useful_prefix():
    tokens = ["That's", "stupid"]
    item = take("That's stupid.", words_for(tokens))
    ctx = context((
        event("facial_expression_shift_candidate", 0.10, 0.20, 0.90),
        event("body_reset_candidate", 0.15, 0.25, 0.90),
    ))

    refined, diagnostics = trim_visual_self_critique_suffixes((item,), ctx)

    assert refined == (item,)
    assert diagnostics == ()
