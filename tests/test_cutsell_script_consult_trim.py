from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.script_consult_trim import trim_script_consult_pauses
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take():
    words = (
        Word("this", 10.0, 10.3),
        Word("product", 10.4, 10.8),
        Word("works", 10.9, 11.3),
        Word("because", 11.4, 11.8),
        Word("it", 13.2, 13.4),
        Word("helps", 13.5, 13.9),
        Word("with", 14.0, 14.2),
        Word("sleep", 14.3, 14.7),
    )
    return CandidateTake(
        clip_id="take-1",
        source_asset_id="src",
        source_order=0,
        start=9.9,
        end=15.0,
        text="this product works because it helps with sleep",
        words=words,
        signals=MediaSignals("src", 9.9, 15.0),
        complete_idea=True,
    )


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator explains the product and consults notes before continuing",
            dominant_style="talking_head",
            creator_intent="deliver scripted product explanation",
            events=tuple(events),
            edit_mode="sales",
            sales_intent=0.9,
            main_topic="product explanation",
            product_or_subject="product",
            story_logic="one continuous scripted idea with recording retries removed",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_splits_visible_note_consult_gap_and_preserves_spoken_words():
    take = _take()
    context = _context(
        TemporalEvent("src", 11.9, 12.8, "camera_disengagement_candidate", 0.91, "looks down away from lens"),
        TemporalEvent("src", 12.0, 12.9, "body_reset_candidate", 0.93, "resets posture while reading"),
    )
    refined, diagnostics = trim_script_consult_pauses((take,), context)

    assert len(refined) == 2
    assert refined[0].text == "this product works because"
    assert refined[0].end == 11.8
    assert refined[1].text == "it helps with sleep"
    assert refined[1].start == 13.2
    assert diagnostics[0]["action"] == "split_script_consult_pause"
    assert diagnostics[0]["removed_gaps"][0]["duration_sec"] == 1.4


def test_camera_look_away_without_reset_fails_open():
    take = _take()
    context = _context(
        TemporalEvent("src", 11.9, 12.8, "camera_disengagement_candidate", 0.95, "brief natural look away"),
    )
    refined, diagnostics = trim_script_consult_pauses((take,), context)
    assert refined == (take,)
    assert diagnostics == ()


def test_authoritative_word_search_event_can_split_without_dense_candidate_pair():
    take = _take()
    context = _context(
        TemporalEvent("src", 11.9, 13.0, "searching_for_words", 0.88, "creator checks script before resuming"),
    )
    refined, diagnostics = trim_script_consult_pauses((take,), context)
    assert len(refined) == 2
    assert diagnostics[0]["removed_gaps"][0]["evidence"][0].startswith("event:searching_for_words")


def test_short_normal_pause_is_never_split():
    take = _take()
    shifted_words = tuple(
        Word(word.text, word.start if index < 4 else word.start - 0.9, word.end if index < 4 else word.end - 0.9)
        for index, word in enumerate(take.words)
    )
    normal = CandidateTake(
        clip_id=take.clip_id,
        source_asset_id=take.source_asset_id,
        source_order=take.source_order,
        start=take.start,
        end=take.end - 0.9,
        text=take.text,
        words=shifted_words,
        signals=MediaSignals("src", take.start, take.end - 0.9),
        complete_idea=True,
    )
    context = _context(
        TemporalEvent("src", 11.9, 12.2, "camera_disengagement_candidate", 0.95, "small glance"),
        TemporalEvent("src", 11.9, 12.2, "body_reset_candidate", 0.95, "small posture change"),
    )
    refined, diagnostics = trim_script_consult_pauses((normal,), context)
    assert refined == (normal,)
    assert diagnostics == ()
