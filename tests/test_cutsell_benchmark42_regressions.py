from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.session_boundaries import infer_session_boundaries
from cutsell_worker.story_coverage_guard import restore_unique_story_coverage
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        complete_idea=complete,
        signals=MediaSignals("src", float(start), float(end)),
    )


def test_session_boundary_exposes_legacy_at_alias_used_by_attempt_reconstruction():
    events = (
        TemporalEvent("src", 2.00, 2.08, "camera_disengagement_candidate", 0.92, "camera"),
        TemporalEvent("src", 2.02, 2.10, "facial_expression_shift_candidate", 0.90, "face"),
        TemporalEvent("src", 2.01, 2.09, "body_reset_candidate", 0.96, "reset"),
    )
    context = WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator compilation",
            dominant_style="talking_head",
            creator_intent="recording",
            events=events,
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )

    boundary = infer_session_boundaries(context, "src")[0]
    assert boundary.at == boundary.timestamp


def test_partial_fragments_do_not_block_restoring_long_coherent_story_delivery():
    partial_nutrients = _take(
        "partial-nutrients",
        11.10,
        14.38,
        "but these a day it's gonna help them get those nutrients",
        complete=False,
    )
    partial_worry = _take(
        "partial-worry",
        22.84,
        27.54,
        "if they're not eating healthy already worried that their kid is missing something",
        complete=False,
    )
    coherent_story = _take(
        "coherent-story",
        30.39,
        74.09,
        "A kid wakes up in the morning and has frozen waffles or their favorite cereal. "
        "At lunchtime they have chicken nuggets and french fries. At dinner they have mac and cheese "
        "and one piece of broccoli. That is hidden hunger: they are eating enough calories but can still "
        "miss important nutrients. A daily multivitamin helps fill those nutritional gaps and is non GMO, "
        "gluten free and vegan.",
    )

    kept, discarded, diagnostics = restore_unique_story_coverage(
        (partial_nutrients, partial_worry),
        (coherent_story,),
        (partial_nutrients, partial_worry, coherent_story),
        None,
    )

    assert {take.clip_id for take in kept} == {
        "partial-nutrients",
        "partial-worry",
        "coherent-story",
    }
    assert discarded == ()
    assert diagnostics[0]["clip_id"] == "coherent-story"
    assert diagnostics[0]["reason"] == "restore_unique_story_coverage"


def test_full_competing_retry_still_blocks_story_restore():
    first = _take(
        "first",
        10.0,
        20.0,
        "People ask me all the time if I actually have fun doing my job and the answer is yes because we have great chemistry",
    )
    fuller_retry = _take(
        "retry",
        22.0,
        35.0,
        "People ask me all the time if I actually have fun doing my job and the answer is yes because we have great chemistry on camera",
    )

    kept, discarded, diagnostics = restore_unique_story_coverage(
        (fuller_retry,),
        (first,),
        (first, fuller_retry),
        None,
    )

    assert kept == (fuller_retry,)
    assert discarded == (first,)
    assert diagnostics == ()
