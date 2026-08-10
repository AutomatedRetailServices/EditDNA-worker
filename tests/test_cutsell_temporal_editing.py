from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.temporal_editing import refine_takes_with_temporal_context
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(start=10.0, end=15.0, text="this works really well"):
    words = (
        Word("this", 10.0, 10.4),
        Word("works", 10.5, 11.0),
        Word("really", 11.1, 11.6),
        Word("well", 11.7, 12.1),
    )
    return CandidateTake(
        clip_id="take-1",
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        signals=MediaSignals("src-1", start, end),
        complete_idea=True,
    )


def _context(*events, mode="sales"):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src-1",
            summary="creator records a product explanation",
            dominant_style="talking_head",
            creator_intent="explain clearly",
            events=tuple(events),
            edit_mode=mode,
            sales_intent=0.9 if mode == "sales" else 0.0,
            main_topic="product explanation" if mode == "sales" else "personal story",
            product_or_subject="product" if mode == "sales" else "story",
            story_logic="good line followed by recording reset",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_trims_bad_reaction_after_good_line_without_deleting_line():
    take = _take()
    context = _context(TemporalEvent(
        "src-1", 12.2, 15.0, "body_reset", 0.96, "looks away and resets to retry"
    ))
    refined, diagnostics = refine_takes_with_temporal_context((take,), context)
    assert len(refined) == 1
    assert refined[0].start == 10.0
    assert refined[0].end == 12.2
    assert refined[0].text == "this works really well"
    assert diagnostics[0]["applied"][0]["action"] == "trim_end"
    assert diagnostics[0]["applied"][0]["kind"] == "body_reset"


def test_meaningful_pause_is_not_trimmed_as_dead_air():
    take = _take()
    context = _context(TemporalEvent(
        "src-1", 12.2, 15.0, "meaningful_pause", 0.99, "intentional reveal beat"
    ))
    refined, diagnostics = refine_takes_with_temporal_context((take,), context)
    assert refined[0].start == take.start
    assert refined[0].end == take.end
    assert diagnostics[0]["applied"] == []


def test_whole_video_wrong_take_can_override_complete_transcript():
    take = _take()
    context = _context(TemporalEvent(
        "src-1", 10.0, 14.0, "wrong_take", 0.97, "creator visibly knows the attempt failed"
    ))
    decision = evaluate_take(take, context)
    assert decision.keep is False
    assert decision.reason == "whole_video_bad_take:wrong_take"


def test_natural_mode_uses_same_performance_cleanup_without_sales_requirement():
    take = _take(text="and then I missed my flight")
    context = _context(TemporalEvent(
        "src-1", 14.0, 15.0, "frustration", 0.92, "recording frustration after line"
    ), mode="natural")
    refined, _ = refine_takes_with_temporal_context((take,), context)
    assert refined[0].end == 14.0
    assert context.dominant_edit_mode == "natural"
