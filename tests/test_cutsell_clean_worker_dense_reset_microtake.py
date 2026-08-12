from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _context(*events):
    source = SourceVideoContext(
        source_asset_id="src",
        summary="",
        dominant_style="talking_head",
        creator_intent="",
        edit_mode="natural",
        sales_intent=0.0,
        main_topic="",
        product_or_subject="",
        story_logic="",
        events=tuple(events),
    )
    return WholeVideoContext(
        sources=(source,),
        status=ProviderStatus("local", True, True, "applied"),
    )


def _reset(start, end, confidence=0.98):
    return TemporalEvent(
        source_asset_id="src",
        start=start,
        end=end,
        kind="hand_motion_reset_candidate",
        confidence=confidence,
        description="dense hand reset",
    )


def test_incomplete_microtake_with_dense_resets_is_discarded():
    take = CandidateTake(
        "frag", "src", 0, 149.2, 151.58, "I people it was", complete_idea=False
    )
    context = _context(
        _reset(149.30, 149.38),
        _reset(149.51, 149.59),
        _reset(150.31, 150.39),
    )
    decision = evaluate_take(take, context)
    assert decision.keep is False
    assert decision.reason == "incomplete_microtake_dense_reset"


def test_incomplete_microtake_with_single_reset_is_preserved():
    take = CandidateTake(
        "frag", "src", 0, 10.0, 12.0, "Maybe I", complete_idea=False
    )
    context = _context(_reset(10.5, 10.6))
    decision = evaluate_take(take, context)
    assert decision.keep is True


def test_complete_short_line_with_dense_resets_is_preserved():
    take = CandidateTake(
        "line", "src", 0, 20.0, 22.2, "That was wild!", complete_idea=True
    )
    context = _context(
        _reset(20.1, 20.2),
        _reset(20.8, 20.9),
        _reset(21.4, 21.5),
    )
    decision = evaluate_take(take, context)
    assert decision.keep is True
