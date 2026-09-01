from cutsell_worker.clean_cut import apply_clean_cut, evaluate_take
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


def test_incomplete_microtake_with_dense_resets_is_detected():
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


def test_dense_reset_microtake_without_nearby_retry_is_preserved_in_batch():
    take = CandidateTake(
        "line", "src", 0, 20.0, 22.2, "It is amazing", complete_idea=False
    )
    unrelated = CandidateTake(
        "next", "src", 0, 23.0, 25.0, "What is happening", complete_idea=True
    )
    context = _context(
        _reset(20.1, 20.2),
        _reset(20.8, 20.9),
        _reset(21.4, 21.5),
    )
    kept, discarded, decisions = apply_clean_cut((take, unrelated), context)
    assert {item.clip_id for item in kept} == {"line", "next"}
    assert discarded == ()
    line_decision = next(item for item in decisions if item.clip_id == "line")
    assert line_decision.reason == "dense_reset_without_retry_corroboration"


def test_dense_reset_microtake_with_strong_nearby_retry_is_discarded_in_batch():
    fragment = CandidateTake(
        "frag", "src", 0, 20.0, 22.2, "And increases circulation", complete_idea=False
    )
    retry = CandidateTake(
        "retry", "src", 0, 24.0, 27.0, "Increases circulation which stimulates cell turnover", complete_idea=True
    )
    context = _context(
        _reset(20.1, 20.2),
        _reset(20.8, 20.9),
        _reset(21.4, 21.5),
    )
    kept, discarded, decisions = apply_clean_cut((fragment, retry), context)
    assert [item.clip_id for item in discarded] == ["frag"]
    assert [item.clip_id for item in kept] == ["retry"]
    frag_decision = next(item for item in decisions if item.clip_id == "frag")
    assert frag_decision.reason == "incomplete_microtake_dense_reset"
