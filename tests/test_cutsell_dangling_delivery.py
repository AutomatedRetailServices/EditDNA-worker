from cutsell_worker.contracts import CandidateTake
from cutsell_worker.dangling_delivery import apply_dangling_delivery_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(text, *, start=0.0, end=4.0, clip_id="a"):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def _context(*events):
    return WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id="src",
                summary="",
                dominant_style="talking_head",
                creator_intent="explain",
                events=tuple(events),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _event(kind, start=3.6, end=3.9, confidence=0.95):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def test_dangling_article_plus_strong_reset_is_removed():
    take = _take("Look at the detail on this shirt and oh my god I don't have a")
    kept, removed, diagnostics = apply_dangling_delivery_cleanup(
        (take,),
        _context(_event("body_reset_candidate")),
    )
    assert kept == ()
    assert removed == (take,)
    assert diagnostics[0]["reason"] == "dangling_open_ending_with_physical_reset"


def test_dangling_article_without_reset_is_preserved():
    take = _take("I was going to tell you about a")
    kept, removed, diagnostics = apply_dangling_delivery_cleanup((take,), _context())
    assert kept == (take,)
    assert removed == ()
    assert diagnostics == ()


def test_valid_complete_phrase_ending_are_is_preserved():
    take = _take("I love them exactly for who they are")
    kept, removed, _ = apply_dangling_delivery_cleanup(
        (take,),
        _context(_event("body_reset_candidate")),
    )
    assert kept == (take,)
    assert removed == ()


def test_ambiguous_auxiliary_requires_restart_churn_and_reset():
    take = _take(
        "You got a perfectly perfect for anybody for christmas perfect what if I told you your kids are",
        end=8.0,
    )
    context = _context(_event("hand_motion_reset_candidate", start=7.55, end=7.9, confidence=0.99))
    kept, removed, diagnostics = apply_dangling_delivery_cleanup((take,), context)
    assert kept == ()
    assert removed == (take,)
    assert diagnostics[0]["reason"] == "dangling_auxiliary_with_internal_restart_and_reset"


def test_repeated_emphasis_without_dangling_auxiliary_is_preserved():
    take = _take("They are so so super cute and perfect for this outfit")
    kept, removed, _ = apply_dangling_delivery_cleanup(
        (take,),
        _context(_event("hand_motion_reset_candidate")),
    )
    assert kept == (take,)
    assert removed == ()
