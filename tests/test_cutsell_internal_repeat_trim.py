from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.internal_repeat_trim import trim_internal_repeated_restarts
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _word(text: str, start: float, end: float) -> Word:
    return Word(text, start, end)


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator records a sales CTA and retries the phrase",
            dominant_style="talking_head",
            creator_intent="deliver one clean CTA",
            events=tuple(events),
            edit_mode="sales",
            sales_intent=1.0,
            main_topic="CTA",
            product_or_subject="product",
            story_logic="keep one clean CTA and remove repeated restart tails",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _repeated_take():
    texts = [
        "You", "see", "the", "orange", "shopping", "cart",
        "you", "might", "want", "to", "grab", "and", "if",
        "you", "see", "the", "orange", "shopping", "cart",
    ]
    words = []
    t = 43.78
    for text in texts:
        words.append(_word(text, t, t + 0.20))
        t += 0.23
    return CandidateTake(
        clip_id="cta",
        source_asset_id="src",
        source_order=0,
        start=43.78,
        end=t + 0.20,
        text=" ".join(texts),
        words=tuple(words),
        signals=MediaSignals("src", 43.78, t + 0.20),
    )


def test_trims_trailing_exact_restart_inside_one_take_with_visual_reset():
    take = _repeated_take()
    restart = take.words[11].start
    context = _context(TemporalEvent(
        "src", restart + 0.2, restart + 0.4,
        "body_reset_candidate", 0.98, "creator physically resets while restarting CTA",
    ))
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), context)

    assert len(trimmed) == 1
    assert trimmed[0].text == "You see the orange shopping cart you might want to grab"
    assert trimmed[0].end == take.words[10].end
    assert diagnostics[0]["reason"] == "internal_trailing_repeated_restart_trim"


def test_exact_rhetorical_repetition_without_recording_structure_fails_open():
    take = _repeated_take()
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take,), _context())
    assert trimmed == (take,)
    assert diagnostics == ()


def test_immediate_following_substantive_take_can_corroborate_restart_tail():
    take = _repeated_take()
    following = CandidateTake(
        clip_id="next",
        source_asset_id="src",
        source_order=0,
        start=take.end + 0.5,
        end=take.end + 3.0,
        text="but if you do not see it they probably sold out",
    )
    trimmed, diagnostics = trim_internal_repeated_restarts((take,), (take, following), _context())
    assert trimmed[0].text == "You see the orange shopping cart you might want to grab"
    assert diagnostics[0]["following_clip_id"] == "next"
