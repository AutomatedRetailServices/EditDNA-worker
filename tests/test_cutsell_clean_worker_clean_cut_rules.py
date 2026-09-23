from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import CandidateTake


def take(text: str, *, duration: float = 2.0, clip_id: str = "clip") -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="source",
        source_order=0,
        start=0.0,
        end=duration,
        text=text,
    )


def test_discards_explicit_recording_directions_from_real_bloopers():
    cases = [
        "Damn it okay stop",
        "What am I saying that person that waits to get?",
        "Why is it so wobbly when I do that",
        "Okay, one more cuz you moved",
        "Let me redo that",
    ]
    for index, text in enumerate(cases):
        decision = evaluate_take(take(text, clip_id=f"c{index}"))
        assert decision.keep is False
        assert decision.reason == "explicit_restart_direction"


def test_discards_short_isolated_again_but_not_real_sentence_beginning_with_again():
    isolated = evaluate_take(take("Again", duration=1.3))
    sentence = evaluate_take(take("Again, this product works every single time", duration=3.2))
    assert isolated.keep is False
    assert isolated.reason == "isolated_restart_marker"
    assert sentence.keep is True


def test_does_not_delete_profanity_or_emotional_reaction_by_itself():
    assert evaluate_take(take("This is fucking amazing and I use it every day")).keep is True
    assert evaluate_take(take("Oh my god this is so good")).keep is True
    assert evaluate_take(take("Damn this looks good on camera")).keep is True


def test_does_not_delete_one_more_when_it_is_normal_product_language():
    assert evaluate_take(take("One more feature I love is the magnetic lid")).keep is True
