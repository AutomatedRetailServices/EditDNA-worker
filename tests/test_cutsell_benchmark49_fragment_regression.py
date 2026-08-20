from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.semantic_fragment_guard import remove_semantic_fragment_debris


def _words(text, start=0.0, step=0.25):
    out = []
    cursor = float(start)
    for token in text.split():
        out.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(out)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def test_video02_failed_085_pronoun_collision_fragment_is_removed():
    failed = _take("failed", 149.20, 152.22, "I people It was very funny")
    winner = _take(
        "winner",
        177.83,
        205.01,
        "People ask me all the time do you actually have fun in your job and the answer is yes obviously",
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (failed, winner),
        (("failed", "failed", 0.85), ("winner", "winner", 0.92)),
    )

    assert tuple(t.clip_id for t in kept) == ("winner",)
    assert tuple(t.clip_id for t in removed) == ("failed",)
    assert diagnostics[0]["reason"] == "semantic_failed_pronoun_collision_fragment"


def test_generic_complete_085_failed_short_speech_remains_fail_open():
    ordinary = _take("ordinary", 0.0, 2.0, "candidate speech number one")

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (ordinary,),
        (("ordinary", "failed", 0.85),),
    )

    assert tuple(t.clip_id for t in kept) == ("ordinary",)
    assert removed == ()
    assert diagnostics == ()


def test_short_winner_hook_is_never_removed_by_failed_fragment_rule():
    hook = _take("hook", 0.0, 2.5, "This is actually crazy good")

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (hook,),
        (("hook", "winner", 0.99),),
    )

    assert tuple(t.clip_id for t in kept) == ("hook",)
    assert removed == ()
    assert diagnostics == ()
