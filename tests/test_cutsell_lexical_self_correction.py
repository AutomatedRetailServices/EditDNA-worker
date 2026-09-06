from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.lexical_self_correction import split_explicit_lexical_self_corrections


def _word(text, start, end):
    return Word(text=text, start=float(start), end=float(end), confidence=0.99)


def _take(words):
    words = tuple(words)
    return CandidateTake(
        clip_id="parent",
        source_asset_id="src",
        source_order=0,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
        words=words,
        signals=MediaSignals("src", words[0].start, words[-1].end),
        complete_idea=True,
    )


def test_video04_worthless_oh_priceless_slip_is_cut_without_losing_unique_story():
    take = _take((
        _word("for", 24.70, 24.95),
        _word("the", 24.96, 25.10),
        _word("peace", 25.11, 25.45),
        _word("of", 25.46, 25.57),
        _word("mind", 25.58, 25.90),
        _word("experience", 25.91, 26.55),
        _word("worthless.", 26.56, 27.10),
        _word("Oh,", 27.11, 27.32),
        _word("priceless.", 27.33, 27.92),
    ))

    repaired, diagnostics = split_explicit_lexical_self_corrections((take,))

    assert len(repaired) == 2
    assert repaired[0].text == "for the peace of mind experience"
    assert repaired[1].text == "priceless."
    assert repaired[0].end < repaired[1].start
    assert diagnostics[0]["reason"] == "explicit_lexical_self_correction_cut"
    assert diagnostics[0]["wrong_word"] == "worthless."
    assert diagnostics[0]["corrected_word"] == "priceless."


def test_ordinary_oh_reaction_does_not_trigger_without_lexical_affix_evidence():
    take = _take((
        _word("this", 0.0, 0.2),
        _word("fountain", 0.21, 0.6),
        _word("has", 0.61, 0.8),
        _word("been", 0.81, 1.0),
        _word("really", 1.01, 1.3),
        _word("great.", 1.31, 1.65),
        _word("Oh,", 1.66, 1.85),
        _word("wow.", 1.86, 2.1),
    ))

    repaired, diagnostics = split_explicit_lexical_self_corrections((take,))

    assert repaired == (take,)
    assert diagnostics == ()
