from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.internal_self_correction import trim_internal_self_corrections


def _take(clip_id, start, words):
    timed = []
    cursor = start
    for text in words:
        timed.append(Word(text=text, start=cursor, end=cursor + 0.2, confidence=0.99))
        cursor += 0.2
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=cursor,
        text=" ".join(words),
        words=tuple(timed),
    )


def test_trims_broken_internal_correction_when_next_take_is_contiguous():
    first = _take(
        "a",
        0.0,
        ["I'm", "gonna", "be", "trying", "out", "this", "menopause", "support", "med", "this", "not", "medicine"],
    )
    second = _take("b", first.end, ["Supplement", "I", "don't", "know", "what", "y'all", "call", "this"])

    trimmed, diagnostics = trim_internal_self_corrections((first, second), (first, second))

    assert trimmed[0].text == "I'm gonna be trying out this menopause support"
    assert trimmed[0].end == first.words[7].end
    assert trimmed[1] == second
    assert diagnostics[0]["reason"] == "internal_self_correction_suffix_trim"
    assert diagnostics[0]["following_clip_id"] == "b"


def test_normal_negation_is_not_treated_as_self_correction():
    first = _take(
        "a",
        0.0,
        ["This", "is", "not", "medicine", "it", "is", "a", "daily", "supplement"],
    )
    second = _take("b", first.end, ["And", "I", "take", "it", "with", "breakfast"])

    trimmed, diagnostics = trim_internal_self_corrections((first, second), (first, second))

    assert trimmed == (first, second)
    assert diagnostics == ()


def test_broken_pattern_without_contiguous_following_take_is_preserved():
    first = _take(
        "a",
        0.0,
        ["I'm", "gonna", "be", "trying", "out", "this", "menopause", "support", "med", "this", "not", "medicine"],
    )
    second = _take("b", first.end + 1.5, ["Something", "totally", "different", "comes", "next"])

    trimmed, diagnostics = trim_internal_self_corrections((first, second), (first, second))

    assert trimmed == (first, second)
    assert diagnostics == ()


def test_exact_word_before_not_does_not_trigger_prefix_correction():
    first = _take(
        "a",
        0.0,
        ["I", "really", "like", "this", "because", "it", "is", "medicine", "not", "medicine"],
    )
    second = _take("b", first.end, ["Anyway", "moving", "on", "to", "the", "next", "thing"])

    trimmed, diagnostics = trim_internal_self_corrections((first, second), (first, second))

    assert trimmed == (first, second)
    assert diagnostics == ()
