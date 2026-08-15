from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.trailing_retry_restart import trim_trailing_retry_restarts


def words(texts, start, step=0.16):
    return tuple(
        Word(text=text, start=start + index * step, end=start + (index + 1) * step)
        for index, text in enumerate(texts)
    )


def take(clip_id, text, start, end, word_texts=()):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words(word_texts, start) if word_texts else (),
    )


def test_boundary_word_is_kept_and_repeated_retry_suffix_is_trimmed():
    earlier = take(
        "earlier",
        "were one of those people that use your desk to pop your back",
        0.0,
        3.0,
    )
    previous = take(
        "previous",
        "drop the link below if you want to check them",
        37.72,
        40.48,
    )
    tail = take(
        "tail",
        "out if you were one of those",
        40.48,
        41.54,
        ("out", "if", "you", "were", "one", "of", "those"),
    )

    trimmed, diagnostics = trim_trailing_retry_restarts(
        (earlier, previous, tail),
        (earlier, previous, tail),
    )

    by_id = {item.clip_id: item for item in trimmed}
    assert by_id["tail"].text == "out"
    assert by_id["tail"].end == tail.words[0].end
    assert diagnostics[0]["reason"] == "trailing_retry_restart_suffix_trim"
    assert diagnostics[0]["kept_text"] == "out"


def test_same_short_boundary_without_word_timing_fails_open():
    earlier = take("earlier", "were one of those people that use your desk to pop your back", 0.0, 3.0)
    previous = take("previous", "drop the link below if you want to check them", 37.72, 40.48)
    tail = take("tail", "out if you were one of those", 40.48, 41.54)

    trimmed, diagnostics = trim_trailing_retry_restarts(
        (earlier, previous, tail),
        (earlier, previous, tail),
    )

    assert trimmed[-1] == tail
    assert diagnostics == ()


def test_valid_contiguous_transition_without_earlier_retry_is_unchanged():
    earlier = take("earlier", "the jacket has two pockets and a removable hood", 0.0, 3.0)
    previous = take("previous", "drop the link below if you want to check them", 37.72, 40.48)
    tail = take(
        "tail",
        "out and let me show you the zipper",
        40.48,
        41.70,
        ("out", "and", "let", "me", "show", "you", "the", "zipper"),
    )

    trimmed, diagnostics = trim_trailing_retry_restarts(
        (earlier, previous, tail),
        (earlier, previous, tail),
    )

    assert trimmed[-1] == tail
    assert diagnostics == ()


def test_noncontiguous_short_retry_fragment_is_not_boundary_trimmed():
    earlier = take("earlier", "were one of those people that use your desk to pop your back", 0.0, 3.0)
    previous = take("previous", "drop the link below if you want to check them", 37.72, 39.0)
    tail = take(
        "tail",
        "out if you were one of those",
        40.48,
        41.54,
        ("out", "if", "you", "were", "one", "of", "those"),
    )

    trimmed, diagnostics = trim_trailing_retry_restarts(
        (earlier, previous, tail),
        (earlier, previous, tail),
    )

    assert trimmed[-1] == tail
    assert diagnostics == ()
