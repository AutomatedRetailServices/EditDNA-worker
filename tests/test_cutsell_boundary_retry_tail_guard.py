from cutsell_worker.contracts import DraftClip, Word
from cutsell_worker import final_boundary_authority as authority


def _clip(clip_id, start, end, text, words):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        words=tuple(words),
    )


def test_retry_covered_short_tail_is_trimmed_to_prior_completed_sentence():
    left_words = (
        Word("terminó.", 0.0, 1.0),
        Word("Ahí", 1.4, 1.7),
        Word("me", 1.72, 1.9),
        Word("mandó", 1.92, 2.25),
        Word("a", 2.27, 2.36),
        Word("hacer", 2.38, 2.72),
        Word("sonografías.", 2.74, 3.35),
    )
    right_words = (
        Word("a", 10.0, 10.1),
        Word("hacer", 10.12, 10.45),
        Word("sonografía", 10.47, 10.95),
        Word("de", 10.97, 11.08),
        Word("tiroides", 11.10, 11.55),
        Word("y", 11.57, 11.65),
        Word("otras", 11.67, 11.95),
        Word("sonografías.", 11.97, 12.50),
    )
    left_original = _clip("left", 0.0, 1.72, "terminó. Ahí", left_words[:2])
    right_original = _clip("right", 10.0, 12.5, "a hacer sonografía de tiroides y otras sonografías.", right_words)
    left_expanded = _clip(
        "left", 0.0, 3.35, "terminó. Ahí me mandó a hacer sonografías.", left_words
    )
    right_expanded = right_original

    output, rows = authority._reconcile_same_source_overlaps(
        (left_original, right_original),
        [left_expanded, right_expanded],
        {"src": left_words + right_words},
    )

    assert output[0].end == 1.0
    assert output[0].text == "terminó."
    assert any(row.get("action") == "trim_retry_covered_trailing_clause" for row in rows)


def test_unique_short_tail_fails_open_and_is_preserved():
    left_words = (
        Word("terminó.", 0.0, 1.0),
        Word("Luego", 1.4, 1.75),
        Word("compré", 1.77, 2.15),
        Word("medicación.", 2.17, 2.75),
    )
    right_words = (
        Word("me", 10.0, 10.15),
        Word("hicieron", 10.17, 10.55),
        Word("una", 10.57, 10.72),
        Word("sonografía.", 10.74, 11.3),
    )
    left_original = _clip("left", 0.0, 1.5, "terminó. Luego", left_words[:2])
    right_original = _clip("right", 10.0, 11.3, "me hicieron una sonografía.", right_words)
    left_expanded = _clip("left", 0.0, 2.75, "terminó. Luego compré medicación.", left_words)

    output, rows = authority._reconcile_same_source_overlaps(
        (left_original, right_original),
        [left_expanded, right_original],
        {"src": left_words + right_words},
    )

    assert output[0].end == 2.75
    assert output[0].text.endswith("medicación.")
    assert not any(row.get("action") == "trim_retry_covered_trailing_clause" for row in rows)
