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


def _trim_rows(rows):
    return [row for row in rows if row.get("action") == "trim_retry_fully_covered_trailing_clause"]


def test_retry_covered_short_tail_is_trimmed_to_prior_completed_sentence():
    left_words = (
        Word("terminó.", 0.0, 1.0),
        Word("a", 1.4, 1.5),
        Word("hacer", 1.52, 1.86),
        Word("sonografías.", 1.88, 2.45),
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
    left_original = _clip("left", 0.0, 1.5, "terminó. a", left_words[:2])
    right_original = _clip("right", 10.0, 12.5, "a hacer sonografía de tiroides y otras sonografías.", right_words)
    left_expanded = _clip(
        "left", 0.0, 2.45, "terminó. a hacer sonografías.", left_words
    )
    right_expanded = right_original

    output, rows = authority._reconcile_same_source_overlaps(
        (left_original, right_original),
        [left_expanded, right_expanded],
        {"src": left_words + right_words},
    )

    assert output[0].end == 1.0
    assert output[0].text == "terminó."
    assert _trim_rows(rows)


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
    assert not _trim_rows(rows)


def test_high_overlap_tail_with_one_unique_concept_is_preserved():
    left_words = (
        Word("terminó.", 0.0, 1.0),
        Word("Era", 1.4, 1.65),
        Word("como", 1.67, 1.9),
        Word("un", 1.92, 2.05),
        Word("brote", 2.07, 2.40),
        Word("una", 2.42, 2.58),
        Word("alergia.", 2.60, 3.10),
    )
    right_words = (
        Word("parecía", 10.0, 10.35),
        Word("una", 10.37, 10.50),
        Word("alergia", 10.52, 10.90),
        Word("como", 10.92, 11.10),
        Word("brote.", 11.12, 11.45),
    )
    left_original = _clip("left", 0.0, 1.65, "terminó. Era", left_words[:2])
    right_original = _clip("right", 10.0, 11.45, "parecía una alergia como brote.", right_words)
    left_expanded = _clip("left", 0.0, 3.10, "terminó. Era como un brote una alergia.", left_words)

    output, rows = authority._reconcile_same_source_overlaps(
        (left_original, right_original),
        [left_expanded, right_original],
        {"src": left_words + right_words},
    )

    assert output[0].end == 3.10
    assert output[0].text.endswith("una alergia.")
    assert not _trim_rows(rows)
