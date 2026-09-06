from benchmarks.video00_semantic_alignment import (
    COMPOSITE,
    DUPLICATE,
    EXACT,
    EXTRA,
    MISSING,
    RECHUNKED,
    align,
)


def test_exact_one_to_one_alignment():
    gold = ["the product launched successfully", "customers loved the results"]
    candidate = ["the product launched successfully", "customers loved the results"]

    result = align(gold, candidate)

    assert result.aligned is True
    assert [row.relation for row in result.rows] == [EXACT, EXACT]
    assert result.missing_count == 0


def test_rechunked_two_gold_segments_into_one_candidate_segment():
    gold = ["the swelling started in my face", "and I also gained some weight"]
    candidate = ["the swelling started in my face and I also gained some weight"]

    result = align(gold, candidate)

    assert result.aligned is True
    assert len(result.rows) == 1
    assert result.rows[0].relation == RECHUNKED
    assert result.rows[0].gold_span == (0, 2)
    assert result.rows[0].candidate_span == (0, 1)


def test_composite_one_gold_segment_split_into_two_candidate_pieces():
    gold = ["the swelling started in my face and I also gained some weight"]
    candidate = ["the swelling started in my face", "and I also gained some weight"]

    result = align(gold, candidate)

    assert result.aligned is True
    assert len(result.rows) == 1
    assert result.rows[0].relation == COMPOSITE
    assert result.rows[0].gold_span == (0, 1)
    assert result.rows[0].candidate_span == (0, 2)


def test_missing_gold_segment_with_no_candidate_realization_anywhere():
    gold = ["the product launched successfully", "the biopsy confirmed a diagnosis"]
    candidate = ["the product launched successfully"]

    result = align(gold, candidate)

    assert result.aligned is False
    assert result.missing_count == 1
    missing_rows = [row for row in result.rows if row.relation == MISSING]
    assert missing_rows[0].gold_text == "the biopsy confirmed a diagnosis"


def test_extra_candidate_content_not_asked_for_by_any_gold_segment():
    gold = ["the product launched successfully"]
    candidate = ["the product launched successfully", "an unrelated aside about the weather today"]

    result = align(gold, candidate)

    # Extra, unexplained content does not by itself fail alignment.
    assert result.aligned is True
    assert result.extra_candidate_indices == (1,)
    assert result.duplicate_candidate_indices == ()


def test_duplicate_candidate_segment_repeating_already_matched_gold_content():
    gold = ["the product launched successfully"]
    candidate = [
        "the product launched successfully",
        "the product launched successfully",  # same idea rendered twice
    ]

    result = align(gold, candidate)

    assert result.aligned is False  # a duplicate rendering IS flagged
    assert result.duplicate_candidate_indices == (1,)
    assert result.extra_candidate_indices == ()


def test_reordered_content_reports_missing_rather_than_a_false_match():
    # The candidate delivers idea B before idea A -- the aligner walks
    # strictly left-to-right and must never match idea A's content by
    # looking backward, so idea A correctly reports MISSING (a real
    # ordering break), not a silently-accepted out-of-order match.
    gold = ["the biopsy confirmed a diagnosis", "we discussed treatment options afterward"]
    candidate = ["we discussed treatment options afterward", "the biopsy confirmed a diagnosis"]

    result = align(gold, candidate)

    assert result.aligned is False
    assert result.rows[0].relation == MISSING
    assert result.rows[0].gold_text == "the biopsy confirmed a diagnosis"


def test_minor_asr_wording_variance_still_aligns_exact_enough():
    # Not byte-identical, but high content-token overlap -- exactly the
    # tolerance a real transcript's run-to-run ASR variance needs.
    gold = ["they found a nodule during the sonography and ordered a biopsy"]
    candidate = ["during the sonography they found a nodule and ordered a biopsy"]

    result = align(gold, candidate)

    assert result.aligned is True
    assert result.rows[0].relation == EXACT
    assert result.rows[0].content_coverage >= 0.9


def test_extra_segment_between_two_real_matches_does_not_break_alignment():
    gold = ["the product launched successfully", "customers loved the results"]
    candidate = [
        "the product launched successfully",
        "an unrelated retry fragment that got left in by mistake",
        "customers loved the results",
    ]

    result = align(gold, candidate)

    assert result.aligned is True
    assert [row.relation for row in result.rows] == [EXACT, EXACT]
    assert result.extra_candidate_indices == (1,)


def test_tiny_trailing_fragment_still_rechunks_with_its_real_neighbor():
    # A near-content-free trailing fragment ("finally.") legitimately merges
    # with the PRECEDING gold segment when the candidate has already
    # absorbed it there -- exempt from the per-segment coverage floor
    # (its own content is too thin to prove/disprove anything), unlike a
    # real, substantial missing idea (see the next test).
    gold = [
        "they found something unusual during the scan",
        "finally.",
        "we discussed treatment options afterward",
    ]
    candidate = [
        "they found something unusual during the scan finally.",
        "we discussed treatment options afterward",
    ]

    result = align(gold, candidate)

    # The tiny fragment merges with WHICHEVER real neighbor the candidate
    # actually grouped it with (direction is not the point here) -- what
    # matters is it is never reported as its own missing gold segment.
    assert result.aligned is True
    assert result.missing_count == 0
    assert any(row.relation == RECHUNKED for row in result.rows)


def test_a_substantial_missing_gold_segment_is_never_masked_by_a_strong_neighbor():
    # The opposite of the tiny-fragment case above: a real, substantial
    # idea with no candidate realization anywhere must not be hidden just
    # because it happens to be windowed next to a well-covered neighbor
    # during the search.
    gold = [
        "the product launched successfully",
        "customers loved the results",
        "call to action asking viewers to subscribe",
    ]
    candidate = [
        "the product launched successfully",
        "call to action asking viewers to subscribe",
    ]

    result = align(gold, candidate)

    assert result.aligned is False
    assert result.missing_count == 1
    missing = [row for row in result.rows if row.relation == MISSING]
    assert missing[0].gold_text == "customers loved the results"


def test_empty_gold_and_candidate_aligns_trivially():
    result = align([], [])
    assert result.aligned is True
    assert result.rows == ()
    assert result.extra_candidate_indices == ()
