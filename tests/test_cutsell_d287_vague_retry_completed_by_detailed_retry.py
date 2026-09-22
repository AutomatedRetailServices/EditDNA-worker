"""D-287 (RAW #120 audit) -- a punctuation-complete but editorially VAGUE
take, immediately completed by a much more detailed same-opening retry,
is never merged into the same retry family.

Real RAW #120 evidence: "Ahi fue cuando me mando." (that is when they sent
me) and its completion "Ahi fue cuando me mandaron a hacer sonografia de
tiroides y otras sonografias." (that is when they sent me to get a thyroid
ultrasound...) both survived to KEEP with DIFFERENT `take_group_id`s in the
engine's own JSON -- they never even competed for Best Take. D-156 (RAW
#119 audit) already fixed `incomplete_attempt_completed_by_retry`'s
verb-inflection matching, but that function's own precondition requires
`earlier.complete_idea is False` -- and `take_segmentation._looks_complete_
idea` grades completeness from PUNCTUATION ALONE (`_ends_sentence`): "Ahi
fue cuando me mando." ends with a period, so it is graded complete_idea=
True despite being editorially vague (it never says what they were sent to
do). The pair never reaches D-156's fix at all.

Root cause fix: a narrowly-scoped sibling rule for exactly the punctuation-
complete-but-vague shape, deliberately NOT a change to `_looks_complete_
idea` itself (used pervasively; loosening it would be a much larger blast
radius than this defect). Its safety bar is STRICTER than `incomplete_
attempt_completed_by_retry`'s own `minimum_shared_content>=2` floor: FULL
content coverage of the vague side (every real content word the shorter
delivery makes must appear in the longer one) plus a minimum 2x length
ratio -- directly answering the audit concern that a lexical/stem
coincidence must never fuse two COMPLETE realizations with UNEQUAL
coverage (e.g. a long distinct conclusion folded into an unrelated short
microclip that merely opens similarly). Generic fixtures only (no Video00
text); reuses the same verified stem-match pair ("reporter"/"reported")
D-156's own test file already proved crosses the 80% prefix floor.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import vague_retry_completed_by_detailed_retry


def _take(clip_id, start, end, text, complete_idea, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete_idea)


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------

def test_the_real_reported_shape_exact_content_word_shared():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True)
    later = _take(
        "later", 105.0, 112.0,
        "That is when they shipped the replacement parts to the regional warehouse team.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) == "vague_retry_completed_by_detailed_retry"


def test_stem_matched_content_word_via_the_d156_matcher():
    earlier = _take("earlier", 100.0, 102.5, "That is when the office reported.", True)
    later = _take(
        "later", 105.0, 113.0,
        "That is when the office reporter finally confirmed the full details of the regional incident downtown.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) == "vague_retry_completed_by_detailed_retry"


def test_order_independent_later_take_may_be_passed_as_left_argument():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True)
    later = _take(
        "later", 105.0, 112.0,
        "That is when they shipped the replacement parts to the regional warehouse team.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(later, earlier) == "vague_retry_completed_by_detailed_retry"


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def test_opening_only_match_with_zero_further_overlap_declines():
    earlier = _take("earlier", 100.0, 102.0, "That is when they left.", True)
    later = _take(
        "later", 105.0, 112.0,
        "That is when the weather finally cleared up across the entire coastal region.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_partial_coverage_of_the_vague_side_declines_the_core_safety_bar():
    # D-156 audit concern, directly exercised: the vague side makes TWO real
    # claims; the longer take only covers ONE of them. A partial/coincidental
    # lexical overlap must never fuse two unequal-coverage realizations.
    earlier = _take("earlier", 100.0, 103.0, "That is when the manager reported the delay.", True)
    later = _take(
        "later", 106.0, 114.0,
        "That is when the manager finally reached the regional office downtown after the drive.",
        True,
    )
    # later covers "manager" but never covers "reported"/"delay" (or a stem
    # of either) -- partial coverage only.
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_later_take_not_materially_longer_declines_even_with_full_coverage():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True)
    later = _take("later", 105.0, 107.0, "That is when they shipped ok.", True)
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_grammatically_incomplete_take_is_not_this_rules_shape():
    # Falls to incomplete_attempt_completed_by_retry instead -- this rule is
    # scoped to BOTH sides being punctuation-complete.
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped", False)
    later = _take(
        "later", 105.0, 112.0,
        "That is when they shipped the replacement parts to the regional warehouse team.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_both_takes_vague_declines():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True)
    later = _take("later", 105.0, 107.0, "That is when they shipped that.", True)
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_different_source_asset_declines():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True, source="src_a")
    later = _take(
        "later", 105.0, 112.0,
        "That is when they shipped the replacement parts to the regional warehouse team.",
        True, source="src_b",
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_gap_too_large_declines():
    earlier = _take("earlier", 100.0, 102.0, "That is when they shipped it.", True)
    later = _take(
        "later", 130.0, 137.0,
        "That is when they shipped the replacement parts to the regional warehouse team.",
        True,
    )
    assert vague_retry_completed_by_detailed_retry(earlier, later) is None


def test_reverse_shape_long_conclusion_then_unrelated_short_microclip_declines():
    # The exact adversarial shape the audit was worried about: a LONG,
    # complete conclusion comes first chronologically, an unrelated SHORT
    # microclip with a superficially similar opening comes later. Because
    # the chronologically-earlier take is the long one, its own content
    # would need FULL coverage inside the much SHORTER later clip -- the
    # length-ratio gate alone already forecloses this regardless of any
    # lexical/stem coincidence in the opening.
    long_conclusion = _take(
        "long_conclusion", 200.0, 218.0,
        "That is my story. I am the only one in my family with this condition and I "
        "believe most of it comes down to daily choices, so please take care of yourself.",
        True,
    )
    short_microclip = _take("short_microclip", 240.0, 244.0, "That is the only thing left.", True)
    assert vague_retry_completed_by_detailed_retry(long_conclusion, short_microclip) is None
