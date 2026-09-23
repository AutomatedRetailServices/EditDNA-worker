"""D-144 (Gate 6 / RAW #118 Gap A): a short, grammatically-incomplete false
start is structurally excluded from EVER being compared against its later
complete realization, because `_cross_group_candidate_pairs` (take_grouping_
provider.py) drops any take whose text is 3 words or fewer BEFORE any of the
retry-family matching rules (`same_opening_restart`,
`incomplete_attempt_completed_by_retry`, `multimodal_corroborated_retry`) get
a chance to run. A false start is short by definition -- the creator caught
themselves and stopped -- so this eligibility gate forecloses exactly the
shape D-097.12 and D-100 were built to catch, whenever the abandoned opening
is very short. This is independent of wording: the fix does not require any
literal text/opening-token overlap, only that a SHORT candidate remains
eligible for comparison when it is itself an incomplete delivery (an already-
general, non-lexical field every `CandidateTake` carries).

Generic fixtures only -- no Video00 wording, timestamps, or clip ids.

D-147 correction (Gate 6, real RAW #118 audit): the original version of this
file used a false start sharing only ONE content word with its completion.
That was itself a false-positive risk -- see
`test_negative_control_a_single_coincidental_shared_word_never_corroborates`
below, which reproduces (with generic text) the real audit's finding that a
single incidental shared word plus a nearby confirmed event is indistinguishable
from a genuine retry using lexical evidence alone. The positive fixtures below
now share TWO real content words, matching the raised
`_MULTIMODAL_CORROBORATION_MINIMUM_SHARED_CONTENT` floor (1 -> 2).
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import multimodal_corroborated_retry
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence


def _take(clip_id, start, end, text, *, complete, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


# The false start: three words, grammatically incomplete (no terminal
# punctuation), sharing TWO real content words with its later, differently-
# worded completion -- still a short false start (semantic_key word count
# <= 3, exactly the class the eligibility fix admits), just not reduced to a
# single coincidental word.
SHORT_FALSE_START = "I felt awful"
LATER_COMPLETE = "I still felt pretty awful about the whole situation afterward"

CONFIRMED_EVENTS = {"src": (("wrong_take", 4.5, 5.0),)}


def test_short_incomplete_false_start_is_corroborated_and_merges():
    """AFTER the eligibility fix: the short incomplete take is now offered
    as a candidate pair, and the EXISTING, already-general multimodal
    corroboration authority (D-100) -- not any new lexical/lit-text rule --
    merges it because a confirmed `wrong_take` event sits at the boundary
    and real (if sparse) shared content exists."""
    takes = (
        _take("false_start", 0.0, 4.0, SHORT_FALSE_START, complete=False),
        _take("realization", 6.0, 12.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("false_start",), ("realization",)), takes, None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert len(merged) == 1 and set(merged[0]) == {"false_start", "realization"}
    assert diag["status"] == "applied"
    merge_row = diag["restart_evidence_merges"][0]
    assert merge_row["accepted_by"] == "multimodal_corroborated_retry"


def test_negative_control_short_but_already_complete_fragment_stays_excluded():
    """A short fragment that is already a COMPLETE idea (a filler
    acknowledgement, not a false start) must NOT gain new eligibility --
    only incompleteness earns the exception. Protects against the
    eligibility relaxation turning every short aside into a candidate."""
    takes = (
        _take("aside", 0.0, 1.5, "Thanks everyone", complete=True),
        _take("realization", 6.0, 12.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("aside",), ("realization",)), takes, None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert merged == (("aside",), ("realization",))
    assert diag["status"] == "no_eligible_pairs"


def test_negative_control_short_incomplete_fragment_with_no_evidence_or_content_stays_separate():
    """Eligibility alone must never cause a merge: a short incomplete
    fragment that shares no real content and has no confirmed event at the
    boundary is offered as a candidate pair (now eligible) but every
    matching rule still declines it -- `no_eligible_pairs` becomes
    `not_requested` (a rule ran and said no), never `applied`."""
    takes = (
        _take("false_start", 0.0, 4.0, "I had problems", complete=False),
        _take(
            "unrelated", 6.0, 12.0,
            "our return policy allows exchanges within thirty days of purchase",
            complete=True,
        ),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("false_start",), ("unrelated",)), takes, None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert merged == (("false_start",), ("unrelated",))
    assert diag["status"] == "not_requested"


def test_negative_control_bare_one_or_two_token_interjection_still_never_corroborates():
    """The lowered multimodal minimum-tokens floor (4 -> 3) must not open
    the door to a bare interjection -- 2 tokens stays below the floor, so
    this rule declines before even reaching its own content/event gates."""
    interjection = _take("interjection", 0.0, 1.0, "no wait", complete=False)
    later = _take("realization", 6.0, 12.0, LATER_COMPLETE, complete=True)
    assert multimodal_corroborated_retry(interjection, later, CONFIRMED_EVENTS) is None


def test_negative_control_a_single_coincidental_shared_word_never_corroborates():
    """D-147 (real RAW #118 audit finding): a short, unrelated fragment that
    happens to share exactly ONE incidental content word with a long, topically
    distant later clip -- PLUS a confirmed event at the boundary -- must never
    merge. Before the fix (minimum_shared_content=1), this scored identically
    to a genuine short false start on every gate (a single word out of one
    possible always scores ratio=1.0): there was no lexical signal separating
    an accidental one-word echo from a real retry. Two unrelated topics merely
    mentioning the same one word (here: "meetings") is not evidence of a
    retry -- it is evidence of nothing more than that one shared word."""
    unrelated_short = _take("unrelated_short", 0.0, 4.0, "I had meetings", complete=False)
    unrelated_long = _take(
        "unrelated_long", 6.0, 12.0,
        "the schedule had many meetings planned for next quarter",
        complete=True,
    )
    assert multimodal_corroborated_retry(unrelated_short, unrelated_long, CONFIRMED_EVENTS) is None

    merged, diag = reconcile_semantic_idea_equivalence(
        (("unrelated_short",), ("unrelated_long",)), (unrelated_short, unrelated_long), None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert merged == (("unrelated_short",), ("unrelated_long",))
    assert diag["status"] == "not_requested"
