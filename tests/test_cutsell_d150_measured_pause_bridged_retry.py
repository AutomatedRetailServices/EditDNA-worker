"""D-150 (Gate 6 correction, real RAW #118 audit): a real audited pair could
not be caught by ANY existing retry-family rule. A short false start shared
only its 2-word opening with its later, complete realization -- the rest of
each text used COMPLETELY DIFFERENT vocabulary (the creator restarted with a
different noun/phrasing choice throughout, not just at the opening), and no
confirmed multimodal marker (`wrong_take`/`retry_setup`) existed at the
boundary either -- only a long stretch of measured source silence.

`incomplete_attempt_completed_by_retry` structurally cannot reach this shape
(its `minimum_shared_content` gate needs lexical overlap BEYOND the opening
that, by this shape's own definition, does not exist). `multimodal_
corroborated_retry` cannot reach it either (no confirmed event exists).
D-097.12's own module comment already named the risk of merging on a shared
opener alone (an unrelated aside sharing the same generic opener) and
refused to solve it lexically -- correctly, since there is no reliable
lexical signal left to check.

The general, non-lexical evidence that tells a genuine paused restart apart
from a coincidental unrelated aside: a real abandoned-then-completed attempt
has NOTHING ELSE said in the gap -- the creator paused, then continued. An
unrelated aside has ITS OWN speech content occupying that time instead of
silence. `measured_pause_bridged_retry` requires the SAME 2-word opening
match, completeness asymmetry, AND that most of the gap is already-measured
source silence.

Generic fixtures only -- no Video00 wording, timestamps, or clip ids.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import measured_pause_bridged_retry
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext, measured_silence_intervals
from cutsell_worker.providers import ProviderStatus


def _take(clip_id, start, end, text, *, complete, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


# Shares only the 2-word opening ("we had") -- zero further lexical overlap,
# exactly the shape `incomplete_attempt_completed_by_retry` cannot reach.
SHORT_FALSE_START = "We had trouble with the"
LATER_COMPLETE = "We had setbacks getting the new system running, but the team eventually fixed everything and shipped on time"

SILENCE_MOSTLY_COVERING_GAP = {"src": ((4.5, 9.7),)}  # take ends 4.0, later starts 10.0 -> gap 6.0s, covered 5.2s


# ---------------------------------------------------------------------------
# Unit tests: take_grouping.measured_pause_bridged_retry
# ---------------------------------------------------------------------------

def test_positive_control_shared_opening_plus_mostly_silent_gap_bridges():
    earlier = _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False)
    later = _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True)
    result = measured_pause_bridged_retry(earlier, later, SILENCE_MOSTLY_COVERING_GAP)
    assert result == "measured_pause_bridged_retry"


def test_negative_control_no_shared_opening_never_bridges():
    """An unrelated pair with no shared opener at all -- even with a mostly
    silent gap -- must never merge. The opener match is still required."""
    earlier = _take("earlier", 0.0, 4.0, "I forgot what I wanted to say", complete=False)
    later = _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True)
    assert measured_pause_bridged_retry(earlier, later, SILENCE_MOSTLY_COVERING_GAP) is None


def test_negative_control_shared_opener_but_gap_mostly_unrelated_speech_never_bridges():
    """The core false-positive guard: same shared opener, but the gap is
    NOT silence -- something else was said in between (represented here by
    a silence interval far too short to explain the gap). This is the
    'unrelated aside sharing a generic opener' shape D-097.12's own comment
    warns about -- must not merge."""
    earlier = _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False)
    later = _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True)
    barely_any_silence = {"src": ((9.9, 10.0),)}  # 0.1s of a 6.0s gap
    assert measured_pause_bridged_retry(earlier, later, barely_any_silence) is None


def test_negative_control_no_silence_evidence_at_all_never_bridges():
    earlier = _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False)
    later = _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True)
    assert measured_pause_bridged_retry(earlier, later, None) is None
    assert measured_pause_bridged_retry(earlier, later, {}) is None


def test_negative_control_completeness_symmetry_never_bridges():
    """Two already-COMPLETE statements sharing an opener, separated by a
    silent gap, must not merge -- completeness asymmetry is still required
    (the same safe shape every rule in this family shares)."""
    a = _take("a", 0.0, 4.0, "We had a great time at the conference last year", complete=True)
    b = _take("b", 10.0, 18.0, "We had a great time visiting the new office downtown", complete=True)
    assert measured_pause_bridged_retry(a, b, SILENCE_MOSTLY_COVERING_GAP) is None


def test_negative_control_gap_too_large_never_bridges():
    earlier = _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False)
    later = _take("later", 30.0, 38.0, LATER_COMPLETE, complete=True)
    far_silence = {"src": ((4.5, 29.7),)}
    assert measured_pause_bridged_retry(earlier, later, far_silence) is None


def test_negative_control_different_source_never_bridges():
    earlier = _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False, source="src_a")
    later = _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True, source="src_b")
    assert measured_pause_bridged_retry(earlier, later, SILENCE_MOSTLY_COVERING_GAP) is None


# ---------------------------------------------------------------------------
# whole_video_analysis.measured_silence_intervals
# ---------------------------------------------------------------------------

def _context_with_events(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src", summary="", dominant_style="creator_raw",
            creator_intent="recording_clean_cut", events=events,
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_silence_extraction_keeps_only_confident_silence_events():
    context = _context_with_events(
        TemporalEvent("src", 4.5, 9.7, "audio_silence_interval", 1.0, "measured"),
        TemporalEvent("src", 1.0, 1.1, "body_reset_candidate", 0.9, "unrelated kind"),
        TemporalEvent("src", 20.0, 20.3, "audio_silence_interval", 0.5, "below confidence floor"),
    )
    result = measured_silence_intervals(context)
    assert result == {"src": ((4.5, 9.7),)}


def test_silence_extraction_of_none_context_is_empty():
    assert measured_silence_intervals(None) == {}


def test_silence_extraction_omits_sources_with_no_silence_events():
    context = _context_with_events(TemporalEvent("src", 1.0, 1.1, "wrong_take", 0.95, "unrelated kind"))
    assert measured_silence_intervals(context) == {}


# ---------------------------------------------------------------------------
# Integration: reconcile_semantic_idea_equivalence (the full authority)
# ---------------------------------------------------------------------------

def test_before_without_silence_evidence_the_pair_stays_separate():
    takes = (
        _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False),
        _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence((("earlier",), ("later",)), takes, None)
    assert merged == (("earlier",), ("later",))


def test_after_silence_evidence_lets_the_pair_enter_the_same_family():
    takes = (
        _take("earlier", 0.0, 4.0, SHORT_FALSE_START, complete=False),
        _take("later", 10.0, 18.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("earlier",), ("later",)), takes, None,
        measured_silence_evidence=SILENCE_MOSTLY_COVERING_GAP,
    )
    assert len(merged) == 1 and set(merged[0]) == {"earlier", "later"}
    assert diag["status"] == "applied"
    merge_row = diag["restart_evidence_merges"][0]
    assert merge_row["accepted_by"] == "measured_pause_bridged_retry"


def test_existing_lexical_rule_still_wins_and_is_never_overridden_by_silence_evidence():
    """When `same_opening_restart` already fires, the merge is recorded
    under ITS OWN kind, never re-labelled -- the silence-bridged rule only
    runs once every lexical rule above has already declined."""
    failed = "When my contract ended I spoke with my doctor about every test available today"
    clean = "When my contract ended I switched to a different doctor about every test available today"
    takes = (_take("failed", 0.0, 5.0, failed, complete=False), _take("clean", 6.0, 11.0, clean, complete=True))
    silence = {"src": ((5.0, 6.0),)}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None,
        measured_silence_evidence=silence,
    )
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    assert diag["restart_evidence_merges"][0]["accepted_by"] == "same_opening_restart"
