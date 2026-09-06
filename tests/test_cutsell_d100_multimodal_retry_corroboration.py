"""D-100 (D-099 Gap #1, bounded encargo): confirmed multimodal
recording-behavior evidence corroborating the EXISTING retry-family
restart-evidence authority.

D-099's dataflow trace found `performance_confirmation.py` already
promotes dense local visual candidates into CONFIRMED `wrong_take`/
`retry_setup` events, and that `take_grouping.py`/`take_grouping_
provider.py` never received any of it -- every restart-evidence rule was
lexical only. This is the minimal plumbing bridge: `whole_video_context`
(already a live variable in `pipeline.py` at the `reconcile_semantic_
idea_equivalence` call site) is narrowed to a plain per-source mapping
of confirmed events (`whole_video_analysis.confirmed_recording_behavior_
events`) and passed down as an OPTIONAL, purely corroborating input.

No new authority. No new provider. `same_opening_restart`,
`_safe_short_prefix_retry` and `incomplete_attempt_completed_by_retry`
all run first, unchanged; `multimodal_corroborated_retry` is tried only
when all three decline. Generic fixtures -- no Video00 wording.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import multimodal_corroborated_retry
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence
from cutsell_worker.whole_video_analysis import (
    SourceVideoContext,
    TemporalEvent,
    WholeVideoContext,
    confirmed_recording_behavior_events,
)
from cutsell_worker.providers import ProviderStatus


def _take(clip_id, start, end, text, *, complete=True, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


# A retry-compatible pair the EXISTING lexical rules do not resolve on
# their own: the openings differ (so `same_opening_restart`'s 4-token
# exact-prefix and `incomplete_attempt_completed_by_retry`'s 2-token
# opening match both miss), even though real shared content exists once
# you look past the opening words.
EARLIER_INCOMPLETE = "shipping delays happen sometimes when trucks break down unexpectedly"
LATER_COMPLETE = "trucks break down and shipping gets delayed for days"

CONFIRMED_EVENTS = {"src": (("wrong_take", 3.5, 4.2),)}


# ---------------------------------------------------------------------------
# Unit tests: take_grouping.multimodal_corroborated_retry
# ---------------------------------------------------------------------------

def test_positive_control_confirmed_event_plus_weak_shared_content_corroborates():
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False)
    later = _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True)
    result = multimodal_corroborated_retry(earlier, later, CONFIRMED_EVENTS)
    assert result is not None
    kind, event = result
    assert kind == "multimodal_corroborated_retry"
    assert event == ("wrong_take", 3.5, 4.2)


def test_negative_control_1_confirmed_event_but_semantically_unrelated_pair_does_not_merge():
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False)
    unrelated = _take(
        "unrelated", 5.0, 9.0,
        "our return policy allows exchanges within thirty days of purchase",
        complete=True,
    )
    assert multimodal_corroborated_retry(earlier, unrelated, CONFIRMED_EVENTS) is None


def test_negative_control_2_complementary_complete_statements_do_not_merge():
    comp_a = _take("comp_a", 0.0, 4.0, "the jacket comes in three different colors for everyone", complete=True)
    comp_b = _take("comp_b", 5.0, 9.0, "the jacket is also machine washable and very durable", complete=True)
    events = {"src": (("retry_setup", 4.2, 4.8),)}
    assert multimodal_corroborated_retry(comp_a, comp_b, events) is None


def test_backward_compatible_no_evidence_supplied_returns_none():
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False)
    later = _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True)
    assert multimodal_corroborated_retry(earlier, later, None) is None
    assert multimodal_corroborated_retry(earlier, later, {}) is None


def test_confirmed_event_far_from_the_boundary_does_not_corroborate():
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False)
    later = _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True)
    far_events = {"src": (("wrong_take", 20.0, 20.5),)}
    assert multimodal_corroborated_retry(earlier, later, far_events) is None


def test_unconfirmed_candidate_kind_is_never_treated_as_corroboration():
    """Only CONFIRMED `wrong_take`/`retry_setup` events corroborate -- the
    dense, unconfirmed `*_candidate` events (D-099's own finding: these
    are measurement, not yet editorial evidence) must never be treated as
    authoritative here, exactly as the bounded task's scope requires."""
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False)
    later = _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True)
    candidate_only = {"src": (("body_reset_candidate", 3.5, 4.2),)}
    assert multimodal_corroborated_retry(earlier, later, candidate_only) is None


def test_different_source_asset_never_corroborates():
    earlier = _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False, source="src_a")
    later = _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True, source="src_b")
    assert multimodal_corroborated_retry(earlier, later, CONFIRMED_EVENTS) is None


# ---------------------------------------------------------------------------
# whole_video_analysis.confirmed_recording_behavior_events
# ---------------------------------------------------------------------------

def _context_with_events(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src", summary="", dominant_style="creator_raw",
            creator_intent="recording_clean_cut", events=events,
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_confirmed_extraction_keeps_only_confirmed_kinds():
    context = _context_with_events(
        TemporalEvent("src", 3.5, 4.2, "wrong_take", 0.9, "confirmed"),
        TemporalEvent("src", 1.0, 1.1, "body_reset_candidate", 0.5, "unconfirmed"),
        TemporalEvent("src", 6.0, 6.4, "retry_setup", 0.7, "confirmed"),
    )
    result = confirmed_recording_behavior_events(context)
    assert result == {"src": (("wrong_take", 3.5, 4.2), ("retry_setup", 6.0, 6.4))}


def test_confirmed_extraction_of_none_context_is_empty():
    assert confirmed_recording_behavior_events(None) == {}


def test_confirmed_extraction_omits_sources_with_no_confirmed_events():
    context = _context_with_events(TemporalEvent("src", 1.0, 1.1, "body_reset_candidate", 0.5, "unconfirmed"))
    assert confirmed_recording_behavior_events(context) == {}


# ---------------------------------------------------------------------------
# Integration: reconcile_semantic_idea_equivalence (the EXISTING authority)
# ---------------------------------------------------------------------------

def test_before_without_confirmed_evidence_the_pair_stays_separate():
    """BEFORE: exactly the production call shape without the new
    parameter -- reproduces today's blindness. No regression to D-097.12
    or any existing rule; this pair simply never merges without the
    multimodal bridge."""
    takes = (
        _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False),
        _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence((("earlier",), ("later",)), takes, None)
    assert merged == (("earlier",), ("later",))
    assert diag["status"] == "not_requested"


def test_after_confirmed_evidence_lets_the_pair_enter_the_same_family():
    """AFTER: the SAME call, only `confirmed_recording_evidence` added --
    the pair now merges through the EXISTING authority, deterministically,
    without ever consulting the arbiter."""
    takes = (
        _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False),
        _take("later", 5.0, 9.0, LATER_COMPLETE, complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("earlier",), ("later",)), takes, None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert len(merged) == 1 and set(merged[0]) == {"earlier", "later"}
    assert diag["status"] == "applied"
    assert diag["provider"] == "deterministic_restart_evidence"
    merge_row = diag["restart_evidence_merges"][0]
    assert merge_row["accepted_by"] == "multimodal_corroborated_retry"
    # Observability: lexical vs. multimodal evidence, event kind, and range
    # are all distinguishable in the diagnostic trail without exposing any
    # benchmark-specific transcript text.
    assert merge_row["corroborating_evidence"] == "confirmed_multimodal_event"
    assert merge_row["corroborating_event_kind"] == "wrong_take"
    assert merge_row["corroborating_event_range"] == [3.5, 4.2]


def test_negative_control_1_through_the_full_authority_no_merge():
    takes = (
        _take("earlier", 0.0, 4.0, EARLIER_INCOMPLETE, complete=False),
        _take("unrelated", 5.0, 9.0, "our return policy allows exchanges within thirty days of purchase", complete=True),
    )
    merged, diag = reconcile_semantic_idea_equivalence(
        (("earlier",), ("unrelated",)), takes, None,
        confirmed_recording_evidence=CONFIRMED_EVENTS,
    )
    assert merged == (("earlier",), ("unrelated",))
    assert diag["status"] == "not_requested"


def test_negative_control_2_through_the_full_authority_no_merge():
    takes = (
        _take("comp_a", 0.0, 4.0, "the jacket comes in three different colors for everyone", complete=True),
        _take("comp_b", 5.0, 9.0, "the jacket is also machine washable and very durable", complete=True),
    )
    events = {"src": (("retry_setup", 4.2, 4.8),)}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("comp_a",), ("comp_b",)), takes, None,
        confirmed_recording_evidence=events,
    )
    assert merged == (("comp_a",), ("comp_b",))
    assert diag["status"] == "not_requested"


def test_existing_lexical_rule_still_wins_and_is_never_overridden_by_multimodal_evidence():
    """When `same_opening_restart` already fires, the merge is recorded
    under its OWN kind, never re-labelled as multimodal -- multimodal
    corroboration is tried only after every lexical rule declines."""
    failed = "When my contract ended I spoke with my doctor about every test available today"
    clean = "When my contract ended I switched to a different doctor about every test available today"
    takes = (_take("failed", 0.0, 5.0, failed), _take("clean", 6.0, 11.0, clean))
    # Even with confirmed evidence present, the lexical rule (which fires
    # first) is what gets credited.
    events = {"src": (("wrong_take", 5.2, 5.8),)}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None,
        confirmed_recording_evidence=events,
    )
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    assert diag["restart_evidence_merges"][0]["accepted_by"] == "same_opening_restart"
    assert "corroborating_evidence" not in diag["restart_evidence_merges"][0]
