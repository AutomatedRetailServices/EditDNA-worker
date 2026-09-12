"""D-239O: EXACT LOST-ATOM TARGET -> P1 MOMENT OBSERVABILITY -- offline tests.

Generic, abstract fixtures only -- NO Video00 literal text/spans/ids. Covers
the directive's 15-item offline fixture matrix for
``editorial_moment_sequence_integration.exact_p1_target_evidence_for``:
matching span+moment+confident process role; matching span+moment+audience
role; low confidence; role uncertain; audience uncertain; no
UnderstandingSpan; span exists but no EditorialMoment; dropped candidate
still represented; selected candidate represented; cross-source isolation;
exact clip/source_span equality; no attempt_id substitution; no aggregate
fallback; deterministic output; no transcript leakage.

Zero policy change: every fixture calls ``p1_moment_role_and_audience_
status_by_clip_id_for`` (Seam C, UNCHANGED) to build the SAME maps the real
pipeline would build, then passes them straight through to
``exact_p1_target_evidence_for`` -- this file never recomputes or bypasses
Seam C's own gate.
"""
from __future__ import annotations

from cutsell_worker.editorial_moment_sequence import (
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    EditorialMoment,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_UNCERTAIN,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    ALLOWED_P1_TARGET_LOOKUP_STATUSES,
    EditorialLocalGroup,
    EditorialMomentUnderstanding,
    P1_TARGET_STATUS_AMBIGUOUS,
    P1_TARGET_STATUS_MOMENT_FOUND_AUDIENCE_UNCERTAIN,
    P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE,
    P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED,
    P1_TARGET_STATUS_MOMENT_FOUND_ROLE_UNCERTAIN,
    P1_TARGET_STATUS_NO_EDITORIAL_MOMENT,
    P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN,
    exact_p1_target_evidence_for,
    p1_moment_role_and_audience_status_by_clip_id_for,
)
from cutsell_worker.watch_listen_understanding import (
    CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
    MEANING_COMPLETE as WL_MEANING_COMPLETE,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)


# ---------------------------------------------------------------------------
# Generic fixture factories.
# ---------------------------------------------------------------------------
def _span(span_id, source_asset_id="src1", start=0.0, end=1.0):
    return UnderstandingSpan(
        span_id=span_id, source_asset_id=source_asset_id, source_start=start, source_end=end,
        behavior_state_hypotheses=(), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(),
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(source_asset_id, *spans):
    return WatchListenUnderstanding(source_asset_id=source_asset_id, understanding_spans=tuple(spans))


def _moment(
    clip_id, source_asset_id="src1", *, moment_id=None, role=MOMENT_ROLE_RECORDING_PROCESS,
    confidence=CONFIDENCE_SUPPORTED, audience_status=AUDIENCE_DELIVERY_SUPPORTED,
    recording_process_status="RECORDING_PROCESS_REGION", proposition_candidate_ids=(),
    attempt_ids=(),
):
    return EditorialMoment(
        source_asset_id=source_asset_id,
        editorial_moment_id=moment_id or f"moment_{clip_id}",
        source_start=0.0, source_end=1.0, source_span_id=clip_id,
        attempt_ids=attempt_ids, proposition_candidate_ids=proposition_candidate_ids,
        related_span_ids=(), moment_role=role, audience_delivery_status=audience_status,
        recording_process_status=recording_process_status, completion_status="COMPLETE",
        local_sequence_position=0, confidence=confidence, conflict_flags=(), provenance=(),
    )


def _understanding(source_asset_id, moments=(), local_groups=()):
    return EditorialMomentUnderstanding(
        source_asset_id=source_asset_id, moments=tuple(moments), sequence_hypotheses=(),
        moment_count=len(moments), sequence_count=0, capability_status="AVAILABLE",
        missing_evidence=(), confidence=CONFIDENCE_SUPPORTED, conflict_flags=(),
        provenance=(), local_groups=tuple(local_groups),
    )


def _evidence_for(target_source_asset_id_by_clip_id, *, editorial_moment_understandings, watch_listen_understandings):
    role_by_clip_id, audience_status_by_clip_id = p1_moment_role_and_audience_status_by_clip_id_for(
        editorial_moment_understandings,
    )
    return exact_p1_target_evidence_for(
        target_source_asset_id_by_clip_id,
        editorial_moment_understandings=editorial_moment_understandings,
        watch_listen_understandings=watch_listen_understandings,
        p1_moment_role_by_clip_id=role_by_clip_id,
        p1_audience_delivery_status_by_clip_id=audience_status_by_clip_id,
    )


def _row(result, clip_id):
    rows = {r["clip_id"]: r for r in result["targets"]}
    assert clip_id in rows, f"expected {clip_id!r} in {sorted(rows)}"
    return rows[clip_id]


# ---------------------------------------------------------------------------
# 1. Matching span + moment + confident process role.
# ---------------------------------------------------------------------------
def test_scenario_1_matching_span_moment_confident_process_role():
    moment = _moment("c1", role=MOMENT_ROLE_RECORDING_PROCESS, audience_status=AUDIENCE_DELIVERY_SUPPORTED)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
    assert row["role"] == MOMENT_ROLE_RECORDING_PROCESS
    assert row["helper_lookup_resolved"] is True
    assert row["helper_lookup_role"] == MOMENT_ROLE_RECORDING_PROCESS


# ---------------------------------------------------------------------------
# 2. Matching span + moment + audience role.
# ---------------------------------------------------------------------------
def test_scenario_2_matching_span_moment_audience_role():
    moment = _moment("c1", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, audience_status=AUDIENCE_DELIVERY_SUPPORTED)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
    assert row["role"] == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert row["audience_delivery_status"] == AUDIENCE_DELIVERY_SUPPORTED
    assert row["helper_lookup_audience_delivery_status"] == AUDIENCE_DELIVERY_SUPPORTED


# ---------------------------------------------------------------------------
# 3. Moment exists, confidence low (CONFIDENCE_MIXED).
# ---------------------------------------------------------------------------
def test_scenario_3_moment_exists_low_confidence():
    moment = _moment("c1", confidence=CONFIDENCE_MIXED)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE
    # Seam C's own gate must also have excluded this clip -- consistency.
    assert row["helper_lookup_resolved"] is False


# ---------------------------------------------------------------------------
# 4. Moment exists, role uncertain.
# ---------------------------------------------------------------------------
def test_scenario_4_moment_exists_role_uncertain():
    moment = _moment("c1", role=MOMENT_ROLE_UNCERTAIN)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_ROLE_UNCERTAIN
    assert row["helper_lookup_resolved"] is False


# ---------------------------------------------------------------------------
# 5. Moment exists, audience uncertain.
# ---------------------------------------------------------------------------
def test_scenario_5_moment_exists_audience_uncertain():
    moment = _moment(
        "c1", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, audience_status=AUDIENCE_DELIVERY_UNCERTAIN,
    )
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_AUDIENCE_UNCERTAIN
    # Seam C's own gate does NOT check audience status -- helper can still
    # resolve True here, a DIFFERENT, finer distinction (see docstring).
    assert row["helper_lookup_resolved"] is True


# ---------------------------------------------------------------------------
# 6. No UnderstandingSpan at all for this clip_id.
# ---------------------------------------------------------------------------
def test_scenario_6_no_understanding_span():
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=()),),
        watch_listen_understandings=(_wlu("src1"),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
    assert row["understanding_span_present"] is False
    assert row["editorial_moment_present"] is False


# ---------------------------------------------------------------------------
# 7. UnderstandingSpan exists but no EditorialMoment was built for it.
# ---------------------------------------------------------------------------
def test_scenario_7_span_exists_no_editorial_moment():
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=()),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_EDITORIAL_MOMENT
    assert row["understanding_span_present"] is True
    assert row["editorial_moment_present"] is False


# ---------------------------------------------------------------------------
# 8. A dropped (discarded/losing) candidate is still represented.
# ---------------------------------------------------------------------------
def test_scenario_8_dropped_candidate_still_represented():
    # "Dropped" here means: the moment still exists (P1 sees the full
    # pre-selection candidate pool per D-239N's own Stage 2 proof) even
    # though the caller's own bounded target set includes a clip that
    # never won its family -- exact_p1_target_evidence_for never filters
    # by selection outcome, only by the caller-supplied bounded set.
    dropped_moment = _moment("c_dropped", role=MOMENT_ROLE_RECORDING_PROCESS)
    winner_moment = _moment("c_winner", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
    result = _evidence_for(
        {"c_dropped": "src1", "c_winner": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[dropped_moment, winner_moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c_dropped"), _span("c_winner")),),
    )
    row = _row(result, "c_dropped")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
    assert row["editorial_moment_present"] is True


# ---------------------------------------------------------------------------
# 9. A selected (winning) candidate is also represented.
# ---------------------------------------------------------------------------
def test_scenario_9_selected_candidate_represented():
    dropped_moment = _moment("c_dropped", role=MOMENT_ROLE_RECORDING_PROCESS)
    winner_moment = _moment("c_winner", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
    result = _evidence_for(
        {"c_dropped": "src1", "c_winner": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[dropped_moment, winner_moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c_dropped"), _span("c_winner")),),
    )
    row = _row(result, "c_winner")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
    assert row["role"] == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY


# ---------------------------------------------------------------------------
# 10. Cross-source isolation -- a clip_id present in one source must never
# match a span/moment belonging to a different source_asset_id.
# ---------------------------------------------------------------------------
def test_scenario_10_cross_source_isolation():
    # Same clip_id string minted independently in two different sources --
    # the target is scoped to src2, so ONLY src2's own span/moment may
    # ever resolve it; src1's own same-named span/moment must never leak
    # across the source boundary.
    moment_src1 = _moment("shared_id", source_asset_id="src1", role=MOMENT_ROLE_RECORDING_PROCESS)
    result = _evidence_for(
        {"shared_id": "src2"},
        editorial_moment_understandings=(
            _understanding("src1", moments=[moment_src1]),
            _understanding("src2", moments=()),
        ),
        watch_listen_understandings=(
            _wlu("src1", _span("shared_id", source_asset_id="src1")),
            _wlu("src2"),
        ),
    )
    row = _row(result, "shared_id")
    assert row["source_asset_id"] == "src2"
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
    assert row["understanding_span_present"] is False
    assert row["editorial_moment_present"] is False


# ---------------------------------------------------------------------------
# 11. Exact clip_id / source_span_id equality -- never a fuzzy/substring
# match.
# ---------------------------------------------------------------------------
def test_scenario_11_exact_clip_source_span_equality():
    moment = _moment("clip_abc", role=MOMENT_ROLE_RECORDING_PROCESS)
    result = _evidence_for(
        {"clip_abc": "src1", "clip_ab": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        # "clip_ab" is a strict prefix of "clip_abc" -- must NOT match.
        watch_listen_understandings=(_wlu("src1", _span("clip_abc")),),
    )
    exact_row = _row(result, "clip_abc")
    prefix_row = _row(result, "clip_ab")
    assert exact_row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED
    assert prefix_row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
    assert prefix_row["moment_id"] is None


# ---------------------------------------------------------------------------
# 12. No attempt_id substitution -- correlation is by clip_id/source_span_id
# only, never by an EditorialMoment's own attempt_ids.
# ---------------------------------------------------------------------------
def test_scenario_12_no_attempt_id_substitution():
    # The moment's attempt_ids deliberately reference a DIFFERENT string
    # than the target clip_id -- if the function ever substituted
    # attempt_id for clip_id it would wrongly resolve "attempt_xyz" instead
    # of leaving it unmatched.
    moment = _moment("c1", role=MOMENT_ROLE_RECORDING_PROCESS, attempt_ids=("attempt_xyz",))
    result = _evidence_for(
        {"attempt_xyz": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "attempt_xyz")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
    assert row["moment_id"] is None


# ---------------------------------------------------------------------------
# 13. No aggregate fallback -- a run-wide majority/aggregate role must never
# be substituted for a target with no real evidence of its own.
# ---------------------------------------------------------------------------
def test_scenario_13_no_aggregate_fallback():
    # 9 other moments all share RECORDING_PROCESS_REGION/POST_TAKE_RESET --
    # an overwhelming aggregate majority -- but the target clip itself has
    # no span/moment at all. The target's own row must report NO_
    # UNDERSTANDING_SPAN, never borrow the aggregate's role/status.
    aggregate_moments = [
        _moment(f"other_{i}", role=MOMENT_ROLE_RECORDING_PROCESS) for i in range(9)
    ]
    result = _evidence_for(
        {"target_with_no_evidence": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=aggregate_moments),),
        watch_listen_understandings=(
            _wlu("src1", *[_span(f"other_{i}") for i in range(9)]),
        ),
    )
    row = _row(result, "target_with_no_evidence")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_NO_UNDERSTANDING_SPAN
    assert row["role"] is None
    assert row["recording_process_status"] is None


# ---------------------------------------------------------------------------
# 14. Deterministic output -- identical inputs produce byte-identical
# results, and rows are sorted by clip_id.
# ---------------------------------------------------------------------------
def test_scenario_14_deterministic_output():
    moment_b = _moment("c_b", role=MOMENT_ROLE_RECORDING_PROCESS)
    moment_a = _moment("c_a", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
    args = dict(
        target_source_asset_id_by_clip_id={"c_b": "src1", "c_a": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment_b, moment_a]),),
        watch_listen_understandings=(_wlu("src1", _span("c_b"), _span("c_a")),),
    )
    result1 = _evidence_for(**args)
    result2 = _evidence_for(**args)
    assert result1 == result2
    assert [row["clip_id"] for row in result1["targets"]] == ["c_a", "c_b"]


# ---------------------------------------------------------------------------
# 15. No transcript leakage -- the output rows never carry raw text/
# transcript fields, only identity/status evidence.
# ---------------------------------------------------------------------------
def test_scenario_15_no_transcript_leakage():
    moment = _moment("c1", role=MOMENT_ROLE_RECORDING_PROCESS)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    forbidden_keys = {"text", "transcript", "excerpt", "words", "bounded_excerpt"}
    assert forbidden_keys.isdisjoint(row.keys())
    # Every value in the row is a bounded identity/status primitive -- no
    # arbitrarily long free-text field.
    for value in row.values():
        if isinstance(value, str):
            assert len(value) < 200


# ---------------------------------------------------------------------------
# Additional structural guarantees (ambiguous status + local_group_id +
# allowed-status closure) -- kept alongside the 15 named scenarios above.
# ---------------------------------------------------------------------------
def test_ambiguous_multi_moment_match_is_reported_honestly():
    moment_1 = _moment("c1", moment_id="m1", role=MOMENT_ROLE_RECORDING_PROCESS)
    moment_2 = _moment("c1", moment_id="m2", role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment_1, moment_2]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_AMBIGUOUS


def test_local_group_id_resolved_when_moment_belongs_to_a_group():
    moment = _moment("c1", moment_id="m1", role=MOMENT_ROLE_RECORDING_PROCESS)
    group = EditorialLocalGroup(
        source_asset_id="src1", group_id="grp1", moment_indices=(0,), moment_ids=("m1",),
        source_start=0.0, source_end=1.0, grouping_reason="generic", relation_support=(),
        confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=(),
    )
    result = _evidence_for(
        {"c1": "src1"},
        editorial_moment_understandings=(_understanding("src1", moments=[moment], local_groups=[group]),),
        watch_listen_understandings=(_wlu("src1", _span("c1")),),
    )
    row = _row(result, "c1")
    assert row["local_group_id"] == "grp1"


def test_all_reported_statuses_are_in_the_allowed_closed_set():
    scenarios = [
        ({"c1": "src1"}, (_understanding("src1", moments=[_moment("c1")]),), (_wlu("src1", _span("c1")),)),
        ({"c2": "src1"}, (_understanding("src1", moments=()),), (_wlu("src1"),)),
        ({"c3": "src1"}, (_understanding("src1", moments=()),), (_wlu("src1", _span("c3")),)),
    ]
    for targets, understandings, wlus in scenarios:
        result = _evidence_for(
            targets, editorial_moment_understandings=understandings, watch_listen_understandings=wlus,
        )
        for row in result["targets"]:
            assert row["p1_target_lookup_status"] in ALLOWED_P1_TARGET_LOOKUP_STATUSES


def test_empty_target_set_returns_empty_bounded_output():
    result = _evidence_for(
        {}, editorial_moment_understandings=(), watch_listen_understandings=(),
    )
    assert result["target_count"] == 0
    assert result["targets"] == []
