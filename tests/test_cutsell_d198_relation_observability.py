"""D-198: PRE-RAW RELATION OBSERVABILITY REPAIR -- tests.

Proves the ``relation_to_predecessor`` diagnostic field (D-198) is a pure,
already-computed pass-through -- the SAME value D-197's own grouper
consumed, never re-derived, never re-ranked -- and that adding it changes
NO grouping/classification/authority behavior whatsoever.
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.editorial_moment_sequence_integration import (
    build_editorial_local_groups,
    build_editorial_moment_understanding_for_source,
    build_editorial_moments_for_source,
    editorial_moment_understanding_diagnostics,
)
from cutsell_worker.raw_understanding_map import BehaviorHypothesis
from cutsell_worker.watch_listen_understanding import (
    AttemptRelationHypothesis,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
    MEANING_COMPLETE as WL_MEANING_COMPLETE,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)
from cutsell_worker.raw_understanding_map import BEHAVIOR_CLEAN_ATTEMPT


def _take(clip_id, order, start, end, text="generic abstract statement"):
    return CandidateTake(clip_id=clip_id, source_asset_id="src1", source_order=order, start=start, end=end, text=text)


def _behavior(label, confidence=0.8):
    return BehaviorHypothesis(label=label, confidence=confidence, provenance="VISUAL_SIGNAL", basis="generic")


def _span(span_id, start, end, *, behavior_labels=(), relation=None):
    hyps = tuple(_behavior(label) for label in behavior_labels)
    rel = (AttemptRelationHypothesis(relation, WL_CONFIDENCE_SUPPORTED, "x", None, ()),) if relation else ()
    return UnderstandingSpan(
        span_id=span_id, source_asset_id="src1", source_start=start, source_end=end,
        behavior_state_hypotheses=hyps, behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=rel,
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(*spans):
    return WatchListenUnderstanding(source_asset_id="src1", understanding_spans=tuple(spans))


def _pair_understanding(relation):
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=relation),
    )
    return build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
    )


# ---------------------------------------------------------------------------
# 4-6: RETRY/CORRECTION/CONTINUATION serialize verbatim.
# ---------------------------------------------------------------------------
def test_04_retry_fixture_serializes_retry():
    u = _pair_understanding(RELATION_RETRY)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_RETRY


def test_05_correction_fixture_serializes_correction():
    u = _pair_understanding(RELATION_CORRECTION)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_CORRECTION


def test_06_continuation_fixture_serializes_continuation():
    u = _pair_understanding(RELATION_CONTINUATION)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_CONTINUATION


# ---------------------------------------------------------------------------
# 7-10: split relations observable even though moments land in different
# groups.
# ---------------------------------------------------------------------------
def test_07_new_audience_beat_split_observable_across_groups():
    u = _pair_understanding(RELATION_NEW_AUDIENCE_BEAT)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_NEW_AUDIENCE_BEAT
    # Still two separate groups -- the split relation is preserved even
    # though it never joined anything.
    assert len(u.local_groups) == 2
    assert all(len(g.moment_ids) == 1 for g in u.local_groups)


def test_08_distinct_proposition_split_observable():
    u = _pair_understanding(RELATION_DISTINCT_PROPOSITION)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_DISTINCT_PROPOSITION
    assert len(u.local_groups) == 2


def test_09_complementary_split_observable():
    u = _pair_understanding(RELATION_COMPLEMENTARY)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_COMPLEMENTARY
    assert len(u.local_groups) == 2


def test_10_uncertain_split_observable():
    # This fixture attaches RELATION_UNCERTAIN with real (SUPPORTED)
    # confidence -- the genuine "canonical relation value IS UNCERTAIN"
    # case (D-198's own NULL SEMANTICS: null is reserved for a missing/
    # unavailable relation, not silently substituted for a real UNCERTAIN
    # value). _dominant_relation only collapses UNCERTAIN to null when its
    # own confidence is itself CONFIDENCE_UNKNOWN (test_20 covers that
    # genuinely-missing case).
    u = _pair_understanding(RELATION_UNCERTAIN)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] == RELATION_UNCERTAIN
    assert len(u.local_groups) == 2


# ---------------------------------------------------------------------------
# 11-12: missing relation / first moment serialize null.
# ---------------------------------------------------------------------------
def test_11_missing_relation_serializes_null():
    u = _pair_understanding(None)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] is None


def test_12_first_moment_serializes_null():
    u = _pair_understanding(RELATION_RETRY)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][0]["relation_to_predecessor"] is None


# ---------------------------------------------------------------------------
# 13-14: group/sequence membership unchanged before vs after (regression
# proof against D-197's own already-established behavior).
# ---------------------------------------------------------------------------
def test_13_group_membership_unchanged():
    u = _pair_understanding(RELATION_RETRY)
    assert len(u.local_groups) == 1
    assert len(u.local_groups[0].moment_ids) == 2


def test_14_sequence_membership_unchanged():
    u = _pair_understanding(RELATION_RETRY)
    assert len(u.sequence_hypotheses) == 1
    assert len(u.sequence_hypotheses[0].moment_ids) == 2


# ---------------------------------------------------------------------------
# 15: deterministic serialization.
# ---------------------------------------------------------------------------
def test_15_deterministic_serialization():
    u1 = _pair_understanding(RELATION_CONTINUATION)
    u2 = _pair_understanding(RELATION_CONTINUATION)
    d1 = editorial_moment_understanding_diagnostics(u1)
    d2 = editorial_moment_understanding_diagnostics(u2)
    assert [m["relation_to_predecessor"] for m in d1["moments"]] == [m["relation_to_predecessor"] for m in d2["moments"]]


# ---------------------------------------------------------------------------
# 16: no transcript dump.
# ---------------------------------------------------------------------------
def test_16_no_transcript_dump():
    takes = [_take("c1", 0, 0.0, 1.0, text="a very specific real transcript sentence"), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    d = editorial_moment_understanding_diagnostics(u)
    assert "a very specific real transcript sentence" not in str(d)


# ---------------------------------------------------------------------------
# Single source of truth: the serialized value is the SAME object the
# grouper consumed -- never a second lookup/re-derivation.
# ---------------------------------------------------------------------------
def test_17_single_source_of_truth_matches_grouper_input():
    takes = [_take(f"c{i}", i, float(i), float(i) + 1.0) for i in range(4)]
    wlu = _wlu(
        _span("c0", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c1", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
        _span("c2", 2.0, 3.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_NEW_AUDIENCE_BEAT),
        _span("c3", 3.0, 4.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_CORRECTION),
    )
    understanding_spans_by_id = {s.span_id: s for s in wlu.understanding_spans}
    moments, _u, _f, relation_by_position = build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=takes, understanding_spans_by_id=understanding_spans_by_id,
    )
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relation_by_position)

    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
    )
    d = editorial_moment_understanding_diagnostics(u)
    serialized = [m["relation_to_predecessor"] for m in d["moments"]]
    expected = [relation_by_position.get(i) for i in range(len(moments))]
    assert serialized == expected
    # And the grouper's own real output is unaffected by this serialization.
    assert [tuple(sorted(g.moment_indices)) for g in u.local_groups] == [tuple(sorted(g.moment_indices)) for g in groups]


# ---------------------------------------------------------------------------
# Consistency audit: relation_to_predecessor + group membership explains
# the grouping result, per D-198's own required table.
# ---------------------------------------------------------------------------
def test_18_consistency_audit_join_relations_same_group():
    for relation in (RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION):
        u = _pair_understanding(relation)
        d = editorial_moment_understanding_diagnostics(u)
        assert d["moments"][1]["relation_to_predecessor"] == relation
        assert len(u.local_groups) == 1, f"{relation} should join into one group"


def test_19_consistency_audit_split_relations_different_group():
    for relation in (RELATION_NEW_AUDIENCE_BEAT, RELATION_DISTINCT_PROPOSITION, RELATION_COMPLEMENTARY):
        u = _pair_understanding(relation)
        d = editorial_moment_understanding_diagnostics(u)
        assert d["moments"][1]["relation_to_predecessor"] == relation
        assert len(u.local_groups) == 2, f"{relation} should split into two groups"


def test_20_consistency_audit_missing_relation_different_group():
    u = _pair_understanding(None)
    assert len(u.local_groups) == 2


def test_20b_genuine_unknown_confidence_uncertain_collapses_to_null():
    from cutsell_worker.watch_listen_understanding import CONFIDENCE_UNKNOWN as WL_CONFIDENCE_UNKNOWN

    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    span2 = _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT])
    from dataclasses import replace
    span2 = replace(span2, attempt_relation_hypotheses=(
        AttemptRelationHypothesis(RELATION_UNCERTAIN, WL_CONFIDENCE_UNKNOWN, "insufficient evidence", None, ()),
    ))
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]), span2)
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    d = editorial_moment_understanding_diagnostics(u)
    assert d["moments"][1]["relation_to_predecessor"] is None
    assert len(u.local_groups) == 2


# ---------------------------------------------------------------------------
# Default-off parity / flag-on immutability (pipeline level) -- re-run
# green after the observability repair.
# ---------------------------------------------------------------------------
def _pipeline_fixture():
    from cutsell_worker.contracts import MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole

    weak = CandidateTake(
        clip_id="weak", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.3, eye_contact=0.2),
    )
    strong = CandidateTake(
        clip_id="strong", source_asset_id="src", source_order=0, start=4.0, end=6.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=4.0, end=6.0, audio_quality=0.95, eye_contact=0.95),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    return request, (weak, strong), labels, strong.clip_id


def test_01_default_off_byte_equivalent(monkeypatch):
    from cutsell_worker.pipeline import build_flow_b_draft

    monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
    request, takes, labels, expected_winner = _pipeline_fixture()
    result = build_flow_b_draft(request, takes, labels)
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]
    assert result.draft.diagnostics["editorial_moment_sequence"] == {"status": "disabled"}


def test_02_03_flag_on_winner_family_immutability(monkeypatch):
    from cutsell_worker.pipeline import build_flow_b_draft

    request, takes, labels, expected_winner = _pipeline_fixture()
    wlu = WatchListenUnderstanding(source_asset_id="src", understanding_spans=(
        UnderstandingSpan(
            span_id="weak", source_asset_id="src", source_start=1.0, source_end=3.0,
            behavior_state_hypotheses=(_behavior(BEHAVIOR_CLEAN_ATTEMPT),), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
            attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(), relation_confidence=WL_CONFIDENCE_SUPPORTED,
            meaning_completion_hypothesis=WL_MEANING_COMPLETE, performance_usability_hypothesis=USABILITY_USABLE,
            entry_usability=USABILITY_USABLE, delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
            conflict_flags=(), evidence_provenance={},
        ),
        UnderstandingSpan(
            span_id="strong", source_asset_id="src", source_start=4.0, source_end=6.0,
            behavior_state_hypotheses=(_behavior(BEHAVIOR_CLEAN_ATTEMPT),), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
            attempt_boundary_hypotheses=(),
            attempt_relation_hypotheses=(AttemptRelationHypothesis(RELATION_RETRY, WL_CONFIDENCE_SUPPORTED, "x", "weak", ()),),
            relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
            performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
            delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
            conflict_flags=(), evidence_provenance={},
        ),
    ))
    monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", "1")
    result = build_flow_b_draft(request, takes, labels, watch_listen_understandings=(wlu,))

    assert [c.clip_id for c in result.draft.selected] == [expected_winner]
    p1 = result.draft.diagnostics["editorial_moment_sequence"]
    assert p1["status"] == "evaluated"
    for row in p1["moments"]:
        assert "relation_to_predecessor" in row
    assert p1["moments"][1]["relation_to_predecessor"] == RELATION_RETRY
    assert p1["moments"][0]["relation_to_predecessor"] is None
