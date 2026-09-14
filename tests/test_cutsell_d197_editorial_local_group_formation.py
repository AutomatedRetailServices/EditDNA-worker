"""D-197: P1 LOCAL SEQUENCE GROUP FORMATION -- tests.

Generic, abstract fixtures only -- NO Video00 literal text/spans/ids.
Covers the directive's offline test matrix: the deterministic grouper
itself (join/boundary relation semantics, singletons, chains, multiple
unrelated clusters, the D-196 abstract replay), determinism/order/id
independence, no-magic-time-threshold source scan, and the live-adapter
wiring (auto-grouper replaces the old whole-source default; explicit
overrides still work; default-off/flag-on immutability untouched).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.editorial_moment_sequence import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    SEQUENCE_KIND_BLOOPER_SERIES,
    SEQUENCE_KIND_RETRY_SERIES,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    ALLOWED_GROUPING_REASONS,
    GROUP_REASON_ISOLATED_NO_RELATION_EVIDENCE,
    GROUP_REASON_RELATION_LINKED_CHAIN,
    GROUP_REASON_SOLE_MOMENT_IN_SOURCE,
    EditorialLocalGroup,
    build_editorial_local_groups,
    build_editorial_moment_understanding_for_source,
    build_editorial_moments_for_source,
    build_editorial_sequences_for_moments,
    editorial_local_group_diagnostics,
    editorial_moment_understanding_run_summary,
)
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_CLEAN_ATTEMPT,
    BehaviorHypothesis,
)
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
    MEANING_INCOMPLETE as WL_MEANING_INCOMPLETE,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "cutsell_worker" / "editorial_moment_sequence_integration.py"


# ---------------------------------------------------------------------------
# Generic fixture factories (same shape as the D-195 suite's own helpers).
# ---------------------------------------------------------------------------
def _take(clip_id, order, start, end, text="generic abstract statement"):
    return CandidateTake(clip_id=clip_id, source_asset_id="src1", source_order=order, start=start, end=end, text=text)


def _behavior(label, confidence=0.8):
    return BehaviorHypothesis(label=label, confidence=confidence, provenance="VISUAL_SIGNAL", basis="generic")


def _span(span_id, start, end, *, behavior_labels=(), meaning=WL_MEANING_COMPLETE, relation=None):
    hyps = tuple(_behavior(label) for label in behavior_labels)
    rel = (AttemptRelationHypothesis(relation, WL_CONFIDENCE_SUPPORTED, "x", None, ()),) if relation else ()
    return UnderstandingSpan(
        span_id=span_id, source_asset_id="src1", source_start=start, source_end=end,
        behavior_state_hypotheses=hyps, behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=rel,
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=meaning,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(*spans):
    return WatchListenUnderstanding(source_asset_id="src1", understanding_spans=tuple(spans))


def _moments_and_relations(n, relations_by_position):
    """Builds ``n`` generic clean moments in one source, with an explicit
    relation-to-predecessor at the given positions (1-indexed gaps allowed
    -- position 0 never has a predecessor)."""
    takes = [_take(f"c{i}", i, float(i), float(i) + 1.0) for i in range(n)]
    spans = [
        _span(f"c{i}", float(i), float(i) + 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=relations_by_position.get(i))
        for i in range(n)
    ]
    understanding_spans_by_id = {s.span_id: s for s in spans}
    moments, _unresolved, _fallback, relation_by_position = build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=takes, understanding_spans_by_id=understanding_spans_by_id,
    )
    return moments, relation_by_position


# ---------------------------------------------------------------------------
# 1-6: join relation semantics (RETRY/CORRECTION/CONTINUATION join;
# COMPLEMENTARY/NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION/UNCERTAIN/missing do
# not).
# ---------------------------------------------------------------------------
def test_01_one_related_pair_forms_one_group():
    moments, relations = _moments_and_relations(2, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert len(groups[0].moment_ids) == 2
    assert groups[0].grouping_reason == GROUP_REASON_RELATION_LINKED_CHAIN


@pytest.mark.parametrize("relation", [RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION])
def test_02_04_join_relations(relation):
    moments, relations = _moments_and_relations(2, {1: relation})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert len(groups[0].moment_ids) == 2
    assert relation in groups[0].relation_support


@pytest.mark.parametrize("relation", [
    RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT, RELATION_DISTINCT_PROPOSITION, RELATION_UNCERTAIN,
])
def test_05_boundary_relations_do_not_join(relation):
    moments, relations = _moments_and_relations(2, {1: relation})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 2
    assert all(len(g.moment_ids) == 1 for g in groups)


def test_06_no_relation_does_not_blindly_join():
    moments, relations = _moments_and_relations(2, {})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 2
    assert all(len(g.moment_ids) == 1 for g in groups)
    assert all(g.grouping_reason == GROUP_REASON_ISOLATED_NO_RELATION_EVIDENCE for g in groups)


def test_07_ambiguous_relation_does_not_bridge():
    # A/B linked by CONTINUATION, B/C by UNCERTAIN -- C must NOT join {A,B}.
    moments, relations = _moments_and_relations(3, {1: RELATION_CONTINUATION, 2: RELATION_UNCERTAIN})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 2
    sizes = sorted(len(g.moment_ids) for g in groups)
    assert sizes == [1, 2]


def test_08_conflicting_weak_relation_does_not_bridge():
    # COMPLEMENTARY is explicitly hedged/never-confirmed evidence -- must
    # never bridge two otherwise-unrelated moments either.
    moments, relations = _moments_and_relations(3, {1: RELATION_RETRY, 2: RELATION_COMPLEMENTARY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 2
    sizes = sorted(len(g.moment_ids) for g in groups)
    assert sizes == [1, 2]


# ---------------------------------------------------------------------------
# 9-18: chain shapes, singletons, multiple unrelated clusters, D-196 replay.
# ---------------------------------------------------------------------------
def test_09_source_chronology_preserved():
    moments, relations = _moments_and_relations(3, {1: RELATION_RETRY, 2: RELATION_CONTINUATION})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert list(groups[0].moment_indices) == sorted(groups[0].moment_indices)


def test_10_input_order_independence_via_builder():
    takes_fwd = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    takes_rev = list(reversed(takes_fwd))
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u_fwd = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes_fwd, watch_listen_understanding=wlu)
    u_rev = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes_rev, watch_listen_understanding=wlu)
    assert [g.group_id for g in u_fwd.local_groups] == [g.group_id for g in u_rev.local_groups]


def test_11_group_id_independent_of_clip_id_choice():
    # Two structurally identical sources with DIFFERENT clip ids must
    # still hash to different group ids (id is source+membership-anchored,
    # by construction) -- but the SAME clip ids, called twice, are stable.
    moments_a, relations_a = _moments_and_relations(2, {1: RELATION_RETRY})
    groups_a1 = build_editorial_local_groups(moments_a, relation_candidates_by_position=relations_a)
    groups_a2 = build_editorial_local_groups(moments_a, relation_candidates_by_position=relations_a)
    assert [g.group_id for g in groups_a1] == [g.group_id for g in groups_a2]


def test_12_no_family_id_reference():
    tree = ast.parse(MODULE_PATH.read_text())
    src = MODULE_PATH.read_text()
    assert "retry_family_id" not in src
    assert "take_group_id" not in src
    del tree


def test_13_source_id_separation_never_crosses():
    # build_editorial_local_groups only ever receives one source's moments
    # (build_editorial_moments_for_source already filters to one
    # source_asset_id) -- every emitted group's source_asset_id matches.
    moments, relations = _moments_and_relations(3, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert all(g.source_asset_id == "src1" for g in groups)


def test_14_singleton_retained_first_moment():
    moments, relations = _moments_and_relations(1, {})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert len(groups[0].moment_ids) == 1
    assert groups[0].grouping_reason == GROUP_REASON_SOLE_MOMENT_IN_SOURCE


def test_15_two_moment_group():
    moments, relations = _moments_and_relations(2, {1: RELATION_CORRECTION})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert len(groups[0].moment_ids) == 2


def test_16_five_moment_related_chain():
    moments, relations = _moments_and_relations(5, {1: RELATION_RETRY, 2: RELATION_CONTINUATION, 3: RELATION_CONTINUATION, 4: RELATION_CORRECTION})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1
    assert len(groups[0].moment_ids) == 5
    assert set(groups[0].relation_support) == {RELATION_RETRY, RELATION_CONTINUATION, RELATION_CORRECTION}


def test_17_multiple_unrelated_clusters():
    # cluster A (0,1) retry-joined; moment 2 isolated; cluster B (3,4)
    # correction-joined -- a relation at position i describes i's relation
    # to its predecessor i-1, so the join for {3,4} is set at position 4.
    moments, relations = _moments_and_relations(5, {1: RELATION_RETRY, 4: RELATION_CORRECTION})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    sizes = sorted(len(g.moment_ids) for g in groups)
    assert sizes == [1, 2, 2]


def test_18_d196_abstract_replay_no_whole_source_collapse():
    """Generic, abstract fixture structurally analogous to D-196's real
    finding: 30+ moments, several unrelated idea regions, relation-linked
    local clusters separated by NEW_AUDIENCE_BEAT/no-relation. The OLD
    default (whole source = one sequence) would have produced exactly ONE
    36-ish-moment EditorialSequenceHypothesis; D-197 must produce several
    bounded local groups instead -- no literal Video00 spans/text/region
    names anywhere in this fixture."""
    n = 33
    relations_by_position: dict[int, str] = {}
    # Five clusters of ~6-7 moments each, joined internally by CONTINUATION,
    # separated by an explicit NEW_AUDIENCE_BEAT boundary.
    cluster_starts = [0, 7, 14, 21, 28]
    for start in cluster_starts:
        for i in range(start + 1, min(start + 7, n)):
            relations_by_position[i] = RELATION_CONTINUATION
    for start in cluster_starts[1:]:
        relations_by_position[start] = RELATION_NEW_AUDIENCE_BEAT  # explicit boundary, overrides the loop above
    moments, relations = _moments_and_relations(n, relations_by_position)
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) >= 5  # never one whole-source collapse
    assert max(len(g.moment_ids) for g in groups) < n  # no single group spans (almost) everything


def test_19_no_whole_source_fallback_via_live_builder():
    takes = [_take(f"c{i}", i, float(i), float(i) + 1.0) for i in range(10)]
    # Five isolated pairs (retry-joined), no relation at all between pairs.
    spans = []
    for i in range(10):
        relation = RELATION_RETRY if i % 2 == 1 else None
        spans.append(_span(f"c{i}", float(i), float(i) + 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=relation))
    wlu = _wlu(*spans)
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert len(u.sequence_hypotheses) == 5  # five bounded 2-moment sequences, never one 10-moment sequence
    assert all(len(s.moment_ids) == 2 for s in u.sequence_hypotheses)


# ---------------------------------------------------------------------------
# 20-28: no magic time threshold, exact timing, determinism, transcript,
# confidence, conflict, provenance.
# ---------------------------------------------------------------------------
def test_20_no_magic_time_constants_in_source():
    src = MODULE_PATH.read_text()
    for forbidden in ("MAX_GAP_SECONDS", "LOCAL_WINDOW_SECONDS", "2.0 second", "3.0 second"):
        assert forbidden not in src
    # No new bare numeric-seconds literal introduced by the D-197 section.
    d197_start = src.index("# D-197: P1 LOCAL SEQUENCE GROUP FORMATION")
    d197_end = src.index("def build_editorial_sequences_for_moments")
    d197_body = src[d197_start:d197_end]
    for token in ("1.20", "1.2)", " 2.0", " 3.0"):
        assert token not in d197_body


def test_21_exact_source_timing():
    moments, relations = _moments_and_relations(2, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert groups[0].source_start == 0.0
    assert groups[0].source_end == 2.0


def test_22_23_deterministic_group_ids_and_membership():
    moments, relations = _moments_and_relations(3, {1: RELATION_RETRY, 2: RELATION_CONTINUATION})
    g1 = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    g2 = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert [g.group_id for g in g1] == [g.group_id for g in g2]
    assert [g.moment_ids for g in g1] == [g.moment_ids for g in g2]


def test_24_deterministic_sequence_membership_via_builder():
    takes, wlu = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)], None
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u1 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    u2 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert [s.sequence_id for s in u1.sequence_hypotheses] == [s.sequence_id for s in u2.sequence_hypotheses]


def test_25_no_transcript_in_group_diagnostics():
    takes = [_take("c1", 0, 0.0, 1.0, text="a very specific real transcript sentence"), _take("c2", 1, 1.0, 2.0, text="another one")]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    for g in u.local_groups:
        d = editorial_local_group_diagnostics(g)
        assert "a very specific real transcript sentence" not in str(d)
        assert "text" not in d and "transcript" not in d


def test_26_confidence_categorical_only():
    moments, relations = _moments_and_relations(2, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert groups[0].confidence in {CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_MIXED, CONFIDENCE_UNKNOWN}
    singleton_groups = build_editorial_local_groups((moments[0],))
    assert singleton_groups[0].confidence == CONFIDENCE_UNKNOWN


def test_27_conflict_retained_in_group():
    class _FakeProsodic:
        vocal_continuity_state = "FRAGMENTED"

    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        prosodic_evidence_by_span_id={"c1": _FakeProsodic()},
    )
    assert u.local_groups[0].conflict_flags
    assert u.local_groups[0].confidence == CONFIDENCE_MIXED


def test_28_grouping_provenance_populated():
    moments, relations = _moments_and_relations(2, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert "RELATION_EVIDENCE" in groups[0].provenance
    assert "EDITORIAL_MOMENT_CLASSIFICATION" in groups[0].provenance


# ---------------------------------------------------------------------------
# 29-33: sequence kinds remain reachable end to end; clean unrelated takes
# never become final.
# ---------------------------------------------------------------------------
def test_29_retry_series_remains_reachable():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_RETRY_SERIES


def test_30_blooper_series_remains_reachable():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_BREAKING_CHARACTER]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_BLOOPER_SERIES


def test_31_clean_unrelated_takes_never_become_final():
    takes = [_take(f"c{i}", i, float(i), float(i) + 1.0) for i in range(4)]
    wlu = _wlu(*[_span(f"c{i}", float(i), float(i) + 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]) for i in range(4)])
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses == ()
    assert all(m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY for m in u.moments)


# ---------------------------------------------------------------------------
# 34-37: no chronology-final rule, no proposition invention, fallback
# LanguageAttempt still supported.
# ---------------------------------------------------------------------------
def test_34_no_gap_length_read_by_grouper():
    """No CODE line (comments/docstrings aside -- this section's own audit
    comment quotes D-157's real relation-derivation docstrings verbatim,
    which legitimately mention gap length) in D-197's grouper reads a gap/
    duration/timestamp value to decide a join -- only the categorical
    relation vocabulary."""
    src = MODULE_PATH.read_text()
    d197_start = src.index("# D-197: P1 LOCAL SEQUENCE GROUP FORMATION")
    d197_end = src.index("def build_editorial_sequences_for_moments")
    code_lines = [
        line for line in src[d197_start:d197_end].splitlines()
        if line.strip() and not line.strip().startswith("#") and '"""' not in line
    ]
    # Exclude lines still inside the module-level docstring-style comment
    # block (all '#'-prefixed) -- only genuine statements remain here.
    for line in code_lines:
        assert "gap" not in line.lower()


def test_35_36_no_proposition_invention():
    src = MODULE_PATH.read_text()
    d197_start = src.index("# D-197: P1 LOCAL SEQUENCE GROUP FORMATION")
    d197_end = src.index("def build_editorial_sequences_for_moments")
    assert "PropositionCandidate" not in src[d197_start:d197_end]


def test_37_fallback_language_attempt_still_supported():
    # No real LanguageAttempt supplied anywhere -- grouping still operates
    # purely on relation_candidates_by_position + moments.
    moments, relations = _moments_and_relations(2, {1: RELATION_RETRY})
    groups = build_editorial_local_groups(moments, relation_candidates_by_position=relations)
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# 44-53 (shared with D-195's own module-level scans): re-confirm the SAME
# forbidden-authority-string scan still passes with D-197's additions in
# the same module (the D-195 suite's own test_44_to_50 already covers this
# file end to end -- this is a targeted re-check scoped to the new section).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "take_group_id", "family_complete_context", "selected_clip_id", "_semantic_best_take",
    "bounded_finalist_authority", "bounded_finalist_arbiter", "winner_after",
    "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
    "render_plan", "RenderSegment", "canonical_edit_plan",
])
def test_44_53_no_authority_mutation_strings_in_d197_section(forbidden):
    src = MODULE_PATH.read_text()
    d197_start = src.index("# D-197: P1 LOCAL SEQUENCE GROUP FORMATION")
    d197_end = src.index("def build_editorial_sequences_for_moments")
    assert forbidden not in src[d197_start:d197_end]


# ---------------------------------------------------------------------------
# 54-55: default-off parity / flag-on immutability (pipeline level).
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


def test_38_default_off_byte_equivalent(monkeypatch):
    from cutsell_worker.pipeline import build_flow_b_draft

    monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
    request, takes, labels, expected_winner = _pipeline_fixture()
    result = build_flow_b_draft(request, takes, labels)
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]
    assert result.draft.diagnostics["editorial_moment_sequence"] == {"status": "disabled"}


def test_39_flag_on_local_groups_surfaced_winner_unchanged(monkeypatch):
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
    assert "local_groups" in p1
    assert len(p1["local_groups"]) >= 1
    group_required = {
        "group_id", "source_asset_id", "source_start", "source_end", "moment_ids", "moment_count",
        "grouping_reason", "relation_support", "confidence", "conflict", "provenance",
    }
    for row in p1["local_groups"]:
        assert group_required.issubset(row.keys())
        assert row["grouping_reason"] in ALLOWED_GROUPING_REASONS
        assert "text" not in row and "transcript" not in row
    for key in (
        "local_group_count", "singleton_group_count", "multi_moment_group_count",
        "max_group_moment_count", "sequence_count", "sequence_from_supported_group_count",
        "unsequenced_moment_count",
    ):
        assert key in p1


# ---------------------------------------------------------------------------
# 56-60: run summary shape, runtime, capability, no-recomputation, structure.
# ---------------------------------------------------------------------------
def test_40_run_summary_local_group_keys():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    summary = editorial_moment_understanding_run_summary([u])
    assert summary["local_group_count"] == 1
    assert summary["multi_moment_group_count"] == 1
    assert summary["singleton_group_count"] == 0
    assert summary["max_group_moment_count"] == 2
    assert summary["unsequenced_moment_count"] == 0


def test_41_explicit_local_groups_override_bypasses_auto_grouper():
    # An explicit override (a future, separately-authorized caller with
    # independent grouping evidence) still works exactly as before D-197.
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu, local_groups=[[0, 1]],
    )
    assert len(u.sequence_hypotheses) == 1
    assert u.local_groups == ()  # bypass path reports no auto-computed groups


def test_42_no_asr_visual_prosodic_import_still_holds():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    for forbidden_module in ("asr", "local_performance", "prosodic_audio_v2", "audio_silence", "whole_video_openai"):
        assert forbidden_module not in imported
