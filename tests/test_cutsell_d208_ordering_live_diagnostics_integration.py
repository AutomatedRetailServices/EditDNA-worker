"""D-208: Ordering Live Diagnostic Integration -- offline tests.

Covers: the pure live-integration builder (unit construction from real
``DraftClip``-shaped realizations, P1 local-sequence/continuation/
correction reuse, P2 reuse/optionality, retry-survival conflict,
composite integrity, multi-source behavior, deterministic baseline,
existing-composer compatibility path + D-207 validator reuse,
capability-status ladder, diagnostics/run-summary shape), pipeline-level
wiring (default-off byte parity, flag-on immutability, real live P1/P2
evidence end to end, determinism), and structural no-recompute/no-
authority/no-provider/no-QA/no-Family/BestTake/Boundary/Pacing/Renderer
audits (D-205/D-206/D-207 consolidation replay).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker.contracts import CandidateTake, DraftClip, MediaSignals, SemanticRole
from cutsell_worker.editorial_moment_sequence import (
    CONFIDENCE_SUPPORTED,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_CORRECTION,
    AUDIENCE_DELIVERY_SUPPORTED,
    RECORDING_PROCESS_ABSENT,
    EditorialMoment,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    EditorialLocalGroup,
    EditorialMomentUnderstanding,
)
from cutsell_worker.whole_video_editorial_reasoning import (
    CONTINUITY_UNKNOWN,
    SUPERSESSION_CONFLICTED,
    SUPERSESSION_PARTIAL,
    SUPERSESSION_SUPPORTED,
    WholeVideoEditorialRegion,
    WholeVideoEditorialUnderstanding,
    WholeVideoSupersessionHypothesis,
)
from cutsell_worker import ordering_composer_adapter as oca
from cutsell_worker import ordering_realization_plan as ordp
from cutsell_worker import ordering_live_diagnostics_integration as olive

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "ordering_live_diagnostics_integration.py").read_text()


def _code_only(source: str) -> str:
    marker = '"""\nfrom __future__'
    idx = source.find(marker)
    return source[idx + len(marker):] if idx != -1 else source


CODE_ONLY = _code_only(MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Fixture helpers (same conventions as test_cutsell_d206/d207).
# ---------------------------------------------------------------------------
def _draft(clip_id, source_asset_id, source_order, start, end, *, realization_id=None, source_span_id=None):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=source_order,
        start=start, end=end, text="x", caption_text="x",
        semantic_role=SemanticRole.OTHER, selected=True,
        signals=MediaSignals(source_asset_id=source_asset_id, start=start, end=end),
        realization_id=realization_id, source_span_id=source_span_id or clip_id,
    )


def _moment(source_asset_id, seed, start, end, role, *, proposition_ids=(), source_span_id=None):
    return EditorialMoment(
        source_asset_id=source_asset_id, editorial_moment_id=f"emom_{seed}", source_start=start, source_end=end,
        source_span_id=source_span_id or seed, attempt_ids=(f"att_{seed}",),
        proposition_candidate_ids=tuple(proposition_ids), related_span_ids=(), moment_role=role,
        audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED, recording_process_status=RECORDING_PROCESS_ABSENT,
        completion_status="COMPLETE", local_sequence_position=None, confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _local_group(source_asset_id, seed, moment_ids):
    return EditorialLocalGroup(
        source_asset_id=source_asset_id, group_id=f"elgrp_{seed}", moment_indices=tuple(range(len(moment_ids))),
        moment_ids=tuple(moment_ids), source_start=0.0, source_end=1.0,
        grouping_reason="RELATION_LINKED_CHAIN", relation_support=(), confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _region(seed, moment_ids, start, end, proposition_ids=()):
    return WholeVideoEditorialRegion(
        source_asset_id="src1", region_id=f"wver_{seed}", moment_ids=tuple(moment_ids), local_group_ids=(f"elgrp_{seed}",),
        sequence_ids=(), source_start=start, source_end=end, dominant_process_status="CLEAN_DELIVERY",
        audience_delivery_status="SUPPORTED", proposition_candidate_ids=tuple(proposition_ids),
        confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _p2_understanding(regions, hypotheses):
    return WholeVideoEditorialUnderstanding(
        source_asset_ids=("src1",), regions=tuple(regions), proposition_realization_maps=(),
        cross_source_proposition_links=(), supersession_hypotheses=tuple(hypotheses),
        global_continuity_status=CONTINUITY_UNKNOWN, unresolved_conflicts=(), capability_status="AVAILABLE",
        confidence=CONFIDENCE_SUPPORTED, provenance=("TEST_FIXTURE",),
    )


def _p1_understanding(source_asset_id, moments, local_groups=(), sequence_hypotheses=()):
    return EditorialMomentUnderstanding(
        source_asset_id=source_asset_id, moments=tuple(moments), sequence_hypotheses=tuple(sequence_hypotheses),
        moment_count=len(moments), sequence_count=len(sequence_hypotheses), capability_status="AVAILABLE",
        missing_evidence=(), confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
        local_groups=tuple(local_groups),
    )


# ---------------------------------------------------------------------------
# 1. Flag function.
# ---------------------------------------------------------------------------
class TestFlag:
    def test_default_off(self):
        assert olive.ordering_diagnostics_enabled(env={}) is False

    def test_on_variants(self):
        for value in ("1", "true", "True", "yes", "on"):
            assert olive.ordering_diagnostics_enabled(env={"CUTSELL_ORDERING_DIAGNOSTICS_ENABLED": value}) is True

    def test_off_variants(self):
        for value in ("0", "false", "no", "off", "garbage"):
            assert olive.ordering_diagnostics_enabled(env={"CUTSELL_ORDERING_DIAGNOSTICS_ENABLED": value}) is False


# ---------------------------------------------------------------------------
# 2. Missing-input behavior.
# ---------------------------------------------------------------------------
class TestMissingInput:
    def test_no_selected_realizations_not_evaluable(self):
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=(), p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        assert result.capability_status == olive.CAPABILITY_NOT_EVALUABLE
        assert olive.MISSING_NO_SELECTED_REALIZATIONS in result.missing_evidence
        assert result.baseline_plan is None
        assert result.composer_proposal is None


# ---------------------------------------------------------------------------
# 3. Fixture A -- two same-source realizations, no editorial constraint.
# ---------------------------------------------------------------------------
class TestFixtureA:
    def test_source_order_fallback_partial_membership_preserved(self):
        clips = [_draft("a", "src1", 0, 0.0, 1.0), _draft("b", "src1", 1, 1.0, 2.0)]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        assert result.capability_status == olive.CAPABILITY_PARTIAL
        assert olive.MISSING_P1_UNAVAILABLE in result.missing_evidence
        assert olive.MISSING_P2_UNAVAILABLE in result.missing_evidence
        assert result.baseline_plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN
        assert set(result.baseline_plan.ordered_realization_ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. Fixture B -- P1 continuation.
# ---------------------------------------------------------------------------
class TestFixtureB:
    def test_positive_order_constraint(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        understanding = _p1_understanding("src1", [m1, m2], [g])
        clips = [
            _draft("b", "src1", 1, 2.0, 3.0, source_span_id="s2"),
            _draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1"),
        ]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding],
            p1_diagnostics_enabled=True, p2_diagnostics_enabled=False,
        )
        rel = next(r for r in result.relations if r.ordering_reason == ordp.REASON_P1_CONTINUATION)
        assert rel.ordering_relation == ordp.RELATION_MUST_PRECEDE
        assert result.baseline_plan.ordered_realization_ids == ("a", "b")


# ---------------------------------------------------------------------------
# 5. Fixture C -- correction dependency.
# ---------------------------------------------------------------------------
class TestFixtureC:
    def test_correction_dependency_preserved(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CORRECTION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        understanding = _p1_understanding("src1", [m1, m2], [g])
        clips = [_draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _draft("b", "src1", 1, 2.0, 3.0, source_span_id="s2")]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding],
            p1_diagnostics_enabled=True, p2_diagnostics_enabled=False,
        )
        rel = next(r for r in result.relations if r.ordering_reason == ordp.REASON_P1_CORRECTION)
        assert rel.ordering_relation == ordp.RELATION_MUST_PRECEDE
        assert set(result.baseline_plan.ordered_realization_ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# 6. Fixture D -- retry alternatives both frozen (P2 supersession-survival).
# ---------------------------------------------------------------------------
class TestFixtureD:
    def test_retry_survival_conflict_no_winner_picked(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2")
        understanding_p1 = _p1_understanding("src1", [m1, m2])
        region_a = _region("g1", [m1.editorial_moment_id], 0.0, 1.0)
        region_b = _region("g2", [m2.editorial_moment_id], 10.0, 11.0)
        # Both alternatives frozen -- a real, unresolved retry-survival
        # case: neither side was dropped upstream, so P2 reports
        # SUPPORTED supersession-survival (D-206's own established
        # handling), never a deletion.
        hyp = WholeVideoSupersessionHypothesis(
            source_asset_id="src1", supersession_id="wvsup_1", earlier_region_ids=(region_a.region_id,),
            later_region_ids=(region_b.region_id,), covered_proposition_candidate_ids=("p1",),
            uncovered_earlier_proposition_candidate_ids=(), coverage_status="FULL_COVERAGE",
            recording_process_support="SUPPORTED", audience_delivery_support="SUPPORTED",
            meaning_conflict_status="MEANING_CONFLICT_NONE", supersession_status=SUPERSESSION_SUPPORTED,
            confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
        )
        p2 = _p2_understanding([region_a, region_b], [hyp])
        clips = [_draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _draft("b", "src1", 1, 10.0, 11.0, source_span_id="s2")]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding_p1],
            whole_video_editorial_reasoning_result=p2, p1_diagnostics_enabled=True, p2_diagnostics_enabled=True,
        )
        # Ordering reports the conflict; it never drops a unit or picks a winner.
        assert set(result.baseline_plan.ordered_realization_ids) == {"a", "b"}
        assert result.baseline_plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        assert any("SUPERSESSION_SURVIVAL_CONFLICT" in r.conflict_flags for r in result.relations)
        summary = olive.ordering_live_diagnostics_run_summary(result)
        assert summary["retry_survival_conflict_count"] == 1


# ---------------------------------------------------------------------------
# 7. Fixture E -- P1 local sequence with two surviving units.
# ---------------------------------------------------------------------------
class TestFixtureE:
    def test_local_sequence_order_preserved(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        understanding = _p1_understanding("src1", [m1, m2], [g])
        clips = [_draft("b", "src1", 1, 2.0, 3.0, source_span_id="s2"), _draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1")]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding],
            p1_diagnostics_enabled=True, p2_diagnostics_enabled=False,
        )
        assert result.baseline_plan.ordered_realization_ids == ("a", "b")
        assert result.baseline_plan.ordering_status == ordp.ORDERING_STATUS_ORDERED


# ---------------------------------------------------------------------------
# 8. Fixture F -- composite unit.
# ---------------------------------------------------------------------------
class TestFixtureF:
    def test_composite_atomic_internal_order_preserved(self):
        clips = [_draft("a", "src1", 0, 0.0, 1.0), _draft("b", "src1", 1, 1.0, 2.0), _draft("c", "src1", 2, 2.0, 3.0)]
        units = ordp.build_ordering_units(realizations=clips)
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1"},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        provider = _DummyProvider(("c", "a", "b"))
        proposal = oca.validate_composer_ordering_proposal(
            units=units, realizations=clips, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert proposal.proposal_status == oca.PROPOSAL_ACCEPTED
        # scramble internal order -> rejected
        bad_proposal = oca.validate_composer_ordering_proposal(
            units=units, realizations=clips, relations=relations, baseline_plan=plan,
            provider=_DummyProvider(("c", "b", "a")),
        )
        assert bad_proposal.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert bad_proposal.composite_order_valid is False


class _DummyProvider:
    def __init__(self, proposed):
        self.proposed = proposed

    def order(self, takes, labels, strategy, context_text=""):
        from cutsell_worker.composer_provider import ComposerProviderResult
        from cutsell_worker.providers import ProviderStatus
        return ComposerProviderResult(tuple(self.proposed), ProviderStatus("mock", True, True, "applied"), "")


# ---------------------------------------------------------------------------
# 9. Fixture G -- P2 partial supersession with unique information.
# ---------------------------------------------------------------------------
class TestFixtureG:
    def test_partial_supersession_conflict_no_deletion(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1", proposition_ids=("p1",))
        m2 = _moment("src1", "s2", 20.0, 21.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2", proposition_ids=("p1", "p2"))
        understanding_p1 = _p1_understanding("src1", [m1, m2])
        region_a = _region("g1", [m1.editorial_moment_id], 0.0, 1.0, proposition_ids=("p1",))
        region_b = _region("g2", [m2.editorial_moment_id], 20.0, 21.0, proposition_ids=("p1", "p2"))
        # The earlier region has "p2" NOT covered -- a real unique-
        # information gap. PARTIAL supersession still routes through the
        # SAME "no delete authority" conflict handling as SUPPORTED does
        # (D-206's own established behavior) -- Ordering never resolves
        # this by dropping either side.
        hyp = WholeVideoSupersessionHypothesis(
            source_asset_id="src1", supersession_id="wvsup_2", earlier_region_ids=(region_a.region_id,),
            later_region_ids=(region_b.region_id,), covered_proposition_candidate_ids=("p1",),
            uncovered_earlier_proposition_candidate_ids=("p2",), coverage_status="PARTIAL_COVERAGE",
            recording_process_support="SUPPORTED", audience_delivery_support="SUPPORTED",
            meaning_conflict_status="MEANING_CONFLICT_NONE", supersession_status=SUPERSESSION_PARTIAL,
            confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
        )
        p2 = _p2_understanding([region_a, region_b], [hyp])
        clips = [_draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _draft("b", "src1", 1, 20.0, 21.0, source_span_id="s2")]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding_p1],
            whole_video_editorial_reasoning_result=p2, p1_diagnostics_enabled=True, p2_diagnostics_enabled=True,
        )
        # Membership always preserved; the uncovered unique information
        # (p2) lives on the hypothesis itself, never causing a drop.
        assert set(result.baseline_plan.ordered_realization_ids) == {"a", "b"}
        assert result.baseline_plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        assert any("SUPERSESSION_SURVIVAL_CONFLICT" in r.conflict_flags for r in result.relations)
        assert hyp.uncovered_earlier_proposition_candidate_ids == ("p2",)


# ---------------------------------------------------------------------------
# 10. Fixture H -- P2 meaning conflict.
# ---------------------------------------------------------------------------
class TestFixtureH:
    def test_meaning_conflict_ordering_cannot_resolve(self):
        region_a, region_b = "r_early", "r_late"
        clips = [_draft("a", "src1", 0, 0.0, 1.0), _draft("b", "src1", 1, 1.0, 2.0)]
        units = ordp.build_ordering_units(realizations=clips)
        conflict_hyp = WholeVideoSupersessionHypothesis(
            source_asset_id="src1", supersession_id="sup1", earlier_region_ids=(region_a,), later_region_ids=(region_b,),
            covered_proposition_candidate_ids=(), uncovered_earlier_proposition_candidate_ids=(),
            coverage_status="FULL_COVERAGE", recording_process_support="SUPPORTED", audience_delivery_support="SUPPORTED",
            meaning_conflict_status="MEANING_CONFLICT_PRESENT", supersession_status=SUPERSESSION_CONFLICTED,
            confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, supersession_hypotheses=[conflict_hyp],
            region_ids_by_unit={"a": (region_a,), "b": (region_b,)},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        proposal = oca.validate_composer_ordering_proposal(
            units=units, realizations=clips, relations=relations, baseline_plan=plan,
            provider=_DummyProvider(("a", "b")),
        )
        assert proposal.proposal_status == oca.PROPOSAL_NOT_EVALUABLE


# ---------------------------------------------------------------------------
# 11. Fixture I -- multi-source, no explicit cross-source relation.
# ---------------------------------------------------------------------------
class TestFixtureI:
    def test_uncertain_deterministic_fallback_only(self):
        clips = [_draft("a", "srcA", 0, 100.0, 101.0), _draft("b", "srcB", 1, 0.0, 1.0)]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        assert result.baseline_plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN
        assert result.baseline_plan.ordered_realization_ids == ("a", "b")  # by source_order, never raw start
        assert result.baseline_plan.source_asset_ids == ("srcA", "srcB")


# ---------------------------------------------------------------------------
# 12. Fixture J -- diagnostics flag off (pipeline-level, byte-compatible).
# ---------------------------------------------------------------------------
class TestFixtureJPipelineWiring:
    def _pipeline_fixture(self):
        from cutsell_worker.contracts import ProcessingRequest, SemanticLabel

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

    def test_default_off_byte_equivalent(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.delenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
        request, takes, labels, expected_winner = self._pipeline_fixture()
        result = build_flow_b_draft(request, takes, labels)
        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        assert result.draft.diagnostics["ordering"] == {"status": "disabled"}

    def test_flag_on_no_p1_p2_reports_partial_no_mutation(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.setenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", "1")
        request, takes, labels, expected_winner = self._pipeline_fixture()
        result = build_flow_b_draft(request, takes, labels)
        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        ordering_diag = result.draft.diagnostics["ordering"]
        assert ordering_diag["capability_status"] == olive.CAPABILITY_PARTIAL
        assert olive.MISSING_P1_UNAVAILABLE in ordering_diag["missing_evidence"]
        assert olive.MISSING_P2_UNAVAILABLE in ordering_diag["missing_evidence"]

    def test_flag_on_immutability_same_edit_output(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
        request, takes, labels, expected_winner = self._pipeline_fixture()

        monkeypatch.setenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", "0")
        off_result = build_flow_b_draft(request, takes, labels)
        monkeypatch.setenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", "1")
        on_result = build_flow_b_draft(request, takes, labels)

        assert [c.clip_id for c in on_result.draft.selected] == [c.clip_id for c in off_result.draft.selected] == [expected_winner]
        assert off_result.draft.diagnostics["ordering"] == {"status": "disabled"}
        assert on_result.draft.diagnostics["ordering"]["capability_status"] in (
            olive.CAPABILITY_PARTIAL, olive.CAPABILITY_AVAILABLE, olive.CAPABILITY_NOT_EVALUABLE,
        )
        # Every other diagnostics key stays identical whether Ordering ran or not.
        off_keys = {k: v for k, v in off_result.draft.diagnostics.items() if k != "ordering"}
        on_keys = {k: v for k, v in on_result.draft.diagnostics.items() if k != "ordering"}
        assert off_keys == on_keys

    def test_flag_on_with_real_p1_evidence_reports_units(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", "1")
        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.setenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", "1")
        request, takes, labels, expected_winner = self._pipeline_fixture()
        result = build_flow_b_draft(request, takes, labels)
        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        ordering_diag = result.draft.diagnostics["ordering"]
        assert ordering_diag["unit_count"] >= 1

    def test_determinism_across_repeated_calls(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.setenv("CUTSELL_ORDERING_DIAGNOSTICS_ENABLED", "1")
        request, takes, labels, _ = self._pipeline_fixture()
        r1 = build_flow_b_draft(request, takes, labels)
        r2 = build_flow_b_draft(request, takes, labels)
        assert r1.draft.diagnostics["ordering"] == r2.draft.diagnostics["ordering"]


# ---------------------------------------------------------------------------
# 13. Proposal validation fixture (existing deterministic composer path).
# ---------------------------------------------------------------------------
class TestProposalCompatibilityPath:
    def test_deterministic_provider_free_valid_proposal_accepted(self):
        clips = [_draft("b", "src1", 1, 2.0, 3.0), _draft("a", "src1", 0, 0.0, 1.0)]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        proposal = result.composer_proposal
        assert proposal is not None
        assert proposal.proposal_status == oca.PROPOSAL_ACCEPTED
        assert proposal.composer_path == oca.COMPOSER_PATH_MOCK_PROVIDER
        # compose_selected sorts chronologically -- natural source order.
        assert proposal.accepted_realization_ids == ("a", "b")

    def test_no_openai_composer_used_status_names_the_compat_path(self):
        clips = [_draft("a", "src1", 0, 0.0, 1.0), _draft("b", "src1", 1, 1.0, 2.0)]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        assert "openai" not in result.composer_proposal.composer_path.lower()


# ---------------------------------------------------------------------------
# 14. Diagnostics / run summary shape.
# ---------------------------------------------------------------------------
class TestDiagnosticsAndSummary:
    def test_diagnostics_shape_bounded(self):
        clips = [_draft("a", "src1", 0, 0.0, 1.0), _draft("b", "src1", 1, 1.0, 2.0)]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, p1_diagnostics_enabled=False, p2_diagnostics_enabled=False,
        )
        diag = olive.ordering_live_diagnostics(result)
        assert diag["capability_status"] == olive.CAPABILITY_PARTIAL
        assert diag["units"][0]["ordering_unit_id"] == diag["units"][0]["realization_id"]
        assert diag["relations"] == []
        assert diag["baseline_plan"] is not None
        assert diag["proposal_validation"] is not None
        assert "transcript" not in diag

    def test_run_summary_counts(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        understanding = _p1_understanding("src1", [m1, m2], [g])
        clips = [_draft("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _draft("b", "src1", 1, 2.0, 3.0, source_span_id="s2")]
        result = olive.build_ordering_live_diagnostics(
            selected_realizations=clips, editorial_moment_understandings=[understanding],
            p1_diagnostics_enabled=True, p2_diagnostics_enabled=False,
        )
        summary = olive.ordering_live_diagnostics_run_summary(result)
        assert summary["ordering_unit_count"] == 2
        assert summary["continuation_constraint_count"] == 1
        assert summary["ordering_status"] == ordp.ORDERING_STATUS_ORDERED
        assert summary["composer_proposal_available"] is True
        assert "master_score" not in summary


# ---------------------------------------------------------------------------
# 15. Structural no-recompute / no-authority / no-provider / no-QA /
# no-Family/BestTake/Boundary/Pacing/Renderer audits.
# ---------------------------------------------------------------------------
class TestStructuralAudits:
    def test_no_p1_p2_recompute(self):
        banned = (
            "from .asr", "from .language_spine import", "build_editorial_moment_understanding_for_sources(",
            "build_whole_video_editorial_understanding(", "from .watch_listen_understanding import build",
            "from .local_performance", "from .prosodic_audio_v2",
        )
        for name in banned:
            assert name not in MODULE_SOURCE

    def test_openai_composer_never_imported(self):
        for banned in ("composer_openai", "OpenAIComposerProvider(", "responses.create"):
            assert banned not in MODULE_SOURCE

    def test_no_network(self):
        for banned in ("requests.", "urllib", "http.client", "OPENAI_API_KEY"):
            assert banned not in MODULE_SOURCE

    def test_causal_validator_never_imported(self):
        for banned in ("causal_order_validator", "find_causal_order_breaks("):
            assert banned not in MODULE_SOURCE

    def test_d206_d207_builders_reused_not_reimplemented(self):
        assert "def build_ordering_units(" not in MODULE_SOURCE
        assert "def build_deterministic_ordering_plan(" not in MODULE_SOURCE
        assert "def validate_composer_ordering_proposal(" not in MODULE_SOURCE
        for real_call in (
            "build_ordering_units(", "build_ordering_relation_evidence(",
            "build_deterministic_ordering_plan(", "validate_composer_ordering_proposal(",
        ):
            assert real_call in MODULE_SOURCE

    def test_no_family_besttake_boundary_pacing_renderer_mutation(self):
        forbidden = (
            "take_group_id =", "_semantic_best_take", "bounded_finalist_authority",
            "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
            "RenderSegment", "take_grouping", "composite_resolver", "realization_resolver",
        )
        for name in forbidden:
            assert name not in CODE_ONLY

    def test_no_selected_clip_mutation(self):
        assert "selected_clip_id =" not in CODE_ONLY
        assert ".selected = " not in CODE_ONLY

    def test_no_qa_references(self):
        for banned in ("cut_ai", "cutai", "human_gold", "quality_ladder", "benchmark_label"):
            assert banned not in CODE_ONLY.lower()

    def test_no_commercial_or_funnel_fields(self):
        for banned in ("commercial", "sales_funnel", "funnel", "cta_score", "hook_strength", "narrative_quality"):
            assert banned not in CODE_ONLY.lower()

    def test_no_master_score(self):
        from dataclasses import fields
        assert "score" not in {f.name for f in fields(olive.OrderingLiveDiagnosticsResult)}

    def test_no_pipeline_stage_move(self):
        pipeline_source = (REPO_ROOT / "cutsell_worker" / "pipeline.py").read_text()
        # The existing composer call site must remain exactly where D-205
        # found it -- before the P1/P2/Ordering diagnostic block, never
        # moved after Freeze (which does not mechanically exist in this
        # function at all).
        composer_call_index = pipeline_source.index("composition = safe_compose_order(")
        ordering_import_index = pipeline_source.index("from .ordering_live_diagnostics_integration import")
        ordering_call_index = pipeline_source.index("if ordering_diagnostics_enabled():")
        assert composer_call_index < ordering_call_index
        assert ordering_import_index < ordering_call_index

    def test_existing_composer_files_unchanged_by_this_module(self):
        for banned in ("def compose_selected(", "def safe_compose_order(", "def _repair_order("):
            assert banned not in MODULE_SOURCE


def test_module_compiles_and_imports():
    import cutsell_worker.ordering_live_diagnostics_integration  # noqa: F401
