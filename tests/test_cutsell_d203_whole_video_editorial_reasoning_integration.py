"""D-203: P2 Whole-Video Editorial Reasoning -- Phase B live diagnostic
integration offline tests.

Covers: the integration builder's flag/evidence-dependency logic, real P1/
Language object reuse (never recomputed), diagnostics/run-summary shape,
pipeline-level wiring (default-off parity, flag-on immutability, real
serialization), the D-203 fixture replay (A-F), and structural no-
recompute/no-authority/no-QA-reference audits.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import fields
from pathlib import Path

import pytest

from cutsell_worker.editorial_moment_sequence import (
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_UNCERTAIN,
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    RECORDING_PROCESS_ABSENT,
    RECORDING_PROCESS_PRESENT,
    EditorialMoment,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    EditorialLocalGroup,
    EditorialMomentUnderstanding,
)
from cutsell_worker.language_proposition_relation import (
    MEANING_COMPLETE,
    ClaimSignature,
    PropositionCandidate,
)
from cutsell_worker.language_spine_live_integration import LiveLanguageSpineEvidence
from cutsell_worker import whole_video_editorial_reasoning as p2
from cutsell_worker import whole_video_editorial_reasoning_integration as p2i

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_SOURCE = (REPO_ROOT / "cutsell_worker" / "whole_video_editorial_reasoning_integration.py").read_text()
PIPELINE_SOURCE = (REPO_ROOT / "cutsell_worker" / "pipeline.py").read_text()


def _code_only(source: str, marker: str) -> str:
    idx = source.find(marker)
    return source[idx + len(marker):] if idx != -1 else source


INTEGRATION_CODE_ONLY = _code_only(INTEGRATION_SOURCE, '"""\nfrom __future__')


# ---------------------------------------------------------------------------
# Fixture helpers (same pattern as D-202's own test file).
# ---------------------------------------------------------------------------
def _moment(source_asset_id, seed, start, end, role, *, proposition_ids=(), confidence=CONFIDENCE_SUPPORTED):
    audience_status = (
        AUDIENCE_DELIVERY_SUPPORTED if role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
        else AUDIENCE_DELIVERY_UNCERTAIN if role == MOMENT_ROLE_UNCERTAIN
        else AUDIENCE_DELIVERY_NOT_SUPPORTED
    )
    recording_status = RECORDING_PROCESS_PRESENT if role == MOMENT_ROLE_RECORDING_PROCESS else RECORDING_PROCESS_ABSENT
    return EditorialMoment(
        source_asset_id=source_asset_id, editorial_moment_id=f"emom_{seed}", source_start=start, source_end=end,
        source_span_id=f"span_{seed}", attempt_ids=(f"att_{seed}",), proposition_candidate_ids=tuple(proposition_ids),
        related_span_ids=(), moment_role=role, audience_delivery_status=audience_status,
        recording_process_status=recording_status, completion_status=MEANING_COMPLETE, local_sequence_position=None,
        confidence=confidence, conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _group(source_asset_id, seed, moments):
    return EditorialLocalGroup(
        source_asset_id=source_asset_id, group_id=f"elgrp_{seed}", moment_indices=tuple(range(len(moments))),
        moment_ids=tuple(m.editorial_moment_id for m in moments), source_start=min(m.source_start for m in moments),
        source_end=max(m.source_end for m in moments),
        grouping_reason="RELATION_LINKED_CHAIN" if len(moments) > 1 else "SOLE_MOMENT_IN_SOURCE",
        relation_support=(), confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _understanding(source_asset_id, moments, groups, capability_status="AVAILABLE") -> EditorialMomentUnderstanding:
    return EditorialMomentUnderstanding(
        source_asset_id=source_asset_id, moments=tuple(moments), sequence_hypotheses=(),
        moment_count=len(moments), sequence_count=0, capability_status=capability_status, missing_evidence=(),
        confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=("TEST_FIXTURE",), local_groups=tuple(groups),
    )


def _signature(tokens, *, negation=False, numbers=(), claim_type="CLAIM") -> ClaimSignature:
    content = frozenset(tokens)
    return ClaimSignature(
        content_tokens=content, negation_present=negation, numbers=frozenset(numbers), claim_type=claim_type,
        negation_role="FACTUAL_NEGATION" if negation else "",
        signature_hash=hashlib.sha256("|".join(sorted(content)).encode()).hexdigest()[:20],
    )


def _proposition(source_asset_id, seed, signature, *, start, end, meaning_completion=MEANING_COMPLETE):
    return PropositionCandidate(
        source_asset_id=source_asset_id, proposition_candidate_id=f"prop_{seed}", attempt_ids=(f"att_{seed}",),
        source_start=start, source_end=end, text_raw="", text_normalized="", claim_signature=signature,
        meaning_completion=meaning_completion, editorial_slot_evidence="OTHER", confidence=CONFIDENCE_SUPPORTED,
        provenance="LANGUAGE_ATTEMPT", conflict_flags=(),
    )


def _live_language_spine(source_asset_id, proposition_candidates=()) -> LiveLanguageSpineEvidence:
    return LiveLanguageSpineEvidence(
        source_asset_id=source_asset_id, words=(), phrases=(), utterances=(), attempts=(),
        proposition_candidates=tuple(proposition_candidates), relation_evidence=(), capability_status="AVAILABLE",
        missing_evidence=(), conflicts=(), provenance=("TEST_FIXTURE",),
    )


SIG_A = _signature({"alpha", "topic", "claim"})
SIG_A2 = _signature({"alpha", "topic", "claim"})
SIG_B = _signature({"beta", "other", "matter"})
SIG_A_NEG = _signature({"alpha", "topic", "claim"}, negation=True)


# ---------------------------------------------------------------------------
# 1. Feature flag.
# ---------------------------------------------------------------------------
class TestFeatureFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        assert p2i.whole_video_editorial_reasoning_diagnostics_enabled() is False

    def test_on_when_set(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "1")
        assert p2i.whole_video_editorial_reasoning_diagnostics_enabled() is True

    def test_no_authority_flag_exists(self):
        assert "AUTHORITY_ENABLED" not in INTEGRATION_SOURCE


# ---------------------------------------------------------------------------
# 2. Flag/evidence dependency logic (build_whole_video_editorial_reasoning).
# ---------------------------------------------------------------------------
class TestIntegrationBuilder:
    def test_p1_flag_off_is_not_evaluable(self):
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=(), p1_diagnostics_enabled=False,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.capability_status == p2.CAPABILITY_NOT_EVALUABLE
        assert p2i.MISSING_P1_DIAGNOSTICS_DISABLED in result.missing_evidence
        assert result.understanding is None

    def test_p1_on_zero_evidence_is_not_evaluable(self):
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=(), p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.capability_status == p2.CAPABILITY_NOT_EVALUABLE
        assert p2i.MISSING_NO_P1_EVIDENCE in result.missing_evidence

    def test_p1_on_language_off_is_partial(self):
        m = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m])
        u = _understanding("src1", [m], [g])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.capability_status == p2.CAPABILITY_PARTIAL
        assert p2i.MISSING_LIVE_LANGUAGE_SPINE_DISABLED in result.missing_evidence
        assert result.understanding is not None
        assert len(result.understanding.regions) == 1

    def test_both_flags_on_full_evidence_is_available(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a2",))
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        u = _understanding("src1", [m1, m2], [g1, g2])
        props = [
            _proposition("src1", "a", SIG_A, start=0.0, end=1.0),
            _proposition("src1", "a2", SIG_A2, start=10.0, end=11.0),
        ]
        spine = {"src1": _live_language_spine("src1", props)}
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], live_language_spine_by_source=spine,
            p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=True,
        )
        assert result.capability_status == p2.CAPABILITY_AVAILABLE
        assert not result.missing_evidence
        assert len(result.understanding.supersession_hypotheses) == 1

    def test_missing_p1_source_counted(self):
        m = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m])
        good = _understanding("src1", [m], [g])
        empty = EditorialMomentUnderstanding(
            source_asset_id="src2", moments=(), sequence_hypotheses=(), moment_count=0, sequence_count=0,
            capability_status="NOT_EVALUABLE", missing_evidence=("WATCH_LISTEN_UNDERSTANDING_ABSENT",),
            confidence=CONFIDENCE_UNKNOWN, conflict_flags=(), provenance=(),
        )
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[good, empty], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.p2_missing_p1_source_count == 1
        assert result.p2_source_count == 1

    def test_missing_language_source_counted_per_source(self):
        m1 = _moment("srcA", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("srcB", "b", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u1 = _understanding("srcA", [m1], [_group("srcA", "g1", [m1])])
        u2 = _understanding("srcB", [m2], [_group("srcB", "g2", [m2])])
        spine = {"srcA": _live_language_spine("srcA", [])}  # srcB has no entry at all
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u1, u2], live_language_spine_by_source=spine,
            p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=True,
        )
        assert result.p2_missing_language_source_count == 2  # srcA has empty tuple, srcB has none
        assert result.capability_status == p2.CAPABILITY_PARTIAL

    def test_deterministic_repeat(self):
        m = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u = _understanding("src1", [m], [_group("src1", "g", [m])])
        run1 = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        run2 = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert run1 == run2

    def test_multi_source_isolation(self):
        m1 = _moment("srcA", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("srcB", "b", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u1 = _understanding("srcA", [m1], [_group("srcA", "g1", [m1])])
        u2 = _understanding("srcB", [m2], [_group("srcB", "g2", [m2])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u1, u2], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.understanding.source_asset_ids == ("srcA", "srcB")
        assert {r.source_asset_id for r in result.understanding.regions} == {"srcA", "srcB"}


# ---------------------------------------------------------------------------
# 3. D-203 fixture replay (A-F), at the INTEGRATION layer.
# ---------------------------------------------------------------------------
class TestFixtureReplay:
    def test_fixture_A_early_process_later_clean_same_proposition(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a2",))
        u = _understanding("src1", [m1, m2], [_group("src1", "g1", [m1]), _group("src1", "g2", [m2])])
        spine = {"src1": _live_language_spine("src1", [
            _proposition("src1", "a", SIG_A, start=0.0, end=1.0),
            _proposition("src1", "a2", SIG_A2, start=10.0, end=11.0),
        ])}
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], live_language_spine_by_source=spine,
            p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=True,
        )
        assert len(result.understanding.regions) >= 2
        # NOTE: prop_a/prop_a2 are two DISTINCT proposition_candidate_ids
        # (D-169 mints a new id per distinct timing even for identical
        # content) -- the realization MAP groups strictly by identical id
        # (never by content similarity, per D-201's own "do not equate
        # proposition id with retry family id" instruction), so it
        # correctly reports two SINGLE_REALIZATION rows here. The cross-
        # region "same idea, different distant realizations" claim is
        # exactly what the SUPERSESSION hypothesis (via claim-signature
        # comparison) exists to represent instead.
        assert len(result.understanding.proposition_realization_maps) == 2
        (hyp,) = result.understanding.supersession_hypotheses
        assert hyp.supersession_status == p2.SUPERSESSION_SUPPORTED

    def test_fixture_B_unique_information(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a", "prop_b"))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a2",))
        u = _understanding("src1", [m1, m2], [_group("src1", "g1", [m1]), _group("src1", "g2", [m2])])
        spine = {"src1": _live_language_spine("src1", [
            _proposition("src1", "a", SIG_A, start=0.0, end=0.5),
            _proposition("src1", "b", SIG_B, start=0.5, end=1.0),
            _proposition("src1", "a2", SIG_A2, start=10.0, end=11.0),
        ])}
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], live_language_spine_by_source=spine,
            p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=True,
        )
        (hyp,) = result.understanding.supersession_hypotheses
        assert "prop_b" in hyp.uncovered_earlier_proposition_candidate_ids
        assert hyp.supersession_status != p2.SUPERSESSION_SUPPORTED

    def test_fixture_C_meaning_conflict(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a_neg",))
        u = _understanding("src1", [m1, m2], [_group("src1", "g1", [m1]), _group("src1", "g2", [m2])])
        spine = {"src1": _live_language_spine("src1", [
            _proposition("src1", "a", SIG_A, start=0.0, end=1.0),
            _proposition("src1", "a_neg", SIG_A_NEG, start=10.0, end=11.0),
        ])}
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], live_language_spine_by_source=spine,
            p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=True,
        )
        (hyp,) = result.understanding.supersession_hypotheses
        assert hyp.supersession_status == p2.SUPERSESSION_CONFLICTED

    def test_fixture_D_chronology_only(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u = _understanding("src1", [m1, m2], [_group("src1", "g1", [m1]), _group("src1", "g2", [m2])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert all(h.supersession_status != p2.SUPERSESSION_SUPPORTED for h in result.understanding.supersession_hypotheses)

    def test_fixture_E_no_preassembled_label_still_builds(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u = _understanding("src1", [m1], [_group("src1", "g1", [m1])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        assert result.capability_status in (p2.CAPABILITY_AVAILABLE, p2.CAPABILITY_PARTIAL)
        assert len(result.understanding.regions) == 1

    def test_fixture_F_multi_source(self):
        m1 = _moment("srcA", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("srcB", "b", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS)
        u1 = _understanding("srcA", [m1], [_group("srcA", "g1", [m1])])
        u2 = _understanding("srcB", [m2], [_group("srcB", "g2", [m2])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u1, u2], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        by_source = {r.source_asset_id for r in result.understanding.regions}
        assert by_source == {"srcA", "srcB"}


# ---------------------------------------------------------------------------
# 4. Diagnostics / run summary.
# ---------------------------------------------------------------------------
class TestDiagnosticsAndRunSummary:
    def test_diagnostics_not_evaluable_shape(self):
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=(), p1_diagnostics_enabled=False,
            live_language_spine_diagnostics_enabled=False,
        )
        diag = p2i.whole_video_editorial_reasoning_diagnostics(result)
        assert diag["status"] == "not_evaluable"
        assert diag["region_count"] == 0
        assert "transcript" not in diag

    def test_diagnostics_evaluated_shape(self):
        m = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u = _understanding("src1", [m], [_group("src1", "g", [m])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        diag = p2i.whole_video_editorial_reasoning_diagnostics(result)
        assert diag["status"] == "evaluated"
        assert diag["region_count"] == 1
        assert isinstance(diag["regions"], list) and diag["regions"]
        required = {
            "region_id", "source_asset_id", "source_start", "source_end", "moment_ids", "local_group_ids",
            "sequence_ids", "proposition_candidate_ids", "dominant_process_status", "audience_delivery_status",
            "confidence", "conflict", "provenance",
        }
        assert required.issubset(diag["regions"][0].keys())

    def test_run_summary_has_p2_source_counts_and_no_master_score(self):
        m = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        u = _understanding("src1", [m], [_group("src1", "g", [m])])
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=[u], p1_diagnostics_enabled=True,
            live_language_spine_diagnostics_enabled=False,
        )
        summary = p2i.whole_video_editorial_reasoning_run_summary(result)
        for key in ("p2_source_count", "p2_missing_p1_source_count", "p2_missing_language_source_count"):
            assert key in summary
        assert all(not isinstance(v, float) for v in summary.values())

    def test_run_summary_not_evaluable_zeros(self):
        result = p2i.build_whole_video_editorial_reasoning(
            editorial_moment_understandings=(), p1_diagnostics_enabled=False,
            live_language_spine_diagnostics_enabled=False,
        )
        summary = p2i.whole_video_editorial_reasoning_run_summary(result)
        assert summary["whole_video_region_count"] == 0
        assert summary["capability_status"] == p2.CAPABILITY_NOT_EVALUABLE


# ---------------------------------------------------------------------------
# 5. Pipeline-level wiring: default-off parity / flag-on immutability.
# ---------------------------------------------------------------------------
def _pipeline_fixture():
    from cutsell_worker.contracts import CandidateTake, MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole

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


class TestPipelineWiring:
    def test_default_off_byte_equivalent(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.delenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
        request, takes, labels, expected_winner = _pipeline_fixture()
        result = build_flow_b_draft(request, takes, labels)
        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        assert result.draft.diagnostics["whole_video_editorial_reasoning"] == {"status": "disabled"}

    def test_flag_on_p1_off_reports_not_evaluable_no_mutation(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "1")
        request, takes, labels, expected_winner = _pipeline_fixture()
        result = build_flow_b_draft(request, takes, labels)
        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        p2_diag = result.draft.diagnostics["whole_video_editorial_reasoning"]
        assert p2_diag["status"] == "not_evaluable"
        assert p2i.MISSING_P1_DIAGNOSTICS_DISABLED in p2_diag["missing_evidence"]

    def test_flag_on_with_p1_on_evaluates_without_mutating_winner(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft

        monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", "1")
        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "1")
        monkeypatch.delenv("CUTSELL_LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED", raising=False)
        request, takes, labels, expected_winner = _pipeline_fixture()

        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "0")
        off_result = build_flow_b_draft(request, takes, labels)
        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "1")
        on_result = build_flow_b_draft(request, takes, labels)

        assert [c.clip_id for c in on_result.draft.selected] == [c.clip_id for c in off_result.draft.selected] == [expected_winner]
        p2_diag = on_result.draft.diagnostics["whole_video_editorial_reasoning"]
        assert p2_diag["status"] in ("evaluated", "not_evaluable")
        assert off_result.draft.diagnostics["whole_video_editorial_reasoning"] == {"status": "disabled"}

    def test_flag_on_serializes_real_regions_from_live_p1_evidence(self, monkeypatch):
        from cutsell_worker.pipeline import build_flow_b_draft
        from cutsell_worker.watch_listen_understanding import (
            RELATION_RETRY as WL_RELATION_RETRY,
            CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
            CONFIDENCE_UNKNOWN as WL_CONFIDENCE_UNKNOWN,
            MEANING_COMPLETE as WL_MEANING_COMPLETE,
            USABILITY_USABLE,
            AttemptRelationHypothesis,
            UnderstandingSpan,
            WatchListenUnderstanding,
        )
        from cutsell_worker.raw_understanding_map import BEHAVIOR_CLEAN_ATTEMPT, BehaviorHypothesis

        def _behavior(label):
            return BehaviorHypothesis(label=label, confidence=0.8, provenance="VISUAL_SIGNAL", basis="generic")

        request, takes, labels, expected_winner = _pipeline_fixture()
        wlu = WatchListenUnderstanding(source_asset_id="src", understanding_spans=(
            UnderstandingSpan(
                span_id="weak", source_asset_id="src", source_start=1.0, source_end=3.0,
                behavior_state_hypotheses=(_behavior(BEHAVIOR_CLEAN_ATTEMPT),), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
                attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(), relation_confidence=WL_CONFIDENCE_UNKNOWN,
                meaning_completion_hypothesis=WL_MEANING_COMPLETE, performance_usability_hypothesis=USABILITY_USABLE,
                entry_usability=USABILITY_USABLE, delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
                conflict_flags=(), evidence_provenance={},
            ),
            UnderstandingSpan(
                span_id="strong", source_asset_id="src", source_start=4.0, source_end=6.0,
                behavior_state_hypotheses=(_behavior(BEHAVIOR_CLEAN_ATTEMPT),), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
                attempt_boundary_hypotheses=(),
                attempt_relation_hypotheses=(AttemptRelationHypothesis(WL_RELATION_RETRY, WL_CONFIDENCE_SUPPORTED, "x", "weak", ()),),
                relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
                performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
                delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
                conflict_flags=(), evidence_provenance={},
            ),
        ))
        monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", "1")
        monkeypatch.setenv("CUTSELL_WHOLE_VIDEO_EDITORIAL_REASONING_DIAGNOSTICS_ENABLED", "1")
        result = build_flow_b_draft(request, takes, labels, watch_listen_understandings=(wlu,))

        assert [c.clip_id for c in result.draft.selected] == [expected_winner]
        p2_diag = result.draft.diagnostics["whole_video_editorial_reasoning"]
        assert p2_diag["status"] == "evaluated"
        assert p2_diag["region_count"] >= 1
        # Determinism across repeated calls with identical inputs.
        result2 = build_flow_b_draft(request, takes, labels, watch_listen_understandings=(wlu,))
        assert result2.draft.diagnostics["whole_video_editorial_reasoning"]["regions"] == p2_diag["regions"]


# ---------------------------------------------------------------------------
# 6. No-recompute / no-authority / no-QA-reference structural audits.
# ---------------------------------------------------------------------------
class TestStructuralAudits:
    def test_no_asr_language_spine_watch_listen_visual_prosodic_rerun(self):
        banned_imports = (
            "from .asr", "from .language_spine import", "from .language_utterance_attempt import build",
            "from .watch_listen_understanding import build", "from .local_performance", "from .prosodic_audio_v2",
        )
        for banned in banned_imports:
            assert banned not in INTEGRATION_SOURCE

    def test_no_second_proposition_or_p1_builder(self):
        assert "build_proposition_candidates" not in INTEGRATION_CODE_ONLY
        assert "build_relation_evidence" not in INTEGRATION_CODE_ONLY
        assert "classify_editorial_moment(" not in INTEGRATION_CODE_ONLY
        assert "classify_editorial_sequence(" not in INTEGRATION_CODE_ONLY

    def test_no_provider_call(self):
        for banned in ("openai", "gemini", "whole_video_openai", "responses.create"):
            assert banned not in INTEGRATION_CODE_ONLY

    def test_no_qa_reference(self):
        banned = ("cut_ai", "human_gold", "quality_ladder", "benchmark_label")
        for name in banned:
            assert name not in INTEGRATION_SOURCE.lower()

    def test_no_family_besttake_ordering_boundary_pacing_renderer_mutation(self):
        forbidden = (
            "take_group_id", "family_complete_context", "selected_clip_id", "_semantic_best_take",
            "bounded_finalist_authority", "bounded_finalist_arbiter", "winner_after",
            "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
            "render_plan", "RenderSegment", "canonical_edit_plan", "composite_resolver",
            "realization_resolver", "take_grouping",
        )
        for name in forbidden:
            assert name not in INTEGRATION_CODE_ONLY

    def test_no_p2_authority_field_or_action(self):
        for cls in (p2i.WholeVideoEditorialReasoningResult,):
            for f in fields(cls):
                for banned in ("delete", "winner", "selected_clip_id", "final_winner", "action"):
                    assert banned not in f.name.lower()

    def test_no_commercial_or_funnel_fields(self):
        for banned in ("commercial", "sales_funnel", "funnel", "cta_score"):
            assert banned not in INTEGRATION_SOURCE.lower()

    def test_pipeline_p2_block_is_diagnostics_only(self):
        # The pipeline.py diff region referencing this module's own symbols
        # must never appear inside a Family/BestTake/D-191/Boundary/Pacing/
        # Renderer mutation call -- confirmed by checking the imported
        # symbol names never co-occur with an assignment to those
        # authorities' own known output fields in the same statement.
        assert "whole_video_editorial_reasoning_integration import" in PIPELINE_SOURCE
        forbidden_co_occurrence = ("selected_clip_id =", "final_winner =", "render_plan =")
        # Sanity: these authority-mutation patterns simply don't exist
        # anywhere near the P2 import/wiring block at all.
        p2_block_start = PIPELINE_SOURCE.find("whole_video_editorial_reasoning_diagnostics_enabled()")
        p2_block = PIPELINE_SOURCE[p2_block_start:p2_block_start + 2000]
        for pattern in forbidden_co_occurrence:
            assert pattern not in p2_block

    def test_no_hidden_flag_coupling(self):
        # P2's own flag check never silently forces the P1/live-language-
        # spine flags on -- it only ever READS their already-passed-in
        # boolean state (p1_diagnostics_enabled / live_language_spine_
        # diagnostics_enabled arguments), never sets an env var itself.
        assert "os.environ[" not in INTEGRATION_SOURCE
        assert "setenv" not in INTEGRATION_SOURCE
        assert ".environ.update" not in INTEGRATION_SOURCE


# ---------------------------------------------------------------------------
# 7. Runtime sanity (no invented threshold, just a bounded sanity ceiling).
# ---------------------------------------------------------------------------
def test_pure_integration_runtime_lightweight():
    moments = [_moment("src1", str(i), float(i), float(i) + 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY) for i in range(30)]
    groups = [_group("src1", str(i), [m]) for i, m in enumerate(moments)]
    u = _understanding("src1", moments, groups)
    start = time.monotonic()
    p2i.build_whole_video_editorial_reasoning(
        editorial_moment_understandings=[u], p1_diagnostics_enabled=True, live_language_spine_diagnostics_enabled=False,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
