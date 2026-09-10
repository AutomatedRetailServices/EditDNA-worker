"""D-202: P2 Whole-Video Editorial Reasoning -- Phase A offline tests.

Generic, no Video00-specific phrases/timestamps/clip ids anywhere in this
file (structural audit, Section below). Covers: type/vocabulary contracts,
region construction, proposition realization mapping, cross-source
correspondence, supersession hypotheses (coverage/role/meaning/chronology
firewalls), the aggregate understanding builder, diagnostics/run summary,
the D-201 contract replay (A-E), and structural no-authority / no-second-
ontology / no-pipeline-wiring proofs.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from cutsell_worker.editorial_moment_sequence import (
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_NEW_AUDIENCE_BEAT,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
    MOMENT_ROLE_UNCERTAIN,
    RECORDING_PROCESS_ABSENT,
    RECORDING_PROCESS_PRESENT,
    EditorialMoment,
    EditorialSequenceHypothesis,
)
from cutsell_worker.editorial_moment_sequence_integration import EditorialLocalGroup
from cutsell_worker.language_proposition_relation import (
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    ClaimSignature,
    PropositionCandidate,
)
from cutsell_worker import whole_video_editorial_reasoning as p2

REPO_ROOT = Path(__file__).resolve().parent.parent
P2_MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "whole_video_editorial_reasoning.py").read_text()


def _code_only_source() -> str:
    """Strips the top-of-file module docstring (which necessarily discusses,
    in prose, the very things it promises never to do -- OpenAI/Gemini,
    BEAT_SAME, PREASSEMBLED_FINAL_SEQUENCE, REGION_GAP_SECONDS, "majority
    vote"/"latest wins" as disclaimed non-behaviors) so structural absence
    checks below inspect only actual code, never documentation prose. Same
    pattern this codebase's own D-194 test suite already established."""
    marker = '"""\nfrom __future__'
    idx = P2_MODULE_SOURCE.find(marker)
    if idx == -1:
        # Fallback: strip up to the second occurrence of a bare `"""` line.
        parts = P2_MODULE_SOURCE.split('"""', 2)
        return parts[2] if len(parts) == 3 else P2_MODULE_SOURCE
    return P2_MODULE_SOURCE[idx + len('"""'):]


P2_CODE_ONLY_SOURCE = _code_only_source()


# ---------------------------------------------------------------------------
# Fixture helpers -- deterministic, hand-built objects (never depend on
# extract_claims' own exact NLP heuristics for firewall determinism).
# ---------------------------------------------------------------------------
def _moment(
    source_asset_id: str, seed: str, start: float, end: float, role: str,
    *, proposition_ids=(), confidence=CONFIDENCE_SUPPORTED, conflict_flags=(),
) -> EditorialMoment:
    audience_status = (
        AUDIENCE_DELIVERY_SUPPORTED if role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
        else AUDIENCE_DELIVERY_UNCERTAIN if role == MOMENT_ROLE_UNCERTAIN
        else AUDIENCE_DELIVERY_NOT_SUPPORTED
    )
    recording_status = RECORDING_PROCESS_PRESENT if role == MOMENT_ROLE_RECORDING_PROCESS else RECORDING_PROCESS_ABSENT
    return EditorialMoment(
        source_asset_id=source_asset_id,
        editorial_moment_id=f"emom_{seed}",
        source_start=start,
        source_end=end,
        source_span_id=f"span_{seed}",
        attempt_ids=(f"att_{seed}",),
        proposition_candidate_ids=tuple(proposition_ids),
        related_span_ids=(),
        moment_role=role,
        audience_delivery_status=audience_status,
        recording_process_status=recording_status,
        completion_status=MEANING_COMPLETE,
        local_sequence_position=None,
        confidence=confidence,
        conflict_flags=tuple(conflict_flags),
        provenance=("TEST_FIXTURE",),
    )


def _group(source_asset_id: str, seed: str, moments) -> EditorialLocalGroup:
    return EditorialLocalGroup(
        source_asset_id=source_asset_id,
        group_id=f"elgrp_{seed}",
        moment_indices=tuple(range(len(moments))),
        moment_ids=tuple(m.editorial_moment_id for m in moments),
        source_start=min(m.source_start for m in moments),
        source_end=max(m.source_end for m in moments),
        grouping_reason="RELATION_LINKED_CHAIN" if len(moments) > 1 else "SOLE_MOMENT_IN_SOURCE",
        relation_support=(),
        confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(),
        provenance=("TEST_FIXTURE",),
    )


def _sequence(source_asset_id: str, seed: str, moments) -> EditorialSequenceHypothesis:
    return EditorialSequenceHypothesis(
        source_asset_id=source_asset_id,
        sequence_id=f"eseq_{seed}",
        moment_ids=tuple(m.editorial_moment_id for m in moments),
        source_start=min(m.source_start for m in moments),
        source_end=max(m.source_end for m in moments),
        sequence_kind="CLEAN_DELIVERY_SEQUENCE",
        sequence_completeness=MEANING_COMPLETE,
        audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        recording_process_status=RECORDING_PROCESS_ABSENT,
        proposition_progression_status="UNKNOWN",
        internal_redundancy_status="INTERNAL_REDUNDANCY_NOT_EVALUATED",
        continuity_status="NOT_AVAILABLE",
        earlier_source_redundancy_status="NOT_EVALUATED",
        confidence=CONFIDENCE_WEAK,
        conflict_flags=(),
        provenance=("TEST_FIXTURE",),
    )


def _signature(tokens, *, negation=False, numbers=(), claim_type="CLAIM") -> ClaimSignature:
    content = frozenset(tokens)
    return ClaimSignature(
        content_tokens=content,
        negation_present=negation,
        numbers=frozenset(numbers),
        claim_type=claim_type,
        negation_role="FACTUAL_NEGATION" if negation else "",
        signature_hash=hashlib.sha256("|".join(sorted(content)).encode()).hexdigest()[:20],
    )


def _proposition(
    source_asset_id: str, seed: str, signature: ClaimSignature, *, start: float, end: float,
    meaning_completion=MEANING_COMPLETE,
) -> PropositionCandidate:
    return PropositionCandidate(
        source_asset_id=source_asset_id,
        proposition_candidate_id=f"prop_{seed}",
        attempt_ids=(f"att_{seed}",),
        source_start=start,
        source_end=end,
        text_raw="",
        text_normalized="",
        claim_signature=signature,
        meaning_completion=meaning_completion,
        editorial_slot_evidence="OTHER",
        confidence=CONFIDENCE_SUPPORTED,
        provenance="LANGUAGE_ATTEMPT",
        conflict_flags=(),
    )


SIG_ALPHA = _signature({"alpha", "topic", "claim"})
SIG_ALPHA_2 = _signature({"alpha", "topic", "claim"})
SIG_BETA = _signature({"beta", "other", "matter"})
SIG_ALPHA_NEGATED = _signature({"alpha", "topic", "claim"}, negation=True)
SIG_ALPHA_NUM2 = _signature({"alpha", "topic", "claim"}, numbers=("2",))
SIG_ALPHA_NUM3 = _signature({"alpha", "topic", "claim"}, numbers=("3",))


# ---------------------------------------------------------------------------
# 1. Type / vocabulary contracts (deliverable items 3-6).
# ---------------------------------------------------------------------------
class TestTypesAndVocabulary:
    def test_all_four_types_are_frozen_dataclasses(self):
        for cls in (
            p2.WholeVideoEditorialRegion, p2.WholeVideoPropositionRealizationMap,
            p2.WholeVideoSupersessionHypothesis, p2.WholeVideoEditorialUnderstanding,
        ):
            assert cls.__dataclass_params__.frozen is True

    def test_region_vocabulary_is_small_and_excludes_preassembled(self):
        assert p2.ALLOWED_REGION_PROCESS_STATUSES == {
            p2.REGION_RECORDING_PROCESS, p2.REGION_TAKE_SERIES, p2.REGION_CLEAN_DELIVERY,
            p2.REGION_MIXED, p2.REGION_UNKNOWN,
        }
        assert "PREASSEMBLED" not in "".join(p2.ALLOWED_REGION_PROCESS_STATUSES)

    def test_supersession_vocabulary_is_categorical_never_numeric(self):
        assert p2.ALLOWED_SUPERSESSION_STATUSES == {
            p2.SUPERSESSION_SUPPORTED, p2.SUPERSESSION_PARTIAL, p2.SUPERSESSION_NO_SAFE,
            p2.SUPERSESSION_CONFLICTED, p2.SUPERSESSION_UNKNOWN,
        }
        for value in p2.ALLOWED_SUPERSESSION_STATUSES:
            assert isinstance(value, str)

    def test_confidence_vocabulary_matches_language_utterance_attempt(self):
        assert {CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_MIXED, CONFIDENCE_UNKNOWN} == {
            p2.CONFIDENCE_SUPPORTED, p2.CONFIDENCE_WEAK, p2.CONFIDENCE_MIXED, p2.CONFIDENCE_UNKNOWN,
        }


# ---------------------------------------------------------------------------
# 2. Region construction (deliverable items 7, 10-12, 23-25, 28, 32-33).
# ---------------------------------------------------------------------------
class TestRegionConstruction:
    def test_one_region_per_local_group(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS)
        m2 = _moment("src1", "b", 2.0, 3.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        regions = p2.build_whole_video_editorial_regions(
            source_asset_id="src1", moments=[m1, m2], local_groups=[g1, g2],
        )
        assert len(regions) == 2
        assert regions[0].local_group_ids == ("elgrp_g1",)
        assert regions[1].local_group_ids == ("elgrp_g2",)

    def test_recording_process_region(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_FALSE_START)
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_RECORDING_PROCESS

    def test_take_series_region(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RETRY)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CORRECTION)
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_TAKE_SERIES

    def test_clean_delivery_region(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_CLEAN_DELIVERY
        assert region.audience_delivery_status == AUDIENCE_DELIVERY_SUPPORTED

    def test_clean_delivery_region_mixed_with_new_audience_beat(self):
        # NEW_AUDIENCE_BEAT is still a "clean" structural role for
        # dominant_process_status purposes, but it is NOT itself
        # audience_delivery_status=SUPPORTED (mirrors D-194's own
        # classify_editorial_moment: only CLEAN_AUDIENCE_DELIVERY earns
        # AUDIENCE_DELIVERY_SUPPORTED) -- so the region's own aggregate
        # honestly reports PARTIAL, never a false SUPPORTED.
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_NEW_AUDIENCE_BEAT)
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_CLEAN_DELIVERY
        from cutsell_worker.editorial_moment_sequence import AUDIENCE_DELIVERY_PARTIAL
        assert region.audience_delivery_status == AUDIENCE_DELIVERY_PARTIAL

    def test_mixed_region(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_MIXED

    def test_unknown_region_from_uncertain_moments(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_UNCERTAIN)
        g = _group("src1", "g", [m1])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        assert region.dominant_process_status == p2.REGION_UNKNOWN

    def test_proposition_candidate_ids_are_union_of_member_moments(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_x",))
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_NEW_AUDIENCE_BEAT, proposition_ids=("prop_y",))
        g = _group("src1", "g", [m1, m2])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        assert set(region.proposition_candidate_ids) == {"prop_x", "prop_y"}

    def test_sequence_ids_only_included_when_subset_of_group(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m3 = _moment("src1", "c", 5.0, 6.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1, m2])
        seq_inside = _sequence("src1", "s_in", [m1, m2])
        seq_outside = _sequence("src1", "s_out", [m2, m3])
        (region,) = p2.build_whole_video_editorial_regions(
            source_asset_id="src1", moments=[m1, m2, m3], local_groups=[g], sequences=[seq_inside, seq_outside],
        )
        assert region.sequence_ids == ("eseq_s_in",)

    def test_region_ids_are_deterministic_and_membership_anchored(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1])
        r1 = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])[0]
        r2 = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])[0]
        assert r1.region_id == r2.region_id
        assert r1.region_id.startswith("wver_")

    def test_input_order_independence_for_regions(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS)
        m2 = _moment("src1", "b", 5.0, 6.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        forward = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g1, g2])
        backward = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m2, m1], local_groups=[g2, g1])
        assert [r.region_id for r in forward] == [r.region_id for r in backward]

    def test_source_identity_preserved_no_rendered_timeline(self):
        m1 = _moment("srcXYZ", "a", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("srcXYZ", "g", [m1])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="srcXYZ", moments=[m1], local_groups=[g])
        assert region.source_asset_id == "srcXYZ"
        assert region.source_start == 10.0 and region.source_end == 11.0

    def test_no_numeric_region_gap_constant_anywhere_in_module(self):
        assert "REGION_GAP_SECONDS" not in P2_CODE_ONLY_SOURCE


# ---------------------------------------------------------------------------
# 3. Proposition realization map (deliverable items 8-9, 26).
# ---------------------------------------------------------------------------
class TestPropositionRealizationMap:
    def test_single_realization(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_x",))
        g = _group("src1", "g", [m1])
        regions = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        (row,) = p2.build_whole_video_proposition_realization_maps(source_asset_id="src1", moments=[m1], regions=regions)
        assert row.relationship_status == p2.REALIZATION_SINGLE
        assert row.realization_count == 1

    def test_multiple_local_realizations_same_region(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RETRY, proposition_ids=("prop_x",))
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_x",))
        g = _group("src1", "g", [m1, m2])
        regions = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        (row,) = p2.build_whole_video_proposition_realization_maps(source_asset_id="src1", moments=[m1, m2], regions=regions)
        assert row.relationship_status == p2.REALIZATION_MULTIPLE_LOCAL

    def test_multiple_distant_realizations_different_regions(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_x",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_x",))
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        regions = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g1, g2])
        (row,) = p2.build_whole_video_proposition_realization_maps(source_asset_id="src1", moments=[m1, m2], regions=regions)
        assert row.relationship_status == p2.REALIZATION_MULTIPLE_DISTANT

    def test_distinct_propositions_never_merged(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_x",))
        m2 = _moment("src1", "b", 1.0, 2.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_y",))
        g = _group("src1", "g", [m1, m2])
        regions = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1, m2], local_groups=[g])
        rows = p2.build_whole_video_proposition_realization_maps(source_asset_id="src1", moments=[m1, m2], regions=regions)
        assert {r.proposition_candidate_id for r in rows} == {"prop_x", "prop_y"}

    def test_proposition_id_field_never_named_retry_family(self):
        row_fields = {f.name for f in fields(p2.WholeVideoPropositionRealizationMap)}
        assert "retry_family_id" not in row_fields
        assert "take_group_id" not in row_fields


# ---------------------------------------------------------------------------
# 4. Cross-source proposition correspondence (multi-source support).
# ---------------------------------------------------------------------------
class TestCrossSourceLinks:
    def test_same_proposition_across_two_sources(self):
        p_left = _proposition("srcA", "1", SIG_ALPHA, start=0.0, end=1.0)
        p_right = _proposition("srcB", "2", SIG_ALPHA_2, start=0.0, end=1.0)
        (link,) = p2.build_cross_source_proposition_links({p_left.proposition_candidate_id: p_left, p_right.proposition_candidate_id: p_right})
        assert link.relationship == p2.CROSS_SOURCE_SAME_PROPOSITION
        assert link.left_source_asset_id == "srcA" and link.right_source_asset_id == "srcB"

    def test_cross_source_conflict(self):
        p_left = _proposition("srcA", "1", SIG_ALPHA_NUM2, start=0.0, end=1.0)
        p_right = _proposition("srcB", "2", SIG_ALPHA_NUM3, start=0.0, end=1.0)
        (link,) = p2.build_cross_source_proposition_links({p_left.proposition_candidate_id: p_left, p_right.proposition_candidate_id: p_right})
        assert link.relationship == p2.CROSS_SOURCE_CONFLICTED

    def test_multi_source_isolation_no_identity_merge(self):
        p_left = _proposition("srcA", "1", SIG_ALPHA, start=0.0, end=1.0)
        p_right = _proposition("srcB", "2", SIG_ALPHA_2, start=0.0, end=1.0)
        (link,) = p2.build_cross_source_proposition_links({p_left.proposition_candidate_id: p_left, p_right.proposition_candidate_id: p_right})
        assert link.left_proposition_candidate_id != link.right_proposition_candidate_id
        assert link.left_source_asset_id != link.right_source_asset_id

    def test_same_source_pairs_never_produce_a_cross_source_link(self):
        p_left = _proposition("srcA", "1", SIG_ALPHA, start=0.0, end=1.0)
        p_right = _proposition("srcA", "2", SIG_ALPHA_2, start=5.0, end=6.0)
        links = p2.build_cross_source_proposition_links({p_left.proposition_candidate_id: p_left, p_right.proposition_candidate_id: p_right})
        assert links == ()

    def test_no_link_for_distinct_propositions(self):
        p_left = _proposition("srcA", "1", SIG_ALPHA, start=0.0, end=1.0)
        p_right = _proposition("srcB", "2", SIG_BETA, start=0.0, end=1.0)
        links = p2.build_cross_source_proposition_links({p_left.proposition_candidate_id: p_left, p_right.proposition_candidate_id: p_right})
        assert links == ()


# ---------------------------------------------------------------------------
# 5. Supersession hypotheses -- coverage / role / firewalls (deliverable
# items 13-22, D-201 contract replay A-E).
# ---------------------------------------------------------------------------
def _region(source_asset_id, seed, *, start, end, process_status, proposition_ids=(), confidence=CONFIDENCE_SUPPORTED):
    return p2.WholeVideoEditorialRegion(
        source_asset_id=source_asset_id, region_id=f"wver_{seed}", moment_ids=(f"emom_{seed}",),
        local_group_ids=(f"elgrp_{seed}",), sequence_ids=(), source_start=start, source_end=end,
        dominant_process_status=process_status, audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        proposition_candidate_ids=tuple(proposition_ids), confidence=confidence, conflict_flags=(),
        provenance=("TEST_FIXTURE",),
    )


class TestSupersessionHypotheses:
    def test_full_redundancy_supported_supersession(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_e",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_l",))
        props = {
            "prop_e": _proposition("src1", "e", SIG_ALPHA, start=0.0, end=1.0),
            "prop_l": _proposition("src1", "l", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.coverage_status == p2.COVERAGE_FULL
        assert hyp.supersession_status == p2.SUPERSESSION_SUPPORTED
        assert hyp.uncovered_earlier_proposition_candidate_ids == ()

    def test_d201_contract_A_multiple_retries_later_clean(self):
        earlier = _region("src1", "e", start=0.0, end=2.0, process_status=p2.REGION_TAKE_SERIES, proposition_ids=("prop_e1", "prop_e2"))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_l",))
        props = {
            "prop_e1": _proposition("src1", "e1", SIG_ALPHA, start=0.0, end=1.0),
            "prop_e2": _proposition("src1", "e2", SIG_ALPHA_2, start=1.0, end=2.0),
            "prop_l": _proposition("src1", "l", SIG_ALPHA, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.supersession_status in (p2.SUPERSESSION_SUPPORTED, p2.SUPERSESSION_PARTIAL)
        assert hyp.coverage_status in (p2.COVERAGE_FULL, p2.COVERAGE_PARTIAL)

    def test_d201_contract_B_unique_information_firewall_earlier_A_plus_B_later_A_only(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a", "prop_b"))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_b": _proposition("src1", "b", SIG_BETA, start=0.5, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.supersession_status != p2.SUPERSESSION_SUPPORTED
        assert hyp.supersession_status in (p2.SUPERSESSION_PARTIAL, p2.SUPERSESSION_NO_SAFE)
        assert "prop_b" in hyp.uncovered_earlier_proposition_candidate_ids
        assert "prop_a" in hyp.covered_proposition_candidate_ids

    def test_reverse_coverage_earlier_A_later_A_plus_B(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=12.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2", "prop_b2"))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
            "prop_b2": _proposition("src1", "b2", SIG_BETA, start=11.0, end=12.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.coverage_status == p2.COVERAGE_FULL
        assert hyp.supersession_status == p2.SUPERSESSION_SUPPORTED

    def test_d201_contract_C_contradictory_realization_negation_conflict(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a_neg",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a_neg": _proposition("src1", "aneg", SIG_ALPHA_NEGATED, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.meaning_conflict_status == p2.MEANING_CONFLICT_PRESENT
        assert hyp.supersession_status == p2.SUPERSESSION_CONFLICTED

    def test_number_contradiction_never_averaged(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_2",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_3",))
        props = {
            "prop_2": _proposition("src1", "n2", SIG_ALPHA_NUM2, start=0.0, end=1.0),
            "prop_3": _proposition("src1", "n3", SIG_ALPHA_NUM3, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.supersession_status == p2.SUPERSESSION_CONFLICTED
        assert hyp.confidence != p2.CONFIDENCE_SUPPORTED

    def test_d201_contract_D_distinct_propositions_no_full_supersession(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_beta",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_beta": _proposition("src1", "beta", SIG_BETA, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.supersession_status != p2.SUPERSESSION_SUPPORTED
        # Two entirely unrelated propositions share no comparable content at
        # all -- honestly UNKNOWN (nothing safely comparable), never a
        # fabricated NO_COVERAGE verdict from an absence of overlap alone.
        assert hyp.coverage_status == p2.COVERAGE_UNKNOWN
        assert hyp.supersession_status == p2.SUPERSESSION_UNKNOWN

    def test_d201_contract_E_chronology_reversal_does_not_change_semantic_truth(self):
        # Region content/roles are FIXED to (process, clean); only the
        # region ARGUMENT ORDER passed to the builder is reversed -- the
        # builder must derive earlier/later from source_start, never from
        # list position (input-order independence == the "reversed" half
        # of the chronology firewall).
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        forward = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        backward = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[later, earlier], proposition_candidates_by_id=props)
        assert forward == backward

    def test_chronology_firewall_role_not_position(self):
        # The region with process evidence is now the LATER one by
        # timestamp, and the clean-delivery region is EARLIER. Being later
        # must NOT by itself earn SUPPORTED_SUPERSESSION -- the algorithm
        # must check the chronologically-earlier region's OWN role for
        # process support (here: none) and the later region's OWN role for
        # delivery support (here: none, it's the process side).
        clean_but_early = _region("src1", "clean_early", start=0.0, end=1.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a",))
        process_but_late = _region("src1", "process_late", start=10.0, end=11.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a2",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(
            source_asset_id="src1", regions=[clean_but_early, process_but_late], proposition_candidates_by_id=props,
        )
        assert hyp.supersession_status != p2.SUPERSESSION_SUPPORTED

    def test_chronology_only_later_block_no_free_pass(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_UNKNOWN, proposition_ids=())
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_UNKNOWN, proposition_ids=())
        hyps = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id={})
        # No proposition evidence on either side at all -- nothing to
        # hypothesize about, so no hypothesis is even emitted (never a
        # fabricated SUPPORTED verdict from chronology alone).
        assert hyps == ()

    def test_clean_takes_no_supersession_evidence(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.supersession_status == p2.SUPERSESSION_NO_SAFE

    def test_multiple_propositions_partial_when_one_missing(self):
        earlier = _region("src1", "e", start=0.0, end=3.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a", "prop_beta", "prop_gamma"))
        later = _region("src1", "l", start=10.0, end=12.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2", "prop_gamma2"))
        sig_gamma = _signature({"gamma", "third", "claim"})
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_beta": _proposition("src1", "beta", SIG_BETA, start=1.0, end=2.0),
            "prop_gamma": _proposition("src1", "gamma", sig_gamma, start=2.0, end=3.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
            "prop_gamma2": _proposition("src1", "gamma2", sig_gamma, start=11.0, end=12.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        assert hyp.coverage_status == p2.COVERAGE_PARTIAL
        assert "prop_beta" in hyp.uncovered_earlier_proposition_candidate_ids
        assert hyp.supersession_status == p2.SUPERSESSION_PARTIAL

    def test_no_provider_no_llm_call_anywhere(self):
        for banned in ("openai", "gemini", "OpenAI", "Gemini", "whole_video_openai", "responses.create"):
            assert banned not in P2_CODE_ONLY_SOURCE

    def test_no_delete_selected_clip_or_action_field_on_any_type(self):
        banned_substrings = ("delete", "winner", "selected_clip_id", "final_winner", "action")
        for cls in (
            p2.WholeVideoEditorialRegion, p2.WholeVideoPropositionRealizationMap,
            p2.WholeVideoSupersessionHypothesis, p2.WholeVideoEditorialUnderstanding,
        ):
            for f in fields(cls):
                for banned in banned_substrings:
                    assert banned not in f.name.lower()

    def test_no_family_besttake_ordering_boundary_pacing_import(self):
        banned_modules = (
            "take_grouping", "take_grouping_provider", "composite_resolver", "realization_resolver",
            "bounded_finalist_arbiter", "bounded_finalist_authority", "boundary_engine_pass",
            "dialogue_pacing_transition", "semantic_ledger", "deterministic_best_take_authority",
        )
        for name in banned_modules:
            assert f"from .{name} import" not in P2_MODULE_SOURCE
            assert f"import {name}" not in P2_MODULE_SOURCE

    def test_no_commercial_or_sales_funnel_fields(self):
        banned_substrings = ("hook_strength", "conversion_score", "virality", "sales_funnel", "commercial_moment")
        for cls in (
            p2.WholeVideoEditorialRegion, p2.WholeVideoPropositionRealizationMap,
            p2.WholeVideoSupersessionHypothesis, p2.WholeVideoEditorialUnderstanding,
        ):
            for f in fields(cls):
                for banned in banned_substrings:
                    assert banned not in f.name.lower()
        for banned in banned_substrings:
            assert banned not in P2_MODULE_SOURCE.lower()


# ---------------------------------------------------------------------------
# 6. WholeVideoEditorialUnderstanding aggregate + diagnostics/run summary
# (deliverable items 6, 34-35, 18/... ).
# ---------------------------------------------------------------------------
class TestAggregateAndDiagnostics:
    def test_single_source_understanding(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a2",))
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        understanding = p2.build_whole_video_editorial_understanding(
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g1, g2]},
            proposition_candidates_by_id=props,
        )
        assert understanding.source_asset_ids == ("src1",)
        assert len(understanding.regions) == 2
        assert understanding.capability_status in p2.ALLOWED_CAPABILITY_STATUSES
        assert understanding.global_continuity_status in p2.ALLOWED_CONTINUITY_STATUSES

    def test_multi_source_understanding_no_identity_merge(self):
        m1 = _moment("srcA", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        m2 = _moment("srcB", "b", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g1 = _group("srcA", "g1", [m1])
        g2 = _group("srcB", "g2", [m2])
        understanding = p2.build_whole_video_editorial_understanding(
            moments_by_source={"srcA": [m1], "srcB": [m2]},
            local_groups_by_source={"srcA": [g1], "srcB": [g2]},
        )
        assert understanding.source_asset_ids == ("srcA", "srcB")
        assert {r.source_asset_id for r in understanding.regions} == {"srcA", "srcB"}
        # No supersession hypothesis crosses sources -- Phase A never
        # merges identities across source_asset_id.
        assert all(h.source_asset_id in ("srcA", "srcB") for h in understanding.supersession_hypotheses)

    def test_no_moments_returns_not_evaluable(self):
        understanding = p2.build_whole_video_editorial_understanding(moments_by_source={}, local_groups_by_source={})
        assert understanding.capability_status == p2.CAPABILITY_NOT_EVALUABLE
        assert understanding.global_continuity_status == p2.CONTINUITY_UNKNOWN

    def test_understanding_has_no_edit_plan_no_winners_no_ordering_fields(self):
        field_names = {f.name for f in fields(p2.WholeVideoEditorialUnderstanding)}
        for banned in ("edit_plan", "winners", "ordering_plan", "render_plan"):
            assert banned not in field_names

    def test_region_diagnostics_is_json_safe_and_bounded(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1])
        (region,) = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        diag = p2.whole_video_editorial_region_diagnostics(region)
        assert "transcript" not in diag
        assert isinstance(diag["moment_ids"], list)

    def test_supersession_diagnostics_no_transcript(self):
        earlier = _region("src1", "e", start=0.0, end=1.0, process_status=p2.REGION_RECORDING_PROCESS, proposition_ids=("prop_a",))
        later = _region("src1", "l", start=10.0, end=11.0, process_status=p2.REGION_CLEAN_DELIVERY, proposition_ids=("prop_a2",))
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        (hyp,) = p2.build_whole_video_supersession_hypotheses(source_asset_id="src1", regions=[earlier, later], proposition_candidates_by_id=props)
        diag = p2.whole_video_supersession_diagnostics(hyp)
        assert "text" not in diag and "transcript" not in diag

    def test_run_summary_has_no_master_score(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1])
        understanding = p2.build_whole_video_editorial_understanding(
            moments_by_source={"src1": [m1]}, local_groups_by_source={"src1": [g]},
        )
        summary = p2.whole_video_editorial_understanding_run_summary(understanding)
        for key in summary:
            assert "score" not in key or key.endswith("_count")
        assert all(not isinstance(v, float) for v in summary.values())


# ---------------------------------------------------------------------------
# 7. Determinism / repeatability.
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_deterministic_repeat(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_RECORDING_PROCESS, proposition_ids=("prop_a",))
        m2 = _moment("src1", "b", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_a2",))
        g1 = _group("src1", "g1", [m1])
        g2 = _group("src1", "g2", [m2])
        props = {
            "prop_a": _proposition("src1", "a", SIG_ALPHA, start=0.0, end=1.0),
            "prop_a2": _proposition("src1", "a2", SIG_ALPHA_2, start=10.0, end=11.0),
        }
        run1 = p2.build_whole_video_editorial_understanding(
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g1, g2]},
            proposition_candidates_by_id=props,
        )
        run2 = p2.build_whole_video_editorial_understanding(
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g1, g2]},
            proposition_candidates_by_id=props,
        )
        assert run1 == run2

    def test_stable_ids_across_calls(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1])
        r1 = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        r2 = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        assert r1[0].region_id == r2[0].region_id


# ---------------------------------------------------------------------------
# 8. SAME_EDITORIAL_BEAT / PREASSEMBLED_FINAL_SEQUENCE independence
# (deliverable items 23-24; D-201's own known non-blocking P1 limitation).
# ---------------------------------------------------------------------------
class TestP1LimitationIndependence:
    def test_no_same_editorial_beat_reference_in_module(self):
        assert "BEAT_SAME" not in P2_CODE_ONLY_SOURCE
        assert "SAME_EDITORIAL_BEAT" not in P2_CODE_ONLY_SOURCE

    def test_no_preassembled_final_sequence_requirement(self):
        assert "PREASSEMBLED_FINAL_SEQUENCE" not in P2_CODE_ONLY_SOURCE

    def test_functions_without_sequences_still_work(self):
        m1 = _moment("src1", "a", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY)
        g = _group("src1", "g", [m1])
        regions = p2.build_whole_video_editorial_regions(source_asset_id="src1", moments=[m1], local_groups=[g])
        assert len(regions) == 1
        assert regions[0].sequence_ids == ()


# ---------------------------------------------------------------------------
# 9. Structural no-authority / no-second-ontology / no-pipeline-wiring
# audits (deliverable items 36-48, 68).
# ---------------------------------------------------------------------------
class TestStructuralAudits:
    def test_pipeline_does_not_import_this_module(self):
        pipeline_source = (REPO_ROOT / "cutsell_worker" / "pipeline.py").read_text()
        assert "whole_video_editorial_reasoning" not in pipeline_source

    def test_flow_b_does_not_import_this_module(self):
        flow_b_source = (REPO_ROOT / "cutsell_worker" / "flow_b.py").read_text()
        assert "whole_video_editorial_reasoning" not in flow_b_source

    def test_no_qa_reference_fields_or_imports(self):
        banned = ("cut_ai", "cutai", "human_gold", "quality_ladder", "benchmark_label")
        for name in banned:
            assert name not in P2_MODULE_SOURCE.lower()

    def test_no_feature_flag_env_var(self):
        assert "os.environ" not in P2_MODULE_SOURCE
        assert "getenv" not in P2_MODULE_SOURCE

    def test_module_functions_are_pure_no_global_state(self):
        source_functions = [
            name for name, obj in vars(p2).items()
            if inspect.isfunction(obj) and obj.__module__ == p2.__name__
        ]
        assert len(source_functions) >= 8
        # No module-level mutable cache/registry declared anywhere.
        assert "_CACHE" not in P2_MODULE_SOURCE
        assert "global " not in P2_MODULE_SOURCE

    def test_no_hardcoded_video00_phrase_or_timestamp(self):
        banned = ("papillary", "gynaecologist", "stomach", "vamos", "34029861712", "34045158712")
        for phrase in banned:
            assert phrase not in P2_MODULE_SOURCE.lower()

    def test_conflict_preservation_no_majority_vote_language(self):
        # "majority vote"/"latest wins"/etc. are named ONLY as disclaimed
        # non-behaviors in the module's own docstring -- confirm none of
        # them appear as actual code (function names, branch logic).
        for phrase in ("majority vote", "latest wins", "most complete wins", "longest wins"):
            assert phrase not in P2_CODE_ONLY_SOURCE
        assert "def majority_vote" not in P2_MODULE_SOURCE
        assert "def latest_wins" not in P2_MODULE_SOURCE
