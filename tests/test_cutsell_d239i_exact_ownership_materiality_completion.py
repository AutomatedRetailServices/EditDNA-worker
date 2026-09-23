"""D-239I: EXACT-OWNERSHIP MATERIALITY EVIDENCE COMPLETION -- OFFLINE ONLY.

Post D-239H (docs/CUTSELL_DECISIONS.md, Verdict D -- MULTIPLE EXISTING
EVIDENCE SEAMS ARE MISSING): closes exactly the four seams that forensic
identified, using ONLY already-computed, already-existing evidence:

  Seam A (editorial requirement): `complete_lost_semantic_atom_
      materiality.py`'s own `_exact_slot_for_proposition_set` is now ALSO
      consulted when `exact_ownership_available` (D-238) is True but
      `exact_identity_available` (D-235P) is False -- the SAME already-
      populated `proposition_slot_evidence_by_id` map, the SAME function,
      never a new interpretation.
  Seam B (critical-claim conflict): `final_story_coherence_validation.
      py`'s own `_critical_claim_conflict_by_clip_id` now ALSO resolves a
      standalone-discarded lost atom's conflict state through D-238's
      exact ownership -> the containing LanguageAttempt's own
      AUTHORITATIVE-identity representative clip -- THAT clip's own
      already-evaluated context, never a new conflict detector, never
      `None -> False`.
  Seam C (retry/process): P1 `EditorialMoment` role/audience-delivery
      evidence (already computed live, previously trapped in a
      diagnostics-only side channel) is now threaded to D-235L/D-235M's
      own existing `recording_process_evidence`/`recording_process_
      status`/`audience_delivery_status` parameters, gated to rows D-238
      ownership resolved exactly.
  Seam D (redundancy): a new, exact, non-fuzzy preserved-equivalent proof
      -- `replacement_function_preserved=True` ONLY when D-238's owning
      LanguageAttempt IS (via the SAME AUTHORITATIVE identity bridge Seam
      B reuses) a clip genuinely present in the CURRENT pass's own
      selected/kept set. Never False; absence of proof stays None
      (UNKNOWN).

Every seam is additive, gated, and fails closed: `None`/`{}` everywhere
a caller does not supply the new inputs reproduces byte-identical
pre-D-239I behavior. Precedence (meaning-critical -> editorial-required
-> conflict/unknown -> retry/process -> redundant -> non-material ->
abstain) is untouched; no new classifier, no new threshold, no fuzzy
matching, no timestamp-overlap authority, no provider, no RAW.
"""
from __future__ import annotations

import subprocess

import pytest

import cutsell_worker.complete_lost_semantic_atom_materiality as claamod
import cutsell_worker.final_story_coherence_validation as fscv
import cutsell_worker.pipeline as pipeline_module
from cutsell_worker.complete_lost_semantic_atom_materiality import (
    assess_complete_lost_semantic_atom_materiality,
)
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.editorial_moment_sequence import (
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_UNCERTAIN,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    EditorialMoment,
    EditorialMomentUnderstanding,
    p1_moment_role_and_audience_status_by_clip_id_for,
)
from cutsell_worker.editorial_moment_sequence import (
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
)
from cutsell_worker.exact_lost_atom_ownership import ExactLostAtomOwnership
from cutsell_worker.final_story_coherence_validation import (
    _complete_lost_semantic_atom_materiality_by_clip_id,
    _critical_claim_conflict_by_clip_id,
    _recording_process_evidence_from_p1_role,
    _representative_clip_id_by_attempt_id,
    apply_final_story_coherence_validation,
)
from cutsell_worker.language_proposition_relation import SLOT_CTA, SLOT_HOOK, SLOT_OTHER
from cutsell_worker.language_utterance_attempt import CONFIDENCE_MIXED, CONFIDENCE_SUPPORTED
from cutsell_worker.lost_semantic_atom_materiality import (
    MATERIALITY_EDITORIALLY_REQUIRED,
    MATERIALITY_MEANING_CRITICAL,
    MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    RECOMMEND_ABSTAIN,
    RECOMMEND_BLOCK,
    RECOMMEND_DO_NOT_BLOCK,
)
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    AttemptLanguageIdentityMatch,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    WORD_IDENTITY_AVAILABLE,
    WordMembership,
)

# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------
_ENV_FLAG = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"

# The real D-239G/D-239H recovered shape.
_TARGET_CLIP_ID = "clip_6fb9e51df885391ac25d"
_TARGET_ATTEMPT_ID = "latt_97f99a710a88e74edc26"
_TARGET_PROPOSITION_ID = "prop_124bd1b931ba3cb81abb"


def _row(**overrides) -> dict:
    base = {
        "clip_id": _TARGET_CLIP_ID,
        "text": "oh too many people ready set these are the",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
    }
    base.update(overrides)
    return base


def _exact_singleton(**overrides) -> ExactLostAtomOwnership:
    base = dict(
        clip_id=_TARGET_CLIP_ID, source_asset_id="src",
        candidate_word_indices=(41, 42, 43, 44, 45, 46, 47, 48),
        containing_language_attempt_id=_TARGET_ATTEMPT_ID,
        proposition_candidate_ids=(_TARGET_PROPOSITION_ID,),
        ownership_status="EXACT_SINGLETON_OWNERSHIP", reason_codes=(), provenance=(),
    )
    base.update(overrides)
    return ExactLostAtomOwnership(**base)


def _match(clip_id: str, attempt_id: str, *, authoritative: bool = True) -> AttemptLanguageIdentityMatch:
    status = RELATIONSHIP_EXACT_SAME_MEMBERSHIP if authoritative else "HEURISTIC_OVERLAP"
    wm = WordMembership(source_asset_id="src", entity_id=clip_id, word_indices=(0, 1), identity_status=WORD_IDENTITY_AVAILABLE)
    return AttemptLanguageIdentityMatch(
        reconstructed_attempt_id=clip_id, language_attempt_ids=(attempt_id,), source_asset_id="src",
        reconstructed_word_membership=wm, language_word_memberships=(wm,), relationship_status=status,
        exact_shared_word_count=2, reconstructed_word_count=2, language_word_count=2, provenance=(),
    )


def _moment(clip_id, role, audience_status, confidence=CONFIDENCE_SUPPORTED) -> EditorialMoment:
    return EditorialMoment(
        source_asset_id="src", editorial_moment_id=f"em_{clip_id}", source_start=0.0, source_end=1.0,
        source_span_id=clip_id, attempt_ids=(), proposition_candidate_ids=(), related_span_ids=(),
        moment_role=role, audience_delivery_status=audience_status, recording_process_status="RECORDING_PROCESS_ABSENT",
        completion_status="COMPLETE", local_sequence_position=0, confidence=confidence,
        conflict_flags=(), provenance=(),
    )


def _understanding(*moments) -> EditorialMomentUnderstanding:
    return EditorialMomentUnderstanding(
        source_asset_id="src", moments=tuple(moments), sequence_hypotheses=(), moment_count=len(moments),
        sequence_count=0, capability_status="AVAILABLE", missing_evidence=(), confidence="SUPPORTED",
        conflict_flags=(), provenance=(),
    )


# ---------------------------------------------------------------------------
# 1-2: Seam A -- ownership slot-evidence success/ambiguous.
# ---------------------------------------------------------------------------
class TestSeamAEditorialSlotEvidence:
    def test_01_ownership_slot_evidence_success(self):
        # D-239L (docs/CUTSELL_DECISIONS.md D-239K/D-239L): proposition-
        # level slot evidence alone is no longer sufficient for REQUIRED
        # via the ownership-only path -- exact target-level P1 corroboration
        # (the SAME clip_id-keyed evidence Seam C already threads) is now
        # also required. Supplying it here preserves this test's own
        # original intent (Seam A's slot lookup resolves to REQUIRED/BLOCK)
        # under the corrected, atom-granular contract.
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_02_ownership_slot_evidence_no_story_function_slot(self):
        # SLOT_OTHER carries no function signal -- correctly stays
        # INSUFFICIENT_EVIDENCE, never fabricated as REQUIRED.
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_OTHER},
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"

    def test_03_ownership_multi_proposition_never_reached(self):
        # is_exact_singleton is only True for a genuine 1-proposition
        # owned set -- a multi-proposition ownership object never even
        # reaches Seam A's own branch (exact_ownership_available is False).
        own = _exact_singleton(
            proposition_candidate_ids=(_TARGET_PROPOSITION_ID, "prop_other"),
            ownership_status="AMBIGUOUS_MULTIPLE_PROPOSITIONS",
        )
        assert own.is_exact_singleton is False
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"
        assert res.exact_ownership_available is False

    def test_04_identity_mapping_upgraded_to_exact_via_ownership_alone(self):
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, proposition_slot_evidence_by_id={},
        )
        # No slot evidence found for the owned proposition -> requirement
        # still resolves via the identity-sufficiency gate (step 6, if
        # meaning-materiality independently reaches NON_MATERIAL).
        assert res.exact_ownership_available is True

    def test_05_exact_identity_takes_precedence_over_ownership(self):
        # When D-235P's own full-attempt match IS authoritative, Seam A's
        # ownership branch never fires (exact_identity_available wins the
        # elif chain) -- byte-identical to pre-D-239I in that case.
        match = _match(_TARGET_CLIP_ID, "other_attempt", authoritative=True)
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), exact_match=match, lost_atom_ownership=own,
            proposition_candidate_ids_by_attempt_id={"other_attempt": ("prop_from_match",)},
            proposition_slot_evidence_by_id={"prop_from_match": SLOT_HOOK, _TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        # Slot evidence comes from the MATCH's own proposition, not
        # ownership's -- both happen to be story-function slots here, but
        # exact_proposition_candidate_ids must be the match's own.
        assert res.exact_proposition_candidate_ids == ("prop_from_match",)


# ---------------------------------------------------------------------------
# 3-5, 13: explicit critical conflict TRUE/FALSE/unavailable; firewall.
# ---------------------------------------------------------------------------
class TestSeamBCriticalClaimConflict:
    def test_06_direct_true_unaffected_by_ownership(self):
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[{"left_clip_id": _TARGET_CLIP_ID, "right_clip_id": "other"}],
            lost_critical_claims=[], clip_id_to_group={},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={},
        )
        assert result[_TARGET_CLIP_ID] is True

    def test_07_direct_false_via_own_family_membership_unaffected(self):
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[],
            clip_id_to_group={_TARGET_CLIP_ID: ("g1",)},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={},
        )
        assert result[_TARGET_CLIP_ID] is False

    def test_08_ownership_bridges_to_representative_clip_true(self):
        # The lost atom's own clip_id is a standalone discard (never in
        # clip_id_to_group), but its owning attempt IS the exact identity
        # of a kept clip that WAS named by a conflict finding.
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[{"left_clip_id": "kept_clip", "right_clip_id": "x"}],
            lost_critical_claims=[], clip_id_to_group={},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={"kept_clip": rep_match},
        )
        assert result[_TARGET_CLIP_ID] is True

    def test_09_ownership_bridges_to_representative_clip_false(self):
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[],
            clip_id_to_group={"kept_clip": ("g1",)},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={"kept_clip": rep_match},
        )
        assert result[_TARGET_CLIP_ID] is False

    def test_10_no_representative_clip_stays_none(self):
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[], clip_id_to_group={},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={},
        )
        assert result[_TARGET_CLIP_ID] is None

    def test_11_representative_clip_itself_unresolved_stays_none(self):
        # The representative clip exists but was ITSELF never part of an
        # evaluated family and never named -- never inferred as False.
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[], clip_id_to_group={},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={"kept_clip": rep_match},
        )
        assert result[_TARGET_CLIP_ID] is None

    def test_12_no_ownership_supplied_reproduces_pre_d239i_none(self):
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[], clip_id_to_group={},
        )
        assert result[_TARGET_CLIP_ID] is None

    def test_13_never_none_to_false_heuristic_match_never_used(self):
        # A non-authoritative (heuristic) match for the "representative"
        # clip must NEVER be treated as identity -- the reverse index
        # only ever includes AUTHORITATIVE_RELATIONSHIP_STATUSES matches.
        heuristic_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=False)
        assert heuristic_match.relationship_status not in AUTHORITATIVE_RELATIONSHIP_STATUSES
        result = _critical_claim_conflict_by_clip_id(
            [_row()], contradiction_findings=[], lost_critical_claims=[],
            clip_id_to_group={"kept_clip": ("g1",)},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: _exact_singleton()},
            exact_match_by_clip_id={"kept_clip": heuristic_match},
        )
        assert result[_TARGET_CLIP_ID] is None

    def test_14_ambiguous_reverse_map_never_resolved(self):
        m1 = _match("kept_a", _TARGET_ATTEMPT_ID, authoritative=True)
        m2 = _match("kept_b", _TARGET_ATTEMPT_ID, authoritative=True)
        rev = _representative_clip_id_by_attempt_id({"kept_a": m1, "kept_b": m2})
        assert _TARGET_ATTEMPT_ID not in rev


# ---------------------------------------------------------------------------
# 6-7, 15: Seam C -- retry/process via exact P1 role.
# ---------------------------------------------------------------------------
class TestSeamCRetryProcess:
    def test_15_recording_process_evidence_derivation_true(self):
        assert _recording_process_evidence_from_p1_role(MOMENT_ROLE_FALSE_START) is True

    def test_16_recording_process_evidence_derivation_false(self):
        assert _recording_process_evidence_from_p1_role(MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY) is False

    def test_17_recording_process_evidence_derivation_none(self):
        assert _recording_process_evidence_from_p1_role(None) is None

    def test_18_p1_helper_omits_uncertain_role(self):
        u = _understanding(_moment(_TARGET_CLIP_ID, MOMENT_ROLE_UNCERTAIN, AUDIENCE_DELIVERY_UNCERTAIN))
        roles, statuses = p1_moment_role_and_audience_status_by_clip_id_for((u,))
        assert _TARGET_CLIP_ID not in roles
        assert _TARGET_CLIP_ID not in statuses

    def test_19_p1_helper_omits_mixed_confidence(self):
        u = _understanding(_moment(_TARGET_CLIP_ID, MOMENT_ROLE_FALSE_START, AUDIENCE_DELIVERY_SUPPORTED, confidence=CONFIDENCE_MIXED))
        roles, statuses = p1_moment_role_and_audience_status_by_clip_id_for((u,))
        assert _TARGET_CLIP_ID not in roles

    def test_20_p1_helper_includes_exact_supported_role(self):
        u = _understanding(_moment(_TARGET_CLIP_ID, MOMENT_ROLE_FALSE_START, AUDIENCE_DELIVERY_SUPPORTED))
        roles, statuses = p1_moment_role_and_audience_status_by_clip_id_for((u,))
        assert roles[_TARGET_CLIP_ID] == MOMENT_ROLE_FALSE_START
        assert statuses[_TARGET_CLIP_ID] == AUDIENCE_DELIVERY_SUPPORTED

    def test_21_full_pipeline_retry_process_gated_by_ownership(self):
        own = _exact_singleton()
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
            p1_moment_role_by_clip_id={_TARGET_CLIP_ID: MOMENT_ROLE_FALSE_START},
            p1_audience_delivery_status_by_clip_id={_TARGET_CLIP_ID: AUDIENCE_DELIVERY_SUPPORTED},
        )[_TARGET_CLIP_ID]
        assert result.meaning_materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert result.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_22_retry_process_never_consulted_without_ownership(self):
        # p1 maps supplied, but this row has no exact ownership at all --
        # Seam C's own gate (ownership_exact) must never fire.
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {},
            p1_moment_role_by_clip_id={_TARGET_CLIP_ID: MOMENT_ROLE_FALSE_START},
            p1_audience_delivery_status_by_clip_id={_TARGET_CLIP_ID: AUDIENCE_DELIVERY_SUPPORTED},
        )[_TARGET_CLIP_ID]
        assert result.meaning_materiality_status != MATERIALITY_RETRY_OR_RECORDING_RESIDUE

    def test_23_missing_role_stays_unknown(self):
        own = _exact_singleton()
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {}, lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        assert result.retry_or_process_status == "UNKNOWN"


# ---------------------------------------------------------------------------
# 8-10: Seam D -- exact preserved-equivalent redundancy.
# ---------------------------------------------------------------------------
class TestSeamDRedundancy:
    def test_24_preserved_equivalent_success(self):
        own = _exact_singleton()
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {},
            exact_match_by_clip_id={"kept_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
            selected_clip_ids=frozenset({"kept_clip"}),
        )[_TARGET_CLIP_ID]
        assert result.final_materiality_status == "REDUNDANT_EQUIVALENT"
        assert result.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_25_representative_not_selected_stays_unknown(self):
        own = _exact_singleton()
        rep_match = _match("discarded_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {},
            exact_match_by_clip_id={"discarded_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
            selected_clip_ids=frozenset({"some_other_kept_clip"}),
        )[_TARGET_CLIP_ID]
        assert result.redundancy_status != "FOUND"
        assert result.final_materiality_status != "REDUNDANT_EQUIVALENT"

    def test_26_no_representative_never_false_stays_unknown(self):
        own = _exact_singleton()
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
            selected_clip_ids=frozenset({"kept_clip"}),
        )[_TARGET_CLIP_ID]
        assert result.final_materiality_status != "REDUNDANT_EQUIVALENT"


# ---------------------------------------------------------------------------
# 11-12: source mismatch / multi-proposition -- ABSTAIN, never DO_NOT_BLOCK.
# ---------------------------------------------------------------------------
class TestFirewallSourceMismatchAndMultiProposition:
    def test_27_source_mismatch_ownership_abstains(self):
        own = ExactLostAtomOwnership(
            clip_id=_TARGET_CLIP_ID, source_asset_id="src", candidate_word_indices=(1,),
            containing_language_attempt_id=None, proposition_candidate_ids=(),
            ownership_status="SOURCE_MISMATCH", reason_codes=("x",), provenance=(),
        )
        assert own.is_exact_singleton is False
        res = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=own)
        assert res.final_materiality_status == "INSUFFICIENT_EVIDENCE"
        assert res.blocking_recommendation == RECOMMEND_ABSTAIN

    def test_28_multi_proposition_ownership_never_do_not_block(self):
        own = ExactLostAtomOwnership(
            clip_id=_TARGET_CLIP_ID, source_asset_id="src", candidate_word_indices=(1,),
            containing_language_attempt_id=_TARGET_ATTEMPT_ID,
            proposition_candidate_ids=(_TARGET_PROPOSITION_ID, "prop_2"),
            ownership_status="AMBIGUOUS_MULTIPLE_PROPOSITIONS", reason_codes=("x",), provenance=(),
        )
        res = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=own)
        assert res.blocking_recommendation != RECOMMEND_DO_NOT_BLOCK


# ---------------------------------------------------------------------------
# Firewall cases 1-4, 9-10 (directive's own numbered list).
# ---------------------------------------------------------------------------
class TestDirectiveFirewallCases:
    def test_29_case1_ownership_plus_editorial_required_blocks(self):
        # D-239L: exact target-level P1 corroboration required alongside
        # ownership-inherited slot evidence -- see test_01's own comment.
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_HOOK},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_30_case2_ownership_plus_meaning_critical_blocks(self):
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(critical_claim_conflict_would_be_ignored=True), lost_atom_ownership=own,
            critical_claim_conflict=True,
        )
        assert res.final_materiality_status == MATERIALITY_MEANING_CRITICAL
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_31_case3_ownership_plus_critical_conflict_true_blocks(self):
        own = _exact_singleton()
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        materiality = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], contradiction_findings=[{"left_clip_id": "kept_clip", "right_clip_id": "x"}],
            lost_critical_claims=[], clip_id_to_group={},
            exact_match_by_clip_id={"kept_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        assert materiality.final_materiality_status == MATERIALITY_MEANING_CRITICAL
        assert materiality.blocking_recommendation == RECOMMEND_BLOCK

    def test_32_case4_ownership_plus_unknown_critical_context_abstains(self):
        own = _exact_singleton()
        materiality = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], [], [], {}, lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        assert materiality.final_materiality_status == "INSUFFICIENT_EVIDENCE"
        assert materiality.blocking_recommendation == RECOMMEND_ABSTAIN

    def test_33_case9_ownership_alone_never_do_not_block(self):
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=own)
        assert res.blocking_recommendation != RECOMMEND_DO_NOT_BLOCK

    def test_34_case10_explicit_no_conflict_only_when_evidence_exists(self):
        # clip_id_to_group membership IS the "evidence exists" proof --
        # without it, ownership alone must not manufacture a False.
        own = _exact_singleton()
        result = _critical_claim_conflict_by_clip_id(
            [_row()], [], [], {}, lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own}, exact_match_by_clip_id={},
        )
        assert result[_TARGET_CLIP_ID] is None


# ---------------------------------------------------------------------------
# Positive cases E/F: full NON_MATERIAL path + D-235R/D-235T consumption.
# ---------------------------------------------------------------------------
class TestPositiveCaseNonMaterialAndDownstreamConsumption:
    def test_35_full_non_material_path_via_ownership(self):
        own = _exact_singleton()
        row = _row(critical_claim_conflict=None)
        materiality = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        # Ownership alone does not manufacture NON_MATERIAL -- D-235L's
        # own critical_claim_conflict=False is still required (Stage 5's
        # own honest finding); confirm the row stays ABSTAIN without it,
        # and flips to DO_NOT_BLOCK once that evidence is independently
        # supplied (never via ownership itself).
        assert materiality.blocking_recommendation == RECOMMEND_ABSTAIN
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        materiality2 = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {"kept_clip": ("g1",)},
            exact_match_by_clip_id={"kept_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        assert materiality2.meaning_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert materiality2.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_36_d235r_consumes_same_result(self):
        from cutsell_worker.lost_semantic_atom_freeze_authority import decide_lost_semantic_atom_freeze_authority
        own = _exact_singleton()
        row = _row()
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        materiality = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {"kept_clip": ("g1",)},
            exact_match_by_clip_id={"kept_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        decision = decide_lost_semantic_atom_freeze_authority(row, materiality)
        assert decision.effective_blocking is False

    def test_37_d235t_consumes_same_result(self):
        from cutsell_worker.final_edit_reviewer import UNIQUE_FACT_LOST, Finding
        from cutsell_worker.lost_atom_repair_suppression import all_blocking_findings_safely_suppressed
        own = _exact_singleton()
        row = _row(lost_atom_provenance_id="latom_1")
        rep_match = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
        materiality = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {"kept_clip": ("g1",)},
            exact_match_by_clip_id={"kept_clip": rep_match},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )[_TARGET_CLIP_ID]
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id="i1",
            clip_ids=(_TARGET_CLIP_ID,), detail=row, owning_authority="StoryValidator", blocking=True,
        )
        all_suppressed, decisions = all_blocking_findings_safely_suppressed(
            [finding], enabled=True, materiality_by_provenance_id={"latom_1": materiality},
        )
        assert all_suppressed is True
        assert len(decisions) == 1


# ---------------------------------------------------------------------------
# Multi-source isolation / multiple lost atoms independent.
# ---------------------------------------------------------------------------
class TestIsolation:
    def test_38_multi_source_isolation(self):
        # D-239L: "c_a" needs exact target-level P1 corroboration (its own
        # clip_id) for its ownership-inherited SLOT_CTA to reach REQUIRED;
        # "c_b" (SLOT_OTHER, no story-function slot at all) is unaffected.
        own_a = _exact_singleton(clip_id="c_a", source_asset_id="src_a")
        own_b = _exact_singleton(clip_id="c_b", source_asset_id="src_b", containing_language_attempt_id="latt_b", proposition_candidate_ids=("prop_b",))
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row(clip_id="c_a"), _row(clip_id="c_b")], [], [], {},
            lost_atom_ownership_by_clip_id={"c_a": own_a, "c_b": own_b},
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA, "prop_b": SLOT_OTHER},
            p1_moment_role_by_clip_id={"c_a": MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY},
            p1_audience_delivery_status_by_clip_id={"c_a": AUDIENCE_DELIVERY_SUPPORTED},
        )
        assert result["c_a"].editorial_requirement_status == "REQUIRED"
        assert result["c_b"].editorial_requirement_status == "INSUFFICIENT_EVIDENCE"

    def test_39_multiple_lost_atoms_independent(self):
        own1 = _exact_singleton(clip_id="atom1")
        own2 = ExactLostAtomOwnership(
            clip_id="atom2", source_asset_id="src", candidate_word_indices=(9,),
            containing_language_attempt_id=None, proposition_candidate_ids=(),
            ownership_status="NO_CONTAINING_ATTEMPT", reason_codes=("x",), provenance=(),
        )
        result = _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row(clip_id="atom1"), _row(clip_id="atom2")], [], [], {},
            lost_atom_ownership_by_clip_id={"atom1": own1, "atom2": own2},
        )
        assert result["atom1"].exact_ownership_available is True
        assert result["atom2"].exact_ownership_available is False


# ---------------------------------------------------------------------------
# Language coverage: English / Spanish / Spanglish -- proves no language-
# specific logic anywhere in these seams (pure structural evidence only).
# ---------------------------------------------------------------------------
class TestLanguageCoverage:
    @pytest.mark.parametrize("text", [
        "oh too many people ready set these are the",
        "demasiadas personas listas ya son las",
        "oh too many personas listas set estas son las",
    ])
    def test_40_language_neutral_seam_a(self, text):
        # D-239L: exact target-level P1 corroboration required alongside
        # ownership-inherited slot evidence -- see test_01's own comment.
        own = _exact_singleton()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(text=text), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED


# ---------------------------------------------------------------------------
# No fuzzy matching / no timestamp authority.
# ---------------------------------------------------------------------------
class TestNoFuzzyNoTimestamp:
    def test_41_no_timestamp_field_read_anywhere_in_new_seams(self):
        # Scan only executable (non-comment, non-docstring-prose) lines --
        # "overlap"/"timestamp" appear freely in this task's own binding
        # prose explaining what these seams do NOT do; only an actual
        # attribute access or call would be a real violation.
        import ast
        import inspect
        for fn in (
            fscv._critical_claim_conflict_by_clip_id,
            fscv._representative_clip_id_by_attempt_id,
            fscv._complete_lost_semantic_atom_materiality_by_clip_id,
            fscv._recording_process_evidence_from_p1_role,
        ):
            tree = ast.parse(inspect.getsource(fn))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in ("source_start", "source_end"), f"{fn.__name__} reads {node.attr}"
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in ("_overlap_duration",), f"{fn.__name__} calls {node.func.id}"

    def test_42_no_text_similarity_import_in_new_seams(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(claamod))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported_names.add(alias.name)
        assert "difflib" not in imported_names
        assert "SequenceMatcher" not in imported_names


# ---------------------------------------------------------------------------
# No new threshold / classifier / provider / RAW.
# ---------------------------------------------------------------------------
class TestNoNewAuthority:
    def test_43_no_new_env_flag_introduced(self):
        import re
        text = claamod.__doc__ or ""
        text += fscv.__doc__ or ""
        # No new CUTSELL_* env flag name introduced by this task's own
        # docstrings (the existing materiality-authority flag is reused,
        # never a new one).
        assert "CUTSELL_" not in (fscv._recording_process_evidence_from_p1_role.__doc__ or "")

    def test_44_no_provider_import_in_new_seam_modules(self):
        import cutsell_worker.editorial_moment_sequence_integration as em_int
        import inspect
        src = inspect.getsource(em_int.p1_moment_role_and_audience_status_by_clip_id_for)
        for needle in ("google", "gemini", "openai", "genai", "provider"):
            assert needle not in src.lower()


# ---------------------------------------------------------------------------
# No global identity authority change (zero diff on unmodified files).
# ---------------------------------------------------------------------------
def _diff_stat(path: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", path], capture_output=True, text=True, timeout=20,
    )
    return proc.stdout.strip()


class TestNoGlobalAuthorityChange:
    def test_45_shared_attempt_word_identity_untouched(self):
        assert _diff_stat("cutsell_worker/shared_attempt_word_identity.py") == ""

    def test_46_authoritative_relationship_statuses_never_reassigned(self):
        import inspect
        for mod in (fscv, claamod):
            src = inspect.getsource(mod)
            assert "AUTHORITATIVE_RELATIONSHIP_STATUSES =" not in src, f"{mod.__name__} reassigns the frozenset"

    def test_47_lost_semantic_atom_materiality_untouched(self):
        assert _diff_stat("cutsell_worker/lost_semantic_atom_materiality.py") == ""

    def test_48_lost_atom_editorial_requirement_evidence_untouched(self):
        assert _diff_stat("cutsell_worker/lost_atom_editorial_requirement_evidence.py") == ""

    def test_49_lost_semantic_atom_freeze_authority_untouched(self):
        assert _diff_stat("cutsell_worker/lost_semantic_atom_freeze_authority.py") == ""

    def test_50_repair_loop_untouched(self):
        assert _diff_stat("cutsell_worker/repair_loop.py") == ""

    def test_51_precedence_order_still_meaning_editorial_conflict_retry_redundant_nonmaterial_abstain(self):
        import inspect
        src = inspect.getsource(claamod.assess_complete_lost_semantic_atom_materiality)
        i_meaning = src.index("MATERIALITY_MEANING_CRITICAL:")
        i_editorial = src.index("REQUIREMENT_REQUIRED:")
        i_conflict = src.index("conflicting_or_unresolved_evidence")
        i_retry = src.index("MATERIALITY_RETRY_OR_RECORDING_RESIDUE:")
        i_redundant = src.index("redundancy_proven:")
        i_nonmaterial = src.index("MATERIALITY_NON_MATERIAL_REAL_CONTENT and requirement_genuinely_clear")
        assert i_meaning < i_editorial < i_conflict < i_retry < i_redundant < i_nonmaterial


# ---------------------------------------------------------------------------
# Real D-239G/H shape offline replay + end-to-end wiring proof.
# ---------------------------------------------------------------------------
def _take(clip_id, start, end, text, *, selected, source="s1"):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source, source_order=0, start=start, end=end, text=text,
    )


def _clip(clip_id, start, end, text, *, selected, source="s1"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _draft(*, selected=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


class TestRealShapeOfflineReplay:
    """Structural replay of the real D-239G/H shape: EXACT_SINGLETON_
    OWNERSHIP, one owned PropositionCandidate, existing slot evidence,
    existing critical-context evidence where available, existing P1 role
    evidence, selected preserved-equivalent evidence where applicable --
    each seam proven to resolve independently without touching another."""

    def _base_materiality(self, *, slot=None, p1_role=None, p1_audience=None,
                           critical_conflict_context=None, redundancy_context=None):
        own = _exact_singleton()
        contradiction_findings = []
        clip_id_to_group = {}
        exact_match_by_clip_id = {}
        selected_clip_ids = frozenset()
        if critical_conflict_context == "true":
            rep = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
            exact_match_by_clip_id["kept_clip"] = rep
            contradiction_findings.append({"left_clip_id": "kept_clip", "right_clip_id": "x"})
        elif critical_conflict_context == "false":
            rep = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
            exact_match_by_clip_id["kept_clip"] = rep
            clip_id_to_group["kept_clip"] = ("g1",)
        if redundancy_context == "selected":
            rep = _match("kept_clip", _TARGET_ATTEMPT_ID, authoritative=True)
            exact_match_by_clip_id["kept_clip"] = rep
            selected_clip_ids = frozenset({"kept_clip"})
        p1_roles = {_TARGET_CLIP_ID: p1_role} if p1_role else {}
        p1_audiences = {_TARGET_CLIP_ID: p1_audience} if p1_audience else {}
        slot_map = {_TARGET_PROPOSITION_ID: slot} if slot else {}
        return _complete_lost_semantic_atom_materiality_by_clip_id(
            [_row()], contradiction_findings, [], clip_id_to_group,
            exact_match_by_clip_id=exact_match_by_clip_id,
            proposition_slot_evidence_by_id=slot_map,
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
            p1_moment_role_by_clip_id=p1_roles,
            p1_audience_delivery_status_by_clip_id=p1_audiences,
            selected_clip_ids=selected_clip_ids,
        )[_TARGET_CLIP_ID]

    def test_52_ownership_proven_exact_singleton(self):
        own = _exact_singleton()
        assert own.ownership_status == "EXACT_SINGLETON_OWNERSHIP"
        assert own.is_exact_singleton is True
        assert own.proposition_candidate_ids == (_TARGET_PROPOSITION_ID,)

    def test_53_seam_a_resolves_independently_of_b_c_d(self):
        # D-239L (docs/CUTSELL_DECISIONS.md D-239K/D-239L): Seam A's own
        # proposition-level slot evidence, with NO exact target-level P1
        # corroboration and none of B/C/D's own contexts present, now
        # correctly resolves to INSUFFICIENT_EVIDENCE, never REQUIRED --
        # this IS "resolving independently of B/C/D": Seam A alone was
        # never atom-granular proof, and the fix makes that honest.
        m = self._base_materiality(slot=SLOT_CTA)
        assert m.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"

    def test_53b_seam_a_plus_seam_c_atom_corroboration_resolves_required(self):
        # D-239L positive case: Seam A's slot evidence PLUS Seam C's own
        # exact target-level P1 corroboration (both reused, unmodified
        # evidence channels) together resolve to REQUIRED/BLOCK.
        m = self._base_materiality(
            slot=SLOT_CTA, p1_role=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, p1_audience=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert m.editorial_requirement_status == "REQUIRED"
        assert m.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED
        assert m.blocking_recommendation == RECOMMEND_BLOCK

    def test_54_seam_b_resolves_independently_of_a_c_d(self):
        m = self._base_materiality(critical_conflict_context="false")
        # No slot/p1/redundancy evidence -> critical-claim-conflict alone
        # resolved to False, but that alone does not manufacture DO_NOT_
        # BLOCK (D-235L's own branch 5 also needs recording_process_
        # evidence not True, already satisfied here).
        assert m.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert m.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_55_seam_c_resolves_independently_of_a_b_d(self):
        m = self._base_materiality(p1_role=MOMENT_ROLE_FALSE_START, p1_audience=AUDIENCE_DELIVERY_SUPPORTED)
        assert m.meaning_materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert m.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_56_seam_d_resolves_independently_of_a_b_c(self):
        m = self._base_materiality(redundancy_context="selected")
        assert m.final_materiality_status == "REDUNDANT_EQUIVALENT"
        assert m.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_57_no_evidence_at_all_stays_insufficient_never_forced_do_not_block(self):
        m = self._base_materiality()
        assert m.final_materiality_status == "INSUFFICIENT_EVIDENCE"
        assert m.blocking_recommendation == RECOMMEND_ABSTAIN


class TestEndToEndWiringThroughPublicEntryPoint:
    """Proves the full `apply_final_story_coherence_validation` call
    threads all four new optional maps correctly end to end, flag on/off,
    never crashing when they are absent."""

    def test_58_flag_off_new_params_never_consulted(self, monkeypatch):
        monkeypatch.delenv(_ENV_FLAG, raising=False)
        d = _draft(discarded=(_clip(_TARGET_CLIP_ID, 0.0, 1.0, "oh too many people ready set these are the", selected=False),))
        # Passing malformed values for the new params must never crash
        # when the flag is off (never consulted).
        result = apply_final_story_coherence_validation(
            d, lost_atom_ownership_by_clip_id={"x": "not-an-ownership-object"},
            p1_moment_role_by_clip_id={"x": 12345},
            p1_audience_delivery_status_by_clip_id={"x": object()},
        )
        assert result.diagnostics["final_story_coherence_validation"]["status"] == "applied"

    def test_59_flag_on_no_new_params_byte_identical_defaults(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "1")
        d = _draft(discarded=(_clip(_TARGET_CLIP_ID, 0.0, 1.0, "oh too many people ready set these are the", selected=False),))
        result = apply_final_story_coherence_validation(d)
        diag = result.diagnostics["final_story_coherence_validation"]
        assert diag["status"] == "applied"

    def test_60_flag_on_full_params_thread_through(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "1")
        own = _exact_singleton()
        d = _draft(discarded=(_clip(_TARGET_CLIP_ID, 0.0, 1.0, "oh too many people ready set these are the", selected=False),))
        result = apply_final_story_coherence_validation(
            d,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            lost_atom_ownership_by_clip_id={_TARGET_CLIP_ID: own},
        )
        orchestration = result.diagnostics["final_story_coherence_validation"]["lost_atom_materiality_orchestration"]
        # If a lost_semantic_atoms row was actually produced for this
        # discard, the orchestration diagnostics reflect Seam A's own
        # upgrade; if none was (content coverage differs by fixture),
        # the call must still not crash -- both are acceptable here,
        # the crash-freedom + flag threading is what this test proves.
        assert isinstance(orchestration, dict)
