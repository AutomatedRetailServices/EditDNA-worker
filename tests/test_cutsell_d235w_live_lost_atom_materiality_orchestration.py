"""D-235W: COMPLETE LIVE LOST-ATOM MATERIALITY ORCHESTRATION WIRING.

Covers the task's own required fixture matrix at the ONE orchestration
seam this task's directive authorizes: `final_story_coherence_validation.
apply_final_story_coherence_validation()`'s Part A (live critical-claim
context) + Part C (live exact word/proposition identity plumbing) wiring
into D-235Q/R. Mirrors the established D-235J-V source-code-truth +
fixture-matrix test style.

See docs/CUTSELL_DECISIONS.md D-235W for the full design note, including
the honestly-reported residual gap this task does NOT close: D-235T's own
`decide_lost_atom_repair_suppression()` recomputes materiality fresh from
the row alone (no critical_claim_conflict/exact_match override) and is
untouched by this task -- its own same-atom NON_MATERIAL_REAL_CONTENT
reachability stays exactly as it was before D-235W (see
TestD235TResidualGapUnchanged below).
"""
from __future__ import annotations

import cutsell_worker.final_story_coherence_validation as fscv
from cutsell_worker.complete_lost_semantic_atom_materiality import (
    CompleteLostSemanticAtomMateriality,
    assess_complete_lost_semantic_atom_materiality,
)
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import Finding, UNIQUE_FACT_LOST
from cutsell_worker.final_story_coherence_validation import (
    _clip_id_to_group_members,
    _complete_lost_semantic_atom_materiality_by_clip_id,
    _critical_claim_conflict_by_clip_id,
    _lost_atom_materiality_orchestration_diagnostics,
    apply_final_story_coherence_validation,
)
from cutsell_worker.lost_atom_repair_suppression import decide_lost_atom_repair_suppression
from cutsell_worker.lost_semantic_atom_freeze_authority import (
    AUTHORITY_ABSTAIN_PRESERVE_BLOCK,
    AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK,
    decide_lost_semantic_atom_freeze_authority,
    lost_semantic_atom_freeze_trigger_present,
)
from cutsell_worker.shared_attempt_word_identity import (
    AttemptLanguageIdentityMatch,
    RELATIONSHIP_DISJOINT,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    WordMembership,
)

_ENV_FLAG = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"


# ---------------------------------------------------------------------------
# Fixture helpers (mirror test_cutsell_final_story_coherence_validation.py's
# own `clip`/`draft`/`ranked_row`, plus the D-235Q suite's `_row`/`_match`).
# ---------------------------------------------------------------------------
def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), alternates=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=selected,
        alternates=alternates,
        discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


def ranked_row(clip_id, score, reason="watch_listen_baseline"):
    return {"clip_id": clip_id, "score": score, "reason": reason}


def d235u_row(**overrides):
    """The D-235U real-shape replay fixture (never the literal phrase):
    REAL_CONTENT_LOSS, no missing critical atoms, no atom classifications
    -- the exact shape D-235V's own forensic identified as structurally
    unreachable pre-D-235W."""
    base = {
        "clip_id": "c_d235u",
        "text": "a generic lost fragment of real speech",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
        "missing_critical_atoms": (),
        "atom_classifications": (),
        "own_content_token_count": 20,
        "coverage_against_final_keep": 0.9,
    }
    base.update(overrides)
    return base


def evaluated_group(clip_id, other_id="c_other"):
    """A genuine >=2-member `take_judge_groups` entry naming `clip_id` --
    the same shape `_lost_critical_claims`/`_contradiction_findings` above
    it in the module iterate, and the ONLY thing Part A's tri-state `False`
    is allowed to rest on (never a globally-empty list)."""
    return [{"group_id": "g1", "ranked": [{"clip_id": clip_id}, {"clip_id": other_id}]}]


def exact_match(status=RELATIONSHIP_EXACT_SAME_MEMBERSHIP, lang_ids=("latt1",), entity_id="r1", source="s1"):
    rm = WordMembership(source, entity_id, (0, 1), "AVAILABLE")
    lms = tuple(WordMembership(source, lid, (0, 1), "AVAILABLE") for lid in lang_ids)
    return AttemptLanguageIdentityMatch(
        reconstructed_attempt_id=entity_id, language_attempt_ids=tuple(lang_ids), source_asset_id=source,
        reconstructed_word_membership=rm, language_word_memberships=lms,
        relationship_status=status, exact_shared_word_count=2,
        reconstructed_word_count=2, language_word_count=2, provenance=("test",),
    )


# ---------------------------------------------------------------------------
# Part A: tri-state per-clip critical-claim-conflict context (11-17 in the
# directive's own fixture matrix numbering).
# ---------------------------------------------------------------------------
class TestPartATriState:
    def test_true_from_contradiction_finding(self):
        row = d235u_row()
        result = _critical_claim_conflict_by_clip_id(
            [row], [{"left_clip_id": "c_d235u", "right_clip_id": "c_other", "number_conflict": True}],
            [], {},
        )
        assert result["c_d235u"] is True

    def test_true_from_lost_critical_claim(self):
        row = d235u_row()
        result = _critical_claim_conflict_by_clip_id(
            [row], [], [{"source_clip_id": "c_d235u", "canonical_claim_id": "cc1"}], {},
        )
        assert result["c_d235u"] is True

    def test_false_from_evaluated_family_no_conflict(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        result = _critical_claim_conflict_by_clip_id([row], [], [], clip_id_to_group)
        assert result["c_d235u"] is False

    def test_none_when_never_evaluated(self):
        row = d235u_row()
        result = _critical_claim_conflict_by_clip_id([row], [], [], {})
        assert result["c_d235u"] is None

    def test_do_not_over_correlate_empty_global_lists_still_none_without_group(self):
        """A merely-empty global contradiction/claim list must NEVER be
        read as a clearance -- only genuine evaluated-family membership
        may produce `False` (the directive's own "Do not over-correlate"
        requirement)."""
        row = d235u_row(clip_id="c_ungrouped")
        result = _critical_claim_conflict_by_clip_id([row], [], [], {})
        assert result["c_ungrouped"] is None

    def test_conflict_on_one_clip_never_bleeds_to_sibling(self):
        row_a = d235u_row(clip_id="c_a")
        row_b = d235u_row(clip_id="c_b")
        clip_id_to_group = _clip_id_to_group_members(
            [{"group_id": "g1", "ranked": [{"clip_id": "c_a"}, {"clip_id": "c_b"}]}]
        )
        result = _critical_claim_conflict_by_clip_id(
            [row_a, row_b], [{"left_clip_id": "c_a", "right_clip_id": "c_x"}], [], clip_id_to_group,
        )
        assert result["c_a"] is True
        assert result["c_b"] is False


# ---------------------------------------------------------------------------
# Part A+C combined orchestration seam (18-22, 27+ structural safety).
# ---------------------------------------------------------------------------
class TestCombinedOrchestrationSeam:
    def test_d235u_shape_reaches_non_material_do_not_block_with_parts_a_and_c(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert isinstance(m, CompleteLostSemanticAtomMateriality)
        assert m.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert m.blocking_recommendation == "DO_NOT_BLOCK"
        assert m.exact_identity_available is True

    def test_part_a_alone_without_part_c_stays_insufficient_not_non_material(self):
        """Design-phase finding (see docs/CUTSELL_DECISIONS.md D-235W):
        Part A's `critical_claim_conflict=False` alone is NOT sufficient
        for this atom shape -- D-235Q's own editorial-requirement branch 0
        still needs exact identity. Proves Parts A and C are jointly
        necessary, never independently sufficient, for this shape."""
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.final_materiality_status != "NON_MATERIAL_REAL_CONTENT"
        assert m.blocking_recommendation != "DO_NOT_BLOCK"

    def test_part_c_alone_without_part_a_stays_insufficient(self):
        row = d235u_row()
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},  # never evaluated -> critical_claim_conflict None
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.final_materiality_status != "NON_MATERIAL_REAL_CONTENT"

    def test_full_chain_reaches_freeze_authority_suppression(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        decision = decide_lost_semantic_atom_freeze_authority(row, materiality_by_clip_id["c_d235u"])
        assert decision.authority_status == AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        assert decision.suppression_applied is True
        assert decision.effective_blocking is False

    def test_freeze_trigger_present_false_when_flag_on_and_context_supplied(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        assert lost_semantic_atom_freeze_trigger_present(
            [row], materiality_by_clip_id=materiality_by_clip_id, enabled=True,
        ) is False

    def test_freeze_trigger_present_ignores_materiality_when_flag_off(self):
        """Mandatory byte-identical default-off parity: even a fully-
        suppressible materiality map is IGNORED when the flag is off."""
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        assert lost_semantic_atom_freeze_trigger_present(
            [row], materiality_by_clip_id=materiality_by_clip_id, enabled=False,
        ) is True


# ---------------------------------------------------------------------------
# Safety controls (the directive's own 6 numbered controls).
# ---------------------------------------------------------------------------
class TestSafetyControls:
    def test_1_lost_critical_claim_forces_block_even_with_exact_identity(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [{"source_clip_id": "c_d235u", "canonical_claim_id": "cc1"}], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.meaning_materiality_status == "MEANING_CRITICAL"
        decision = decide_lost_semantic_atom_freeze_authority(row, m)
        assert decision.effective_blocking is True
        assert decision.suppression_applied is False

    def test_2_contradiction_preserves_block(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [{"left_clip_id": "c_d235u", "right_clip_id": "c_other"}], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        decision = decide_lost_semantic_atom_freeze_authority(row, materiality_by_clip_id["c_d235u"])
        assert decision.effective_blocking is True

    def test_3_unknown_critical_claim_context_abstains_preserves(self):
        row = d235u_row()
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},  # ungrouped -> None, never guessed False
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        decision = decide_lost_semantic_atom_freeze_authority(row, materiality_by_clip_id["c_d235u"])
        assert decision.effective_blocking is True
        assert decision.authority_status == AUTHORITY_ABSTAIN_PRESERVE_BLOCK

    def test_4_editorially_required_blocks(self):
        """`idea_coverage_status` is one of D-235Q's own pre-existing,
        unmodified editorial-requirement signals -- not a Part A/B/C
        plumbing target this task adds -- so this control is proven at the
        D-235Q entry point directly (`assess_complete_lost_semantic_atom_
        materiality`), exactly as D-235Q's own test suite already does."""
        row = d235u_row()
        m = assess_complete_lost_semantic_atom_materiality(
            row, critical_claim_conflict=False, exact_match=exact_match(),
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
            idea_coverage_status=True,
        )
        assert m.final_materiality_status == "EDITORIALLY_REQUIRED"
        decision = decide_lost_semantic_atom_freeze_authority(row, m)
        assert decision.effective_blocking is True

    def test_5_identity_unavailable_fails_closed(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,  # no exact_match_by_clip_id at all
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.exact_identity_available is False
        decision = decide_lost_semantic_atom_freeze_authority(row, m)
        assert decision.effective_blocking is True

    def test_6_heuristic_only_relationship_status_fails_closed(self):
        row = d235u_row()
        clip_id_to_group = _clip_id_to_group_members(evaluated_group("c_d235u"))
        non_authoritative = exact_match(status=RELATIONSHIP_DISJOINT)
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], clip_id_to_group,
            exact_match_by_clip_id={"c_d235u": non_authoritative},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.exact_identity_available is False
        decision = decide_lost_semantic_atom_freeze_authority(row, m)
        assert decision.effective_blocking is True


# ---------------------------------------------------------------------------
# Regression: retry/redundant reachability must stay unaffected by D-235W.
# ---------------------------------------------------------------------------
class TestNoRegressionOnRowNativeCategories:
    def test_retry_or_recording_residue_still_reachable_with_no_context(self):
        row = d235u_row(pre_group_restart_consultations=[{"same_idea": True}])
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},  # no group, no exact identity, no context at all
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.final_materiality_status == "RETRY_OR_RECORDING_RESIDUE"
        assert m.blocking_recommendation == "DO_NOT_BLOCK"

    def test_redundant_equivalent_still_reachable_with_no_context(self):
        row = d235u_row(content_loss_suppressed_by="tg_1")
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.final_materiality_status == "REDUNDANT_EQUIVALENT"

    def test_meaning_critical_atom_still_reachable_unconditionally(self):
        row = d235u_row(atom_classifications=({"importance": "CRITICAL"},))
        materiality_by_clip_id = _complete_lost_semantic_atom_materiality_by_clip_id(
            [row], [], [], {},
        )
        m = materiality_by_clip_id["c_d235u"]
        assert m.final_materiality_status == "MEANING_CRITICAL"


# ---------------------------------------------------------------------------
# Flag parity (25-26).
# ---------------------------------------------------------------------------
class TestFlagParity:
    def test_flag_off_apply_final_story_coherence_validation_unaffected(self, monkeypatch):
        monkeypatch.delenv(_ENV_FLAG, raising=False)
        row = d235u_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d235u", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d235u"))
        out = apply_final_story_coherence_validation(d)
        result = out.diagnostics["final_story_coherence_validation"]
        assert result["freeze_blocked"] is True
        orchestration = result["lost_atom_materiality_orchestration"]
        assert orchestration["lost_atom_complete_materiality_status"] == {}
        assert orchestration["lost_atom_critical_claim_conflict_status"] == {}

    def test_flag_on_with_full_context_unblocks_freeze(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "1")
        row = d235u_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d235u", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d235u"))
        out = apply_final_story_coherence_validation(
            d,
            exact_match_by_clip_id={"c_d235u": exact_match()},
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        result = out.diagnostics["final_story_coherence_validation"]
        assert result["freeze_blocked"] is False
        orchestration = result["lost_atom_materiality_orchestration"]
        assert orchestration["lost_atom_complete_materiality_status"]["c_d235u"] == "NON_MATERIAL_REAL_CONTENT"
        assert orchestration["lost_atom_blocking_recommendation"]["c_d235u"] == "DO_NOT_BLOCK"
        assert orchestration["lost_atom_freeze_authority_status"]["c_d235u"] == AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        assert orchestration["lost_atom_exact_word_identity_available"]["c_d235u"] is True
        assert orchestration["lost_atom_critical_claim_conflict_status"]["c_d235u"] is False
        assert orchestration["lost_atom_repair_suppression_status"] == "NOT_COMPUTED_AT_THIS_SEAM_SEE_REPAIR_LOOP"


# ---------------------------------------------------------------------------
# Structural safety (no fuzzy text, no timestamp authority, no P1/P2/
# BestTake/Family/Ordering/Boundary/Pacing/Audio-Join mutation).
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_critical_claim_conflict_helper_never_reads_row_text(self):
        import inspect
        source = inspect.getsource(_critical_claim_conflict_by_clip_id)
        assert '.get("text")' not in source
        assert ".get('text')" not in source

    def test_no_fuzzy_text_matching_in_orchestration_seam(self):
        import inspect
        source = inspect.getsource(_complete_lost_semantic_atom_materiality_by_clip_id)
        for banned in ("difflib", "SequenceMatcher", "fuzz"):
            assert banned not in source

    def test_module_does_not_import_p1_p2_grouping_authority(self):
        with open(fscv.__file__) as f:
            source = f.read()
        for banned in (
            "deterministic_best_take_authority",
            "boundary_engine_pass",
            "dialogue_pacing_transition",
        ):
            assert f"import {banned}" not in source
            assert f"from .{banned}" not in source

    def test_no_new_env_flag_introduced(self):
        with open(fscv.__file__) as f:
            content = f.read()
        # No `os.environ`/`getenv` call in this module at all -- the ONE
        # flag this seam consults is read exclusively through the already-
        # existing `lost_atom_materiality_freeze_authority_enabled()`
        # import from lost_semantic_atom_freeze_authority.py; D-235W adds
        # no new flag and no new direct environment read.
        assert "os.environ" not in content
        assert "os.getenv" not in content


# ---------------------------------------------------------------------------
# D-235T residual gap -- honestly documented, NOT closed by this task, and
# NOT a regression (identical behavior before and after D-235W: D-235T's
# own call never received this context either way).
# ---------------------------------------------------------------------------
class TestD235TResidualGapUnchanged:
    def test_repair_suppression_still_abstains_for_d235u_shape(self):
        """D-235T's own `decide_lost_atom_repair_suppression()` recomputes
        materiality FRESH from the row alone -- no critical_claim_conflict,
        no exact_match override -- so it cannot reach NON_MATERIAL_REAL_
        CONTENT for this shape regardless of what Parts A/C now supply at
        the Freeze-composition seam. This is the honestly-reported verdict-
        D residual gap; asserting it here pins the CURRENT (unchanged, pre-
        existing) behavior so a future change to close it is a deliberate,
        visible decision, not a silent regression in either direction."""
        row = d235u_row(lost_atom_provenance_id="prov1")
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id=None,
            clip_ids=("c_d235u",), detail=dict(row), owning_authority="StoryValidator", blocking=True,
        )
        decision = decide_lost_atom_repair_suppression(finding, all_findings=(finding,), enabled=True)
        assert decision.suppress_repair_escalation is False
        assert decision.suppression_status == "ABSTAIN_PRESERVE_ESCALATION"
