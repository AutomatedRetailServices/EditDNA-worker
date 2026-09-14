"""D-050B: Semantic Ledger, SHADOW MODE. See docs/CUTSELL_DECISIONS.md
D-050 (audit), D-050A (canonical identities), and D-050B (this module).

Every test proves one of:
  (1) the Ledger's own structural invariants (one minting owner, no
      duplicate/conflicting registration, orphan/cycle/unknown-parent
      detection, decision-history ordering and traversal), or
  (2) the shadow-reconstruction driver faithfully mirrors today's
      authoritative diagnostics (winner history, discard/replacement,
      composite membership, claims, CanonicalEditPlan/StoryValidator
      coverage parity), or
  (3) the two D-049 architectural gaps are OBSERVABLE in the Ledger
      without being fixed (Section 11's explicit ask), or
  (4) editorial behavior is completely unchanged by wiring the Ledger
      into universal_clean_cut.py (Section 13's shadow guarantee).

No test exercises the Ledger participating in a decision -- nothing in
this module or in universal_clean_cut.py ever branches on Ledger state.
"""
from dataclasses import replace

import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord,
    CompositeRecord,
    DELIVERY_SCORE_WINNER,
    DiscardRecord,
    LedgerIntegrityError,
    RealizationRecord,
    REPLACEMENT_DECLARED,
    SEMANTIC_WINNER_OVERRIDE,
    SemanticIdeaRecord,
    SemanticLedger,
    build_ledger_parity_report,
    build_semantic_ledger_diagnostics,
    build_semantic_ledger_shadow,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _clip(clip_id, text, *, selected, realization_id=None, semantic_idea_id=None,
          retry_family_id=None, render_fragment_id=None, parent_realization_id=None,
          start=0.0, end=1.0):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        realization_id=realization_id or f"real_{clip_id}",
        semantic_idea_id=semantic_idea_id, retry_family_id=retry_family_id,
        render_fragment_id=render_fragment_id, parent_realization_id=parent_realization_id,
    )


def _draft(*, selected=(), alternates=(), discarded=(), diagnostics=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=alternates, discarded=discarded,
        diagnostics=diagnostics or {},
    )


def _realization(realization_id, *, semantic_idea_id=None, state="selected", **overrides):
    fields = dict(
        realization_id=realization_id, semantic_idea_id=semantic_idea_id, retry_family_id=None,
        source_span_ids=(), attempt_id=None, clip_ids=(realization_id,), text="text",
        start=0.0, end=1.0, delivery_score=None, state=state, discard_reason=None,
        replacement_realization_id=None, claim_ids=(), render_fragment_ids=(),
    )
    fields.update(overrides)
    return RealizationRecord(**fields)


# ---------------------------------------------------------------------------
# 1. One semantic idea, multiple realizations / retry families
# ---------------------------------------------------------------------------

def test_one_semantic_idea_with_multiple_realizations():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r1", semantic_idea_id="idea_1"))
    ledger.register_realization(_realization("r2", semantic_idea_id="idea_1", state="discarded"))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r1", "r2"),
        canonical_claim_ids=(), current_winner_realization_id="r1",
        composite_realization_ids=(), coverage_status="complete", story_order_position=0,
    ))
    idea = ledger.ideas()["idea_1"]
    assert set(idea.realization_ids) == {"r1", "r2"}
    assert idea.current_winner_realization_id == "r1"


def test_multiple_distinct_semantic_ideas_are_independent():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r1", semantic_idea_id="idea_1"))
    ledger.register_realization(_realization("r2", semantic_idea_id="idea_2"))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r1",),
        canonical_claim_ids=(), current_winner_realization_id="r1",
        composite_realization_ids=(), coverage_status="complete", story_order_position=0,
    ))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_2", retry_family_ids=(), realization_ids=("r2",),
        canonical_claim_ids=(), current_winner_realization_id="r2",
        composite_realization_ids=(), coverage_status="complete", story_order_position=1,
    ))
    assert set(ledger.ideas()) == {"idea_1", "idea_2"}


# ---------------------------------------------------------------------------
# 2/9. Provisional winner -> winner override history, traversal
# ---------------------------------------------------------------------------

def test_provisional_winner_then_semantic_override_recorded_in_order():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_local", semantic_idea_id="idea_1"))
    ledger.register_realization(_realization("r_semantic", semantic_idea_id="idea_1"))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r_local", "r_semantic"),
        canonical_claim_ids=(), current_winner_realization_id=None,
        composite_realization_ids=(), coverage_status="complete", story_order_position=None,
    ))
    ledger.record_winner_decision(
        semantic_idea_id="idea_1", realization_id="r_local", stage="take_judge_provider",
        decision_type=DELIVERY_SCORE_WINNER, reason="top_score",
    )
    ledger.record_winner_decision(
        semantic_idea_id="idea_1", realization_id="r_semantic", stage="pipeline_semantic_best_take",
        decision_type=SEMANTIC_WINNER_OVERRIDE, reason="hybrid_override", previous_realization_id="r_local",
    )
    history = ledger.decision_history_for("r_semantic")
    assert [d.decision_type for d in history] == [SEMANTIC_WINNER_OVERRIDE]
    assert ledger.ideas()["idea_1"].current_winner_realization_id == "r_semantic"
    # Full decision log preserves BOTH events in order -- not just the final winner.
    all_types = [d.decision_type for d in ledger.decisions()]
    assert all_types == [DELIVERY_SCORE_WINNER, SEMANTIC_WINNER_OVERRIDE]
    assert ledger.decisions()[0].order_index < ledger.decisions()[1].order_index


# ---------------------------------------------------------------------------
# 4. Discard with verified replacement / without replacement (D-049 Case A)
# ---------------------------------------------------------------------------

def test_discard_with_verified_replacement():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_gone", semantic_idea_id="idea_1", state="discarded"))
    ledger.register_realization(_realization("r_replacement", semantic_idea_id="idea_1"))
    ledger.record_discard(
        DiscardRecord(
            discarded_realization_id="r_gone", discarding_stage="hybrid_editorial_chunks",
            reason="semantic_failed_plus_local_performance",
            replacement_realization_id="r_replacement", replacement_verified=True,
        ),
        stage="hybrid_editorial_chunks", semantic_idea_id="idea_1",
    )
    discard = ledger.discards()[0]
    assert discard.replacement_verified is True
    replacement_events = [d for d in ledger.decisions() if d.decision_type == REPLACEMENT_DECLARED]
    assert len(replacement_events) == 1
    assert replacement_events[0].subject_realization_id == "r_replacement"


def test_discard_without_replacement_is_faithfully_recorded_not_blocked():
    """D-049 Case A shape: OBSERVATION ONLY -- the Ledger must record the
    fact, never prevent it."""
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_gone", semantic_idea_id=None, state="discarded"))
    ledger.record_discard(
        DiscardRecord(
            discarded_realization_id="r_gone", discarding_stage="hybrid_editorial_chunks",
            reason="semantic_failed_plus_local_performance",
            replacement_realization_id=None, replacement_verified=False,
        ),
        stage="hybrid_editorial_chunks",
    )
    discard = ledger.discards()[0]
    assert discard.replacement_realization_id is None
    assert discard.replacement_verified is False
    # No REPLACEMENT_DECLARED event -- nothing to declare.
    assert not [d for d in ledger.decisions() if d.decision_type == REPLACEMENT_DECLARED]
    # And it is NOT reported as an orphan -- the discard record explains it.
    assert "r_gone" not in ledger.find_orphan_realizations()


# ---------------------------------------------------------------------------
# 6. Composite membership
# ---------------------------------------------------------------------------

def test_composite_membership_recorded():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_a", semantic_idea_id="idea_1"))
    ledger.register_realization(_realization("r_b", semantic_idea_id="idea_1"))
    ledger.record_composite(
        CompositeRecord(
            semantic_idea_id="idea_1", member_realization_ids=("r_a", "r_b"),
            composite_kind="claim_coverage_composite", reason="tg_1",
        ),
        stage="claim_coverage_best_take",
    )
    assert len(ledger.composites()) == 1
    assert ledger.composites()[0].member_realization_ids == ("r_a", "r_b")
    composite_events = [d for d in ledger.decisions() if d.decision_type == "COMPOSITE_CREATED"]
    assert len(composite_events) == 1


# ---------------------------------------------------------------------------
# 7. Canonical claims from multiple realizations (D-049 Case B shape)
# ---------------------------------------------------------------------------

def test_canonical_claims_from_multiple_realizations_visible_under_one_idea():
    ledger = SemanticLedger()
    ledger.register_realization(_realization(
        "r_rich", semantic_idea_id="idea_1", claim_ids=("cclaim_a", "cclaim_b"),
    ))
    ledger.register_realization(_realization(
        "r_vague", semantic_idea_id="idea_1", state="discarded", claim_ids=("cclaim_c",),
    ))
    ledger.register_claim(CanonicalClaimRecord(
        canonical_claim_id="cclaim_a", claim_type="NEGATION", content_tokens=frozenset({"no", "cree"}),
        importance="CRITICAL", source_realization_ids=("r_rich",), covered_by_realization_ids=("r_rich",),
        coverage_state="covered",
    ))
    ledger.register_claim(CanonicalClaimRecord(
        canonical_claim_id="cclaim_b", claim_type="MEASUREMENT_QUANTITY", content_tokens=frozenset({"cinco", "diez"}),
        importance="CRITICAL", source_realization_ids=("r_rich",), covered_by_realization_ids=("r_rich",),
        coverage_state="covered",
    ))
    ledger.register_claim(CanonicalClaimRecord(
        canonical_claim_id="cclaim_c", claim_type="MEASUREMENT_QUANTITY", content_tokens=frozenset({"cinco", "diez", "los"}),
        importance="CRITICAL", source_realization_ids=("r_vague",), covered_by_realization_ids=(),
        coverage_state="missing",
    ))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r_rich", "r_vague"),
        canonical_claim_ids=("cclaim_a", "cclaim_b", "cclaim_c"), current_winner_realization_id="r_rich",
        composite_realization_ids=(), coverage_status="complete", story_order_position=None,
    ))
    idea = ledger.ideas()["idea_1"]
    # Both the winner's own claims AND the discarded sibling's near-duplicate
    # measurement claim are visible under the SAME idea -- proving the
    # Ledger has the information a future D-050C dedup fix would need,
    # without deduping anything here.
    quantity_claims = [
        cid for cid in idea.canonical_claim_ids
        if ledger.claims()[cid].claim_type == "MEASUREMENT_QUANTITY"
    ]
    assert len(quantity_claims) == 2
    assert {ledger.claims()[cid].source_realization_ids[0] for cid in quantity_claims} == {"r_rich", "r_vague"}


# ---------------------------------------------------------------------------
# 8. Physical split descendants never become new realizations
# ---------------------------------------------------------------------------

def test_physical_fragment_attaches_to_existing_realization_never_creates_new_one():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_root", semantic_idea_id="idea_1"))
    ledger.record_physical_fragment(realization_id="r_root", render_fragment_id="frag_left")
    ledger.record_physical_fragment(realization_id="r_root", render_fragment_id="frag_right")
    assert set(ledger.realizations()) == {"r_root"}  # no new realization minted
    assert ledger.realizations()["r_root"].render_fragment_ids == ("frag_left", "frag_right")


def test_physical_fragment_for_unknown_realization_raises():
    ledger = SemanticLedger()
    with pytest.raises(LedgerIntegrityError):
        ledger.record_physical_fragment(realization_id="does_not_exist", render_fragment_id="frag")


# ---------------------------------------------------------------------------
# 10/11. Orphan / unknown-parent / cycle / duplicate detection
# ---------------------------------------------------------------------------

def test_orphan_detection_true_orphan_vs_explained_absence():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_true_orphan", semantic_idea_id=None))
    ledger.register_realization(_realization("r_explained", semantic_idea_id=None, state="discarded"))
    ledger.record_discard(
        DiscardRecord("r_explained", "hybrid_editorial_chunks", "deleted", None, False), stage="hybrid_editorial_chunks",
    )
    orphans = ledger.find_orphan_realizations()
    assert "r_true_orphan" in orphans
    assert "r_explained" not in orphans


def test_duplicate_realization_registration_with_conflicting_data_raises():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r1", semantic_idea_id="idea_1"))
    with pytest.raises(LedgerIntegrityError):
        ledger.register_realization(_realization("r1", semantic_idea_id="idea_2"))


def test_duplicate_realization_registration_with_identical_data_is_idempotent():
    ledger = SemanticLedger()
    record = _realization("r1", semantic_idea_id="idea_1")
    ledger.register_realization(record)
    ledger.register_realization(record)  # no raise
    assert len(ledger.realizations()) == 1


def test_duplicate_semantic_idea_registration_with_conflicting_data_raises():
    ledger = SemanticLedger()
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r1",),
        canonical_claim_ids=(), current_winner_realization_id="r1",
        composite_realization_ids=(), coverage_status="complete", story_order_position=0,
    ))
    with pytest.raises(LedgerIntegrityError):
        ledger.register_semantic_idea(SemanticIdeaRecord(
            semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("r2",),
            canonical_claim_ids=(), current_winner_realization_id="r2",
            composite_realization_ids=(), coverage_status="missing", story_order_position=0,
        ))


def test_unknown_parent_detection():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("r1", semantic_idea_id="idea_ghost"))
    unknown = ledger.find_unknown_parent_ids()
    assert "idea_ghost" in unknown


def test_provenance_cycle_detection():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("A", render_fragment_ids=("B",)))
    ledger.register_realization(_realization("B", render_fragment_ids=("A",)))
    cycles = ledger.find_provenance_cycles()
    assert cycles, "a mutual fragment reference must be detected as a cycle"


def test_no_provenance_cycle_in_a_legitimate_chain():
    ledger = SemanticLedger()
    ledger.register_realization(_realization("root", semantic_idea_id="idea_1", render_fragment_ids=("frag_a", "frag_b")))
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=("root",),
        canonical_claim_ids=(), current_winner_realization_id="root",
        composite_realization_ids=(), coverage_status="complete", story_order_position=0,
    ))
    assert ledger.find_provenance_cycles() == []


def test_no_duplicate_semantic_ids_by_construction():
    ledger = SemanticLedger()
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id="idea_1", retry_family_ids=(), realization_ids=(),
        canonical_claim_ids=(), current_winner_realization_id=None,
        composite_realization_ids=(), coverage_status="unknown", story_order_position=None,
    ))
    assert ledger.find_duplicate_semantic_ids() == []


# ---------------------------------------------------------------------------
# Shadow reconstruction driver: real-shape drafts
# ---------------------------------------------------------------------------

def test_shadow_reconstruction_basic_selected_and_discarded():
    winner = _clip("c_win", "Rich complete realization about a topic.", selected=True, semantic_idea_id="idea_1")
    loser = _clip("c_lose", "Rich complete realization about a topic siblings.", selected=False, semantic_idea_id="idea_1")
    draft = _draft(
        selected=(winner,), discarded=(loser,),
        diagnostics={"take_group_members": [["c_win", "c_lose"]]},
    )
    ledger = build_semantic_ledger_shadow(draft)
    assert ledger.realizations()[winner.realization_id].state == "selected"
    assert ledger.realizations()[loser.realization_id].state == "discarded"
    assert ledger.ideas()["idea_1"].current_winner_realization_id == winner.realization_id


def test_shadow_reconstruction_d049_case_a_discard_without_replacement():
    """D-049 Case A shadow observability: a hybrid_editorial_chunks delete
    with no verified replacement must show up as
    discarded/replacement_verified=false -- without fixing the behavior."""
    kept = _clip("c_kept", "La biopsia confirmo el diagnostico.", selected=True, semantic_idea_id="idea_papillary")
    deleted = _clip("c_deleted", "Sintomas que tuve segun yo era sintomatica.", selected=False)
    draft = _draft(
        selected=(kept,), discarded=(deleted,),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_deleted", "applied_delete": True,
                    "delete_basis": "semantic_failed_plus_local_performance",
                    "later_retry_replacement_id": None,
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    discard = next(d for d in ledger.discards() if d.discarded_realization_id == deleted.realization_id)
    assert discard.replacement_realization_id is None
    assert discard.replacement_verified is False
    assert discard.discarding_stage == "hybrid_editorial_chunks"


def test_shadow_reconstruction_discard_with_verified_replacement():
    replacement = _clip("c_repl", "Replacement realization text.", selected=True)
    deleted = _clip("c_del", "Original realization text now gone.", selected=False)
    draft = _draft(
        selected=(replacement,), discarded=(deleted,),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_del", "applied_delete": True, "delete_basis": "semantic_failed",
                    "later_retry_replacement_id": "c_repl",
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    discard = next(d for d in ledger.discards() if d.discarded_realization_id == deleted.realization_id)
    assert discard.replacement_realization_id == replacement.realization_id
    assert discard.replacement_verified is True


def test_shadow_reconstruction_d049_case_b_claim_inflation_visible():
    """D-049 Case B shadow observability: two sibling realizations under
    ONE semantic idea each contributing a near-duplicate statistic claim
    must both be visible -- without deduping them."""
    rich = _clip(
        "c_rich", "Por eso no creo que los cánceres sean hereditarios. Solo un 5-10% son de carácter hereditario.",
        selected=True, semantic_idea_id="idea_family",
    )
    vague = _clip(
        "c_vague", "Así que estoy convencida que solo un 5 -10 % de los casos son de carácter hereditario.",
        selected=False, semantic_idea_id="idea_family",
    )
    draft = _draft(selected=(rich,), discarded=(vague,))
    ledger = build_semantic_ledger_shadow(draft)
    idea = ledger.ideas()["idea_family"]
    quantity_claims = [
        cid for cid in idea.canonical_claim_ids
        if ledger.claims()[cid].claim_type == "MEASUREMENT_QUANTITY"
    ]
    sources = {ledger.claims()[cid].source_realization_ids[0] for cid in quantity_claims}
    # Not deduped: both realizations' quantity claims are present.
    assert rich.realization_id in {s for cid in quantity_claims for s in ledger.claims()[cid].source_realization_ids}
    assert vague.realization_id in {s for cid in quantity_claims for s in ledger.claims()[cid].source_realization_ids}


def test_shadow_reconstruction_physical_split_descendants():
    root = _clip("c_root", "one two three four", selected=True, semantic_idea_id="idea_1", realization_id="real_root")
    left = _clip(
        "c_root", "one two", selected=True, semantic_idea_id="idea_1",
        realization_id="real_root", render_fragment_id="frag_left", parent_realization_id="real_root",
    )
    right = _clip(
        "c_root", "three four", selected=True, semantic_idea_id="idea_1",
        realization_id="real_root", render_fragment_id="frag_right", parent_realization_id="real_root",
    )
    draft = _draft(selected=(left, right))
    ledger = build_semantic_ledger_shadow(draft)
    assert set(ledger.realizations()) == {"real_root"}
    assert set(ledger.realizations()["real_root"].render_fragment_ids) == {"frag_left", "frag_right"}


# ---------------------------------------------------------------------------
# Parity checker
# ---------------------------------------------------------------------------

def test_parity_report_clean_on_a_consistent_draft():
    winner = _clip("c_win", "A complete realization.", selected=True, semantic_idea_id="idea_1")
    draft = _draft(
        selected=(winner,),
        diagnostics={"canonical_edit_plan": {"ideas": [{"idea_id": "idea_1", "coverage_status": "complete"}]}},
    )
    ledger = build_semantic_ledger_shadow(draft)
    report = build_ledger_parity_report(ledger, draft)
    assert report.is_clean, report.mismatches


def test_parity_report_detects_canonical_edit_plan_coverage_mismatch():
    winner = _clip("c_win", "A complete realization.", selected=True, semantic_idea_id="idea_1")
    draft = _draft(
        selected=(winner,),
        # CanonicalEditPlan claims "missing" but the ledger (built from
        # the SAME draft, no coverage record added) still reports "unknown".
        diagnostics={"canonical_edit_plan": {"ideas": [{"idea_id": "idea_1", "coverage_status": "missing"}]}},
    )
    ledger = build_semantic_ledger_shadow(draft)
    report = build_ledger_parity_report(ledger, draft)
    assert not report.is_clean
    assert any(m.kind == "coverage_mismatch" for m in report.mismatches)


def test_diagnostics_builder_is_json_safe():
    winner = _clip("c_win", "A complete realization.", selected=True, semantic_idea_id="idea_1")
    draft = _draft(selected=(winner,))
    ledger = build_semantic_ledger_shadow(draft)
    parity = build_ledger_parity_report(ledger, draft)
    diag = build_semantic_ledger_diagnostics(ledger, parity)
    import json
    json.dumps(diag)  # must not raise
    assert diag["parity"]["is_clean"] is True


# ---------------------------------------------------------------------------
# 13/14. Behavioral shadow guarantee: wiring the Ledger into
# universal_clean_cut.py changes NOTHING about winner/order/discarded/
# ClaimCoverage/Freeze. Proven at the integration level by the full
# pre-existing tests/test_cutsell_*.py glob (1350 tests, CleanCutBench's
# 54 real-chain fixtures included) staying green with this module wired
# in -- see the D-050B commit's own validation section. This test adds a
# direct, minimal proof that the Ledger build call itself cannot throw
# and cannot mutate the draft it observes beyond adding the one new
# diagnostics key.
# ---------------------------------------------------------------------------

def test_building_the_ledger_does_not_mutate_selected_discarded_or_existing_diagnostics():
    winner = _clip("c_win", "A complete realization.", selected=True, semantic_idea_id="idea_1")
    loser = _clip("c_lose", "A discarded realization.", selected=False)
    draft = _draft(selected=(winner,), discarded=(loser,), diagnostics={"existing_key": "existing_value"})
    before_selected = draft.selected
    before_discarded = draft.discarded
    ledger = build_semantic_ledger_shadow(draft)
    parity = build_ledger_parity_report(ledger, draft)
    new_diagnostics = {**draft.diagnostics, "semantic_ledger": build_semantic_ledger_diagnostics(ledger, parity)}
    new_draft = replace(draft, diagnostics=new_diagnostics)
    assert new_draft.selected == before_selected
    assert new_draft.discarded == before_discarded
    assert new_draft.diagnostics["existing_key"] == "existing_value"
    assert "semantic_ledger" in new_draft.diagnostics
