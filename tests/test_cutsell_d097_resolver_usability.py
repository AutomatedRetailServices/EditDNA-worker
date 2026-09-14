"""D-097 (PO adjustment §3) -- RESOLVER LEVEL-1 FIXES.

D-096 root cause #1 / collision C-4: the authoritative Resolver optimised
CRITICAL-claim coverage so strongly that it RESTORED a semantically failed
restatement into a composite after Best Take had chosen one winner. Under
test: a family-scoped hybrid "failed" label at >= 0.85 is evidence that a
realization is NOT USABLE for restoration/composite -- (1) its exclusive
CRITICAL requirement groups are waived (recorded) and mirrored into the
D-089 effective-importance index as SUPPORTING so StoryValidator never
re-blocks on them, (2) a usable realization always outranks it, (3) it is
never a composite member. Labels never delete: when EVERY candidate is
failed the evidence cancels out and the idea resolves exactly as before.
Legitimate composites (usable complementary pieces) are preserved.
Generic fixtures only.
"""
from cutsell_worker.realization_resolver import (
    EFFECTIVE_IMPORTANCE_WAIVED_FAILED_REALIZATION_SOURCE,
    RESOLVED_COMPOSITE,
    RESOLVED_WINNER,
    build_effective_claim_importance_index,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord, DELIVERY_SCORE_WINNER, RealizationRecord, SemanticIdeaRecord, SemanticLedger,
)

IDEA = "idea_generic_conclusion"


def _claim(cid, tokens, importance="CRITICAL", claim_type="STATE_RESULT"):
    return CanonicalClaimRecord(
        canonical_claim_id=cid, claim_type=claim_type, content_tokens=frozenset(tokens), importance=importance,
        source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved", text=" ".join(tokens),
    )


def _realization(rid, *, claim_ids, state, label="", confidence=0.0, start=0.0, complete_idea=True):
    return RealizationRecord(
        realization_id=rid, semantic_idea_id=IDEA, retry_family_id=IDEA, source_span_ids=(), attempt_id=None,
        clip_ids=(rid,), text=" ".join(claim_ids), start=start, end=start + 5.0, delivery_score=None, state=state,
        discard_reason=None if state == "selected" else "clean_cut_or_composite_resolution",
        replacement_realization_id=None, claim_ids=tuple(claim_ids), render_fragment_ids=(),
        complete_idea=complete_idea, semantic_label=label, semantic_label_confidence=confidence,
    )


def _ledger(realizations, claims, scores=None):
    ledger = SemanticLedger()
    for r in realizations:
        ledger.register_realization(r)
    for c in claims:
        ledger.register_claim(c)
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id=IDEA, retry_family_ids=(IDEA,), realization_ids=tuple(r.realization_id for r in realizations),
        canonical_claim_ids=tuple(c.canonical_claim_id for c in claims), current_winner_realization_id=None,
        composite_realization_ids=(), coverage_status="unresolved_ambiguous", story_order_position=None,
    ))
    if scores:
        ledger.record_winner_decision(
            semantic_idea_id=IDEA, realization_id=max(scores, key=scores.get), stage="take_judge_provider",
            decision_type=DELIVERY_SCORE_WINNER, reason="watch_listen_baseline_top_score",
            evidence={"ranked": [{"clip_id": k, "score": v} for k, v in scores.items()]},
        )
    return ledger


def _conclusion_ledger(*, loser_label="failed", loser_confidence=0.88, complementary=False):
    # W: the complete winner; F: a later restatement carrying one extra
    # CRITICAL-classified statistic that W does not cover. With
    # `complementary=True` F carries ONLY the statistic (a genuine
    # complementary piece: no single realization covers every CRITICAL group,
    # so absent usability evidence the Resolver must composite W+F). Without
    # it F restates W's content too, so D-063 CRITICAL_COVERAGE_DOMINANCE
    # would make F the outright winner whenever it is usable.
    experience = _claim("c_exp", ("shared", "experience", "family"))
    statistic = _claim("c_stat", ("most", "cases", "hereditary", "percent"))
    loser_claims = ("c_stat",) if complementary else ("c_exp", "c_stat")
    winner = _realization("real_W", claim_ids=("c_exp",), state="selected", label="winner", confidence=0.96)
    loser = _realization("real_F", claim_ids=loser_claims, state="discarded", label=loser_label, confidence=loser_confidence, start=20.0)
    claims = [
        CanonicalClaimRecord(**{**experience.__dict__, "source_realization_ids": ("real_W",) if complementary else ("real_W", "real_F")}),
        CanonicalClaimRecord(**{**statistic.__dict__, "source_realization_ids": ("real_F",)}),
    ]
    return _ledger((winner, loser), claims, scores={"real_W": 0.63, "real_F": 0.60})


def test_failed_restatement_is_not_restored_into_a_composite():
    ledger = _conclusion_ledger()
    resolution = resolve_realizations_shadow(ledger).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_WINNER
    assert resolution.winner_realization_id == "real_W"
    assert resolution.composite_realization_ids == ()
    assert "real_F" in resolution.discarded_realization_ids
    assert resolution.evidence["unusable_realization_ids"] == ["real_F"]
    assert resolution.evidence["critical_groups_waived_from_failed_realizations"]


def test_waived_claim_is_downgraded_in_the_effective_importance_index():
    index = build_effective_claim_importance_index(_conclusion_ledger())
    assert index["c_stat"].effective_importance == "SUPPORTING"
    assert index["c_stat"].reason == EFFECTIVE_IMPORTANCE_WAIVED_FAILED_REALIZATION_SOURCE
    assert index["c_exp"].effective_importance == "CRITICAL"


def test_a_usable_complementary_piece_still_forms_a_legitimate_composite():
    ledger = _conclusion_ledger(loser_label="alternate", loser_confidence=0.75, complementary=True)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_COMPOSITE
    assert set(resolution.composite_realization_ids) == {"real_W", "real_F"}
    assert resolution.evidence["unusable_realization_ids"] == []
    index = build_effective_claim_importance_index(ledger)
    assert index["c_stat"].effective_importance == "CRITICAL"


def test_a_failed_complementary_piece_is_never_a_composite_member():
    ledger = _conclusion_ledger(loser_label="failed", loser_confidence=0.90, complementary=True)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_WINNER
    assert resolution.winner_realization_id == "real_W"
    assert resolution.composite_realization_ids == ()
    assert "real_F" in resolution.discarded_realization_ids
    assert resolution.evidence["critical_groups_waived_from_failed_realizations"]


def test_failed_label_below_the_floor_is_not_usability_evidence():
    ledger = _conclusion_ledger(loser_label="failed", loser_confidence=0.70, complementary=True)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_COMPOSITE  # byte-for-byte pre-D-097 behaviour
    assert resolution.evidence["unusable_realization_ids"] == []


def test_a_usable_restatement_covering_more_critical_groups_still_wins_by_dominance():
    # D-063 CRITICAL_COVERAGE_DOMINANCE is untouched: a USABLE restatement
    # that covers strictly more CRITICAL groups than the delivery-score
    # leader remains the winner exactly as before D-097.
    ledger = _conclusion_ledger(loser_label="alternate", loser_confidence=0.75)
    resolution = resolve_realizations_shadow(ledger).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_WINNER
    assert resolution.winner_realization_id == "real_F"


def test_all_failed_labels_cancel_out_and_never_drop_the_idea():
    experience = _claim("c_exp", ("shared", "experience", "family"))
    a = _realization("real_A", claim_ids=("c_exp",), state="selected", label="failed", confidence=0.90)
    b = _realization("real_B", claim_ids=("c_exp",), state="discarded", label="failed", confidence=0.95, start=20.0)
    claims = [CanonicalClaimRecord(**{**experience.__dict__, "source_realization_ids": ("real_A", "real_B")})]
    resolution = resolve_realizations_shadow(_ledger((a, b), claims, scores={"real_A": 0.7, "real_B": 0.5})).idea_resolutions[IDEA]
    assert resolution.decision_status == RESOLVED_WINNER
    assert resolution.winner_realization_id == "real_A"
    assert resolution.evidence["unusable_realization_ids"] == []


def test_a_usable_realization_outranks_a_failed_one_with_a_higher_delivery_score():
    experience = _claim("c_exp", ("shared", "experience", "family"))
    failed_high = _realization("real_A", claim_ids=("c_exp",), state="selected", label="failed", confidence=0.90)
    usable_low = _realization("real_B", claim_ids=("c_exp",), state="discarded", label="alternate", confidence=0.70, start=20.0)
    claims = [CanonicalClaimRecord(**{**experience.__dict__, "source_realization_ids": ("real_A", "real_B")})]
    resolution = resolve_realizations_shadow(_ledger((failed_high, usable_low), claims, scores={"real_A": 0.80, "real_B": 0.55})).idea_resolutions[IDEA]
    assert resolution.winner_realization_id == "real_B"
