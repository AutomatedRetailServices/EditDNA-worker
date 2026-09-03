"""D-058 Phase 2 -- RESOLVER EVIDENCE HIERARCHY.

Root defect (docs/CUTSELL_DECISIONS.md D-057's gastritis forensic): among
candidates that already each satisfy full critical-claim coverage on their
own, `_resolve_one_idea`'s `_pick_winner` ranked by raw DeliveryScorer score
FIRST -- so a realization the Ledger's own event history already recorded
as the high-confidence semantic winner (`SEMANTIC_WINNER_OVERRIDE`,
`pipeline.py`'s `_semantic_best_take`) could still lose to a realization
with a higher watch/listen score but proven incomplete
(`complete_idea=False`). Fixed by an explicit evidence hierarchy: semantic
validity/completeness > high-confidence semantic winner evidence > critical
claim coverage quality > delivery quality > contextual richness, with an
explicit REVIEW_REQUIRED fallback when semantic evidence itself conflicts.

This file is entirely generic -- no Video00 clip ids or phrases -- and
reuses the exact fixture-helper shape `test_cutsell_d050c1_realization_
resolver.py` already established.
"""
from cutsell_worker.realization_resolver import (
    RESOLVED_WINNER,
    REVIEW_REQUIRED,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord,
    DELIVERY_SCORE_WINNER,
    RealizationRecord,
    SEMANTIC_WINNER_OVERRIDE,
    SemanticIdeaRecord,
    SemanticLedger,
)


# --- fixture helpers (same shape as test_cutsell_d050c1_realization_resolver.py) --

def _claim(canonical_claim_id, claim_type, tokens, importance="CRITICAL"):
    return CanonicalClaimRecord(
        canonical_claim_id=canonical_claim_id, claim_type=claim_type,
        content_tokens=frozenset(tokens), importance=importance,
        source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved",
    )


def _realization(realization_id, *, semantic_idea_id, claim_ids=(), state="selected", **overrides):
    fields = dict(
        realization_id=realization_id, semantic_idea_id=semantic_idea_id, retry_family_id=None,
        source_span_ids=(), attempt_id=None, clip_ids=(realization_id,), text="text",
        start=0.0, end=1.0, delivery_score=None, state=state, discard_reason=None,
        replacement_realization_id=None, claim_ids=tuple(claim_ids), render_fragment_ids=(),
    )
    fields.update(overrides)
    return RealizationRecord(**fields)


def _idea(idea_id, realization_ids):
    return SemanticIdeaRecord(
        semantic_idea_id=idea_id, retry_family_ids=(), realization_ids=tuple(realization_ids),
        canonical_claim_ids=(), current_winner_realization_id=None, composite_realization_ids=(),
        coverage_status="unresolved_ambiguous", story_order_position=None,
    )


def _ledger(realizations, claims, ideas):
    ledger = SemanticLedger()
    for r in realizations:
        ledger.register_realization(r)
    for c in claims:
        ledger.register_claim(c)
    for i in ideas:
        ledger.register_semantic_idea(i)
    return ledger


def _record_delivery_scores(ledger, idea_id, scores: dict):
    """Seed the Ledger's own DELIVERY_SCORE_WINNER evidence exactly the way
    `semantic_ledger.build_semantic_ledger_shadow` does from `take_judge_
    groups`, keyed by clip_id == realization_id (this file's fixture
    convention)."""
    ranked = [{"clip_id": rid, "score": score, "reason": "watch_listen_baseline"} for rid, score in scores.items()]
    top_rid = max(scores, key=scores.get)
    ledger.record_winner_decision(
        semantic_idea_id=idea_id, realization_id=top_rid, stage="take_judge_provider",
        decision_type=DELIVERY_SCORE_WINNER, reason="watch_listen_baseline_top_score",
        evidence={"ranked": ranked},
    )


def _record_semantic_override(ledger, idea_id, realization_id, *, confidence):
    ledger.record_winner_decision(
        semantic_idea_id=idea_id, realization_id=realization_id, stage="pipeline_semantic_best_take",
        decision_type=SEMANTIC_WINNER_OVERRIDE, reason="hybrid_semantic_label_override",
        evidence={"confidence": confidence},
    )


# --- 1. High semantic confidence complete take beats small delivery-score --
#        advantage (the exact D-057 gastritis shape) ------------------------

def test_high_semantic_confidence_complete_take_beats_small_delivery_advantage():
    claim = _claim("c1", "MEASUREMENT_QUANTITY", {"stomach", "endoscopy", "diagnosis"})
    incomplete_higher_score = _realization(
        "r_incomplete", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=False,
    )
    complete_lower_score = _realization(
        "r_complete", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True,
    )
    ledger = _ledger(
        [incomplete_higher_score, complete_lower_score], [claim],
        [_idea("idea_1", ["r_incomplete", "r_complete"])],
    )
    _record_delivery_scores(ledger, "idea_1", {"r_incomplete": 0.7292, "r_complete": 0.6663})
    _record_semantic_override(ledger, "idea_1", "r_complete", confidence=0.95)

    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == RESOLVED_WINNER
    assert res.winner_realization_id == "r_complete"


# --- 2. Delivery score wins when semantic evidence is equivalent -----------

def test_delivery_score_wins_when_semantic_evidence_equivalent():
    claim = _claim("c1", "MEASUREMENT_QUANTITY", {"stomach", "endoscopy", "diagnosis"})
    higher = _realization("r_higher", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True)
    lower = _realization("r_lower", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True)
    ledger = _ledger([higher, lower], [claim], [_idea("idea_1", ["r_higher", "r_lower"])])
    _record_delivery_scores(ledger, "idea_1", {"r_higher": 0.80, "r_lower": 0.60})
    # No semantic override recorded for either -- pure delivery-quality tiebreak.

    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == RESOLVED_WINNER
    assert res.winner_realization_id == "r_higher"


# --- 3. Incomplete take cannot beat a complete semantic winner even without -
#        a recorded delivery-score gap (tier 1 alone should decide) ---------

def test_incomplete_take_cannot_beat_complete_semantic_winner():
    claim = _claim("c1", "MEASUREMENT_QUANTITY", {"stomach", "endoscopy", "diagnosis"})
    incomplete = _realization("r_incomplete", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=False)
    complete = _realization("r_complete", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True)
    ledger = _ledger([incomplete, complete], [claim], [_idea("idea_1", ["r_incomplete", "r_complete"])])
    # Delivery score actually favors the incomplete take, same as the live shape.
    _record_delivery_scores(ledger, "idea_1", {"r_incomplete": 0.90, "r_complete": 0.10})

    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == RESOLVED_WINNER
    assert res.winner_realization_id == "r_complete"


# --- 4. Conflicting high-confidence semantic evidence -> REVIEW_REQUIRED ---

def test_conflicting_high_confidence_semantic_evidence_review_required():
    claim = _claim("c1", "MEASUREMENT_QUANTITY", {"stomach", "endoscopy", "diagnosis"})
    a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True)
    b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c1",), complete_idea=True)
    ledger = _ledger([a, b], [claim], [_idea("idea_1", ["r_a", "r_b"])])
    _record_delivery_scores(ledger, "idea_1", {"r_a": 0.70, "r_b": 0.65})
    # Two DIFFERENT realizations both confidently recorded as the semantic
    # winner (e.g. two different take_judge_groups' own overrides disagree)
    # -- never guess between them.
    _record_semantic_override(ledger, "idea_1", "r_a", confidence=0.9)
    _record_semantic_override(ledger, "idea_1", "r_b", confidence=0.95)

    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == REVIEW_REQUIRED
    assert res.decision_reason == "conflicting_high_confidence_semantic_winner_evidence"
    assert res.winner_realization_id is None


# --- 5. No high-confidence semantic evidence -> existing safe fallback -----
#        behavior (delivery quality, unchanged from pre-D-058) -------------

def test_no_high_confidence_semantic_evidence_falls_back_to_delivery_quality():
    claim = _claim("c1", "MEASUREMENT_QUANTITY", {"stomach", "endoscopy", "diagnosis"})
    a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c1",))
    b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c1",))
    ledger = _ledger([a, b], [claim], [_idea("idea_1", ["r_a", "r_b"])])
    _record_delivery_scores(ledger, "idea_1", {"r_a": 0.55, "r_b": 0.75})
    # A low-confidence override (below the 0.85 floor) must not count as
    # "high confidence" -- falls through to delivery quality exactly like
    # having no override recorded at all.
    _record_semantic_override(ledger, "idea_1", "r_a", confidence=0.50)

    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == RESOLVED_WINNER
    assert res.winner_realization_id == "r_b"
