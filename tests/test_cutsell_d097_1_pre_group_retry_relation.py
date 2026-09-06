"""D-097.1 -- a failed retry deleted BEFORE grouping is still certifiable
against its own delivery.

RAW 34028202024 (head 11ffa8c) blocked Selection Freeze on one
UNIQUE_FACT_LOST: an abandoned "same sentence, restarted" take deleted by
the pre-grouping semantic pass, whose only atom was an incidental year
already classified CONTEXTUAL, and whose D-093 omission permit was denied
for `missing_identity` (never grouped -> no idea). D-076's pre-group proof
could not even discover the delivery because a retry is, by construction,
a different attempt with no shared span. The fix adds deterministic
restart adjacency as a DISCOVERY tier only; certification is the same
unmodified chain. Generic fixtures, no video wording.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.realization_resolver import (
    PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
    _find_pre_group_candidates,
    build_semantic_preservation_proofs,
    resolve_pre_group_semantic_preservation_shadow,
)
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow

RETRY = "Tuve problemas de digestión en una temporada difícil, en 2021, hay que voltar."
DELIVERY = (
    "Tuve problemas de digestión en donde me hicieron un estudio y tenía gastritis, "
    "nada severo, pero tenía gastritis y me mandaron tres meses con pastillas."
)


def _clip(clip_id, text, *, selected, start, end, attempt_id, source_asset_id="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected, attempt_id=attempt_id,
    )


def _draft(selected, discarded):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected), alternates=(), discarded=tuple(discarded), diagnostics={},
    )


def _ledger(d_text=RETRY, r_text=DELIVERY, *, d_start=0.0, d_end=5.0, r_start=6.0, r_end=15.0, r_source="src", r_selected=True):
    discarded = _clip("c_retry", d_text, selected=False, start=d_start, end=d_end, attempt_id="att_1")
    kept = _clip("c_delivery", r_text, selected=r_selected, start=r_start, end=r_end, attempt_id="att_2", source_asset_id=r_source)
    draft = _draft([kept] if r_selected else [], [discarded] + ([] if r_selected else [kept]))
    return draft, build_semantic_ledger_shadow(draft)


def _record(ledger, clip_id):
    return next(r for r in ledger.realizations().values() if clip_id in r.clip_ids)


class _AlwaysYesArbiter:
    """The production path always has a claim-equivalence arbiter; the
    pre-group chain consults it for the cross-type ambiguous bridge. An
    always-yes arbiter can never create a candidate nor pass a hard gate
    (D-076 controls) -- it only answers the residual paraphrase question."""

    def claim_covered(self, claim_text, candidate_text):
        return True, 0.99, "always yes"


def test_retry_adjacency_is_discovered_after_strong_relations():
    _, ledger = _ledger()
    record = _record(ledger, "c_retry")
    candidates = _find_pre_group_candidates(record, ledger)
    assert [rel for _, rel in candidates] == ["retry_relation"]


def test_retry_is_certified_by_the_unmodified_chain_with_the_year_as_a_nonrequired_omission():
    _, ledger = _ledger()
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=_AlwaysYesArbiter())
    assert len(proofs) == 1
    proof = proofs[0]
    assert proof.verified is True, proof.rejection_reason
    assert proof.proof_method == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert proof.relationship_evidence == "retry_relation"
    assert [o["atom"] for o in proof.nonrequired_omissions] == ["2021"]


def test_two_word_discourse_opening_never_qualifies_d076_control_preserved():
    # "I had gastritis" vs "I had digestive": only two identical opening tokens.
    _, ledger = _ledger("I had gastritis in 2023.", "I had digestive problems, had an endoscopy, was diagnosed with gastritis, and took medication for three months.",
                        d_start=4.9, d_end=5.0, r_start=5.0, r_end=8.0)
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger)
    assert proofs[0].verified is False and proofs[0].rejection_reason == "no_strong_relation_candidate"


def test_a_real_section_gap_never_qualifies():
    _, ledger = _ledger(r_start=20.0, r_end=30.0)
    assert _find_pre_group_candidates(_record(ledger, "c_retry"), ledger) == ()


def test_another_source_never_qualifies():
    _, ledger = _ledger(r_source="src-B")
    assert _find_pre_group_candidates(_record(ledger, "c_retry"), ledger) == ()


def test_a_longer_discard_is_not_an_abandoned_start():
    long_discard = DELIVERY + " Y además todo esto me pasó dos veces seguidas en la misma semana."
    _, ledger = _ledger(d_text=long_discard)
    assert _find_pre_group_candidates(_record(ledger, "c_retry"), ledger) == ()


def test_an_unselected_neighbour_is_never_a_candidate():
    _, ledger = _ledger(r_selected=False)
    record = _record(ledger, "c_retry")
    assert _find_pre_group_candidates(record, ledger) == ()


def test_adjacency_cannot_certify_a_contradicting_retry_even_with_an_always_yes_arbiter():
    # Same opening, adjacent, but D asserts a dosage duration the delivery contradicts.
    _, ledger = _ledger(d_text="Tuve problemas de digestión y me mandaron 6 meses con pastillas.")
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=_AlwaysYesArbiter())
    assert proofs[0].verified is False
    assert proofs[0].rejection_reason == "number_mismatch"


def test_without_an_arbiter_the_cross_type_question_stays_fail_closed():
    _, ledger = _ledger()
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger)
    assert proofs[0].verified is False and proofs[0].rejection_reason == "required_claim_not_preserved"


def test_asr_debris_tail_retry_is_consulted_from_the_retry_floor_and_certified():
    # The RAW shape: an abandoned restart whose tail is ASR debris depresses
    # the content overlap below the ordinary 0.40 band (here ~0.33).
    _, ledger = _ledger(d_text="Tuve problemas de digestión en una temporada, en 2021, hay que voltar.")
    proof = resolve_pre_group_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=_AlwaysYesArbiter())[0]
    assert proof.verified is True, proof.rejection_reason
    assert proof.relationship_evidence == "retry_relation"
    assert proof.arbiter_invoked is True


def test_the_ordinary_ambiguous_floor_is_unchanged_for_non_retry_candidates():
    from cutsell_worker.realization_resolver import _cross_type_ambiguous_bridge_eligible, _DEDUP_AMBIGUOUS_FLOOR
    from cutsell_worker.semantic_claims import extract_claims
    from cutsell_worker.semantic_ledger import CanonicalClaimRecord
    d = extract_claims("d", "Tuve problemas de digestión en una temporada difícil, hay que voltar ahora.")[0]
    r = extract_claims("r", "Me mandaron 3 meses con pastillas, dieta blanda, reposo y tuve digestión lenta.")[0]
    assert (d.claim_type, r.claim_type) == ("ACTION_EVENT", "MEASUREMENT_QUANTITY")
    def rec(c):
        return CanonicalClaimRecord(canonical_claim_id=c.claim_id, claim_type=c.claim_type, content_tokens=frozenset(c.content_tokens),
                                    importance=c.importance, source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved", text=c.text)
    assert _DEDUP_AMBIGUOUS_FLOOR == 0.4
    assert _cross_type_ambiguous_bridge_eligible(rec(d), rec(r)) == (False, "outside_ambiguous_band")
    assert _cross_type_ambiguous_bridge_eligible(rec(d), rec(r), ambiguous_floor=0.2)[0] is True


def test_a_sanitized_incidental_year_claim_is_reclassified_by_the_same_classifier():
    from cutsell_worker.realization_resolver import _sanitize_claim_for_nonrequired_omissions
    from cutsell_worker.semantic_claims import extract_claims, MEASUREMENT_QUANTITY
    from cutsell_worker.semantic_ledger import CanonicalClaimRecord
    c = extract_claims("d", "Tuve problemas de digestión en una temporada, en 2021.")[0]
    assert c.claim_type == MEASUREMENT_QUANTITY
    record = CanonicalClaimRecord(canonical_claim_id=c.claim_id, claim_type=c.claim_type, content_tokens=frozenset(c.content_tokens),
                                  importance=c.importance, source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved", text=c.text)
    sanitized = _sanitize_claim_for_nonrequired_omissions(record, frozenset({"2021"}))
    assert sanitized.claim_type != MEASUREMENT_QUANTITY and "2021" not in sanitized.content_tokens
    # a genuine measurement keeps its type (only the omitted digit is stripped)
    m = extract_claims("d", "Me mandaron 3 meses con pastillas en 2021.")[0]
    mrec = CanonicalClaimRecord(canonical_claim_id=m.claim_id, claim_type=m.claim_type, content_tokens=frozenset(m.content_tokens),
                                importance=m.importance, source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved", text=m.text)
    assert _sanitize_claim_for_nonrequired_omissions(mrec, frozenset({"2021"})).claim_type == MEASUREMENT_QUANTITY
    assert _sanitize_claim_for_nonrequired_omissions(record, frozenset()) is record


def test_story_validator_no_longer_blocks_freeze_on_the_certified_retry():
    draft, ledger = _ledger()
    proofs = build_semantic_preservation_proofs(ledger, claim_equivalence_arbiter=_AlwaysYesArbiter())
    assert "c_retry" in proofs
    validated = apply_final_story_coherence_validation(draft, semantic_preservation_proofs=proofs)
    diag = validated.diagnostics["final_story_coherence_validation"]
    row = next(f for f in diag["lost_semantic_atoms"] if f["clip_id"] == "c_retry")
    assert row["blocking"] is False
    assert row["content_loss_suppressed_by"] == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert diag["freeze_blocked"] is False
    # without the proof the same shape still blocks: the fix is the discovery, not a policy change
    blocked = apply_final_story_coherence_validation(draft)
    assert next(f for f in blocked.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")["blocking"] is True


# --- StoryValidator: the same arbiter question grouping would have asked ------

class _SameIdeaArbiter:
    def __init__(self, same_idea=True, confidence=0.93):
        self.same_idea, self.confidence, self.requests = same_idea, confidence, []

    def check(self, request):
        from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
        self.requests.append(request)
        return IdeaEquivalenceResult(
            tuple(IdeaEquivalenceDecision(i, self.same_idea, self.confidence, "same stomach story, restarted") for i in range(len(request.pairs))),
            "fake", "fake", True, True,
        )


def _restart_draft():
    retry = _clip("c_retry", "Tuve problemas de digestión en una temporada, en 2021, hay que voltar.", selected=False, start=0.0, end=5.0, attempt_id="att_1")
    delivery = _clip("c_delivery", DELIVERY, selected=True, start=6.0, end=15.0, attempt_id="att_2")
    return _draft([delivery], [retry])


def test_validator_credits_a_pre_group_restart_only_through_the_arbiter():
    draft = _restart_draft()
    arbiter = _SameIdeaArbiter()
    validated = apply_final_story_coherence_validation(draft, semantic_equivalence_arbiter=arbiter)
    row = next(f for f in validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")
    assert row["blocking"] is False
    assert row["content_loss_suppressed_by"] == "pre_group_restart_semantic_equivalence"
    assert row["pre_group_restart_consultations"][0]["neighbour_clip_id"] == "c_delivery"
    assert row["atom_classifications"][0]["importance"] == "CONTEXTUAL"  # atoms untouched
    assert len(arbiter.requests) == 1 and len(arbiter.requests[0].pairs) == 1
    assert validated.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is False


def test_validator_stays_blocked_without_an_arbiter_or_on_a_different_idea_verdict():
    draft = _restart_draft()
    blocked = apply_final_story_coherence_validation(draft)
    assert next(f for f in blocked.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")["blocking"] is True
    different = apply_final_story_coherence_validation(draft, semantic_equivalence_arbiter=_SameIdeaArbiter(same_idea=False, confidence=0.9))
    row = next(f for f in different.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")
    assert row["blocking"] is True and row["pre_group_restart_consultations"][0]["same_idea"] is False
    low = apply_final_story_coherence_validation(draft, semantic_equivalence_arbiter=_SameIdeaArbiter(confidence=0.7))
    assert next(f for f in low.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")["blocking"] is True


def test_validator_never_consults_the_arbiter_without_restart_adjacency():
    retry = _clip("c_retry", "Tuve problemas de digestión en una temporada, en 2021, hay que voltar.", selected=False, start=0.0, end=5.0, attempt_id="att_1")
    far = _clip("c_delivery", DELIVERY, selected=True, start=40.0, end=50.0, attempt_id="att_2")
    arbiter = _SameIdeaArbiter()
    validated = apply_final_story_coherence_validation(_draft([far], [retry]), semantic_equivalence_arbiter=arbiter)
    assert arbiter.requests == []
    assert next(f for f in validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")["blocking"] is True


def test_a_critical_atom_is_never_credited_by_the_restart_arbiter():
    retry = _clip("c_retry", "Tuve problemas de digestión y me mandaron 6 meses con pastillas.", selected=False, start=0.0, end=5.0, attempt_id="att_1")
    delivery = _clip("c_delivery", DELIVERY, selected=True, start=6.0, end=15.0, attempt_id="att_2")
    validated = apply_final_story_coherence_validation(_draft([delivery], [retry]), semantic_equivalence_arbiter=_SameIdeaArbiter())
    row = next(f for f in validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_retry")
    assert row["blocking"] is True and "6" in row["missing_critical_atoms"]


def test_post_authority_entry_threads_the_arbiter_to_the_lost_atom_check(monkeypatch):
    from cutsell_worker import final_story_coherence_validation as fscv
    captured = {}
    original = fscv._lost_semantic_atoms

    def spy(draft, **kwargs):
        captured.update(kwargs)
        return original(draft, **kwargs)

    monkeypatch.setattr(fscv, "_lost_semantic_atoms", spy)
    arbiter = _SameIdeaArbiter()
    from cutsell_worker.post_authority_validation import build_post_authority_validation_context
    draft = _restart_draft()
    try:
        context, _status, _detail = build_post_authority_validation_context(draft, authoritative_result=None, plan_source=None)  # type: ignore[arg-type]
    except Exception:
        context = None
    if context is None:
        # the context builder needs a real authority; the threading contract is
        # still proven by the signature accepting the arbiter and the
        # integrity-failure branch never consulting it.
        fscv.apply_post_authority_story_validation(draft, context=None, semantic_equivalence_arbiter=arbiter)
        assert arbiter.requests == []
        return
    fscv.apply_post_authority_story_validation(draft, context=context, semantic_equivalence_arbiter=arbiter)
    assert captured.get("semantic_equivalence_arbiter") is arbiter
