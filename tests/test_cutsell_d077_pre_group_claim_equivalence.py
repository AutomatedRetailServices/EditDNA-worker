"""D-077: PRE_GROUP CLAIM-EQUIVALENCE COMPLETION -- closes the one D-076
disclosed limitation (a generic-symptom discard vs a richer, specific-
diagnosis candidate that shares no literal words, e.g. "stomach problems"
vs "digestive problems... diagnosed with gastritis") by letting the ONE
NEW, opt-in `cross_type_ambiguous_bridge` claim-matching strategy consult
the EXISTING D-061 claim-equivalence arbiter for a genuinely ambiguous
cross-TYPE paraphrase -- never a candidate-discovery mechanism (D-076's
own strong-relation requirement is completely untouched), never reachable
by PATH B (`allow_cross_type_ambiguous_bridge` defaults False everywhere
except the one pre-group call site).

Every fixture goes through the REAL pipeline: `build_semantic_ledger_
shadow` -> `resolve_pre_group_semantic_preservation_shadow`. No Video00-
specific text/ids.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import (
    PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
    REPLACEMENT_VERIFIED_SEMANTIC,
    REVIEW_REQUIRED,
    build_semantic_preservation_proofs,
    resolve_orphan_realizations_shadow,
    resolve_pre_group_semantic_preservation_shadow,
)
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def _clip(clip_id, text, *, selected, start=0.0, end=1.0, attempt_id=None, source_asset_id="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        attempt_id=attempt_id,
    )


def _draft(*, selected, discarded):
    """No `hybrid_editorial_chunks`/`draft_review_removed_ids` diagnostics
    at all -- both discards fall through to `clean_cut_or_composite_
    resolution`, D-076 Section 3's pre-group target population."""
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(selected,), alternates=(), discarded=(discarded,), diagnostics={},
    )


def _single_proof(d_text, r_text, **kwargs):
    discarded = _clip("c_orphan", d_text, selected=False, start=0.0, end=2.0,
                       attempt_id=kwargs.pop("d_attempt", "att_1"), source_asset_id=kwargs.pop("d_source", "src"))
    kept = _clip("c_kept", r_text, selected=True, start=5.0, end=8.0,
                 attempt_id=kwargs.pop("r_attempt", "att_1"), source_asset_id=kwargs.pop("r_source", "src"))
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = kwargs.pop("arbiter", None)
    call_kwargs = {"claim_equivalence_arbiter": arbiter} if arbiter is not None else {}
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger, **call_kwargs)
    assert len(proofs) == 1
    return proofs[0]


class _AlwaysYesArbiter:
    def __init__(self):
        self.calls = []

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return True, 0.99, "always yes"


class _CountingRealArbiter:
    """A real (non-lying) arbiter that still counts every call -- used for
    Section 12's own 'no arbiter call on deterministic pass/fail cases'
    requirement, and to prove the ambiguous case really does call it
    exactly once."""

    def __init__(self, verdict=True, confidence=0.9, reason="paraphrase confirmed"):
        self.calls = []
        self._verdict, self._confidence, self._reason = verdict, confidence, reason

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return self._verdict, self._confidence, self._reason


# --- Section 6: D-074 generic synonym shape ---------------------------------

_D074_SYNONYM_D = "I had stomach problems for a while."
_D074_SYNONYM_R = "I had digestive problems, had an endoscopy, and was diagnosed with gastritis."


def test_d074_synonym_shape_verifies_with_arbiter():
    arbiter = _CountingRealArbiter(verdict=True)
    proof = _single_proof(_D074_SYNONYM_D, _D074_SYNONYM_R, arbiter=arbiter)
    assert proof.verified is True
    assert proof.proof_method == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert proof.arbiter_invoked is True
    assert len(arbiter.calls) == 1
    assert arbiter.calls[0] == (_D074_SYNONYM_D, _D074_SYNONYM_R)


def test_d074_synonym_shape_fails_closed_without_arbiter():
    """Section 1: 'If the existing arbiter is unavailable: fail closed.'"""
    proof = _single_proof(_D074_SYNONYM_D, _D074_SYNONYM_R)
    assert proof.verified is False
    assert proof.rejection_reason == "required_claim_not_preserved"


def test_d074_synonym_shape_arbiter_declining_never_verifies():
    """A real, honest arbiter that (correctly or not) declines must never
    be overridden -- this is a bounded confirmation question, not a second
    authority."""
    arbiter = _CountingRealArbiter(verdict=False, reason="different proposition")
    proof = _single_proof(_D074_SYNONYM_D, _D074_SYNONYM_R, arbiter=arbiter)
    assert proof.verified is False
    assert len(arbiter.calls) == 1


# --- Section 7: same text without relation ----------------------------------

def test_same_synonym_text_without_relation_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        _D074_SYNONYM_D, _D074_SYNONYM_R, d_attempt="att_1", r_attempt="att_2", arbiter=arbiter,
    )
    assert proof.verified is False
    assert proof.rejection_reason == "no_strong_relation_candidate"
    assert arbiter.calls == []


# --- Section 8: same-topic different-event ----------------------------------

def test_same_topic_different_event_never_verifies_even_with_relation_and_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "Doctor ordered an ultrasound.", "The ultrasound found a suspicious nodule.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


# --- Section 9: hard-mismatch matrix (always-YES arbiter must never win) ---

def test_number_mismatch_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof("The nodule measured 5 cm.", "The nodule measured 3 cm.", arbiter=arbiter)
    assert proof.verified is False
    assert arbiter.calls == []


def test_negation_flip_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof("I did not have gastritis.", "I had gastritis.", arbiter=arbiter)
    assert proof.verified is False
    assert arbiter.calls == []


def test_diagnosis_substitution_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof("The biopsy was benign.", "I was diagnosed with papillary cancer.", arbiter=arbiter)
    assert proof.verified is False
    assert arbiter.calls == []


def test_attribution_asymmetry_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "I did not like it.", "Customers said they did not like it.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


def test_causal_reversal_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "The medication caused the breakout.", "The medication did not cause the breakout.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


def test_temporal_reversal_never_verifies_even_with_always_yes():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "It happened before treatment started.", "It happened after treatment started.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


# --- Section 4/9: existing D-073 diagnosis-substitution fixture must stay
# protected through this NEW pre-group path too (not just PATH B) ----------

def test_existing_d073_diagnosis_fixture_never_verifies_via_pre_group_either():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "The test confirmed it was gastritis.", "The test confirmed it was an ulcer.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


# --- Section 13: sales/UGC generalization -----------------------------------

_SALES_SYNONYM_D = "I had annoying bloating for weeks."
_SALES_SYNONYM_R = "I had digestive bloating, and the doctor confirmed it was a food sensitivity."


def test_sales_positive_ambiguous_paraphrase_may_reach_arbiter_and_verify():
    arbiter = _CountingRealArbiter(verdict=True)
    proof = _single_proof(_SALES_SYNONYM_D, _SALES_SYNONYM_R, arbiter=arbiter)
    assert proof.verified is True
    assert len(arbiter.calls) == 1


def test_sales_negative_dosage_never_reaches_arbiter():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof("I take two gummies every morning.", "They helped my digestion.", arbiter=arbiter)
    assert proof.verified is False
    assert arbiter.calls == []


# --- Section 12: cost/observability -----------------------------------------

def test_deterministic_pass_never_calls_arbiter():
    """A literal-subset D-074-shaped pass (D-076's own mechanism) must
    resolve without ever reaching the D-077 arbiter."""
    arbiter = _CountingRealArbiter(verdict=True)
    proof = _single_proof(
        "I had gastritis in 2023.",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        arbiter=arbiter,
    )
    assert proof.verified is True
    assert proof.arbiter_invoked is False
    assert arbiter.calls == []
    assert proof.arbiter_consultations == ()


def test_deterministic_fail_never_calls_arbiter():
    arbiter = _CountingRealArbiter(verdict=True)
    proof = _single_proof("The nodule measured 5 cm.", "The nodule measured 3 cm.", arbiter=arbiter)
    assert proof.verified is False
    assert arbiter.calls == []
    assert proof.arbiter_consultations == ()


def test_ambiguous_case_records_full_consultation_diagnostics():
    arbiter = _CountingRealArbiter(verdict=True, confidence=0.87, reason="same core proposition")
    proof = _single_proof(_D074_SYNONYM_D, _D074_SYNONYM_R, arbiter=arbiter)
    assert proof.verified is True
    assert len(proof.arbiter_consultations) == 1
    consultation = proof.arbiter_consultations[0]
    assert consultation["method"] == "cross_type_ambiguous_bridge"
    assert consultation["deterministic_result"] == "ambiguous"
    assert consultation["arbiter_eligible"] is True
    assert consultation["arbiter_invoked"] is True
    assert consultation["verdict"] is True
    assert consultation["confidence"] == 0.87
    assert consultation["reason"] == "same core proposition"
    assert "provider" in consultation
    assert "model" in consultation


# --- Section 4: strong relation still mandatory, even for the new bridge --

def test_source_span_relation_alone_sufficient_for_cross_type_bridge():
    arbiter = _CountingRealArbiter(verdict=True)
    discarded = DraftClip(
        clip_id="c_orphan", source_asset_id="src", source_order=0,
        start=0.0, end=2.0, text=_D074_SYNONYM_D, caption_text=_D074_SYNONYM_D, selected=False,
        source_span_id="span_1",
    )
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0,
        start=5.0, end=8.0, text=_D074_SYNONYM_R, caption_text=_D074_SYNONYM_R, selected=True,
        source_span_id="span_1",
    )
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=arbiter)
    assert len(proofs) == 1
    assert proofs[0].verified is True
    assert proofs[0].relationship_evidence == "source_span_relation"


# --- Section 11: StoryValidator remains consumer only -----------------------

def test_storyvalidator_suppresses_synonym_case_via_existing_wiring():
    """No new StoryValidator code path -- the synonym-shaped proof flows
    through the EXACT SAME `semantic_preservation_proofs` lookup D-076
    already wired. Uses a richer D/R pair than the minimal D-074 fixture
    above so `_lost_semantic_atoms`'s own content-loss floor
    (`own_content_token_count >= 5`) is actually cleared and a finding row
    is produced to suppress in the first place."""
    d_text = "I had ongoing stomach ache, mornings of feeling tired with problems and an endoscopy."
    r_text = "I had digestive problems, had an endoscopy, and was diagnosed with gastritis."
    discarded = _clip("c_orphan", d_text, selected=False, start=0.0, end=2.0, attempt_id="att_1")
    kept = _clip("c_kept", r_text, selected=True, start=5.0, end=8.0, attempt_id="att_1")
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = _CountingRealArbiter(verdict=True)
    proofs = build_semantic_preservation_proofs(ledger, claim_equivalence_arbiter=arbiter)
    assert "c_orphan" in proofs

    validated = apply_final_story_coherence_validation(draft, semantic_preservation_proofs=proofs)
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
    row = next(f for f in findings if f["clip_id"] == "c_orphan")
    assert row["blocking"] is False
    assert row["content_loss_suppressed_by"] == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert row["preserving_realization_id"] == "c_kept"


# --- PATH B stays byte-identical (never gains the new bridge) ---------------

def _orphan_decision(clip_id, *, pre_guard_candidate=None):
    return {
        "clip_id": clip_id, "applied_delete": True,
        "delete_basis": "high_confidence_semantic",
        "later_retry_replacement_id": None,
        "replacement_candidate_clip_id_before_guard": pre_guard_candidate,
    }


def test_path_b_never_gains_cross_type_bridge():
    """The EXACT same cross-type ambiguous synonym pair, routed through
    PATH B's own `hybrid_editorial_chunks` discard population instead of
    the pre-group population, must NEVER verify -- even with an always-YES
    arbiter -- proving `allow_cross_type_ambiguous_bridge` truly defaults
    False on PATH B's own call site and D-077 changed nothing there."""
    discarded = DraftClip(
        clip_id="c_orphan", source_asset_id="src", source_order=0,
        start=0.0, end=2.0, text=_D074_SYNONYM_D, caption_text=_D074_SYNONYM_D, selected=False,
    )
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0,
        start=5.0, end=8.0, text=_D074_SYNONYM_R, caption_text=_D074_SYNONYM_R, selected=True,
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(discarded,),
        diagnostics={"hybrid_editorial_chunks": [{"decisions": [_orphan_decision("c_orphan", pre_guard_candidate="c_kept")]}]},
    )
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = _AlwaysYesArbiter()
    reviews = resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter)
    review = next(r for r in reviews if r.realization_id == "c_orphan")
    assert review.verdict == REVIEW_REQUIRED
    assert review.verdict != REPLACEMENT_VERIFIED_SEMANTIC
    assert arbiter.calls == []
