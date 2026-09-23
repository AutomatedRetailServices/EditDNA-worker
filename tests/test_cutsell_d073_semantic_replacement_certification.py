"""D-073: PATH B semantic replacement certification, inside the existing
Unified Resolver orphan-resolution authority (realization_resolver.py).

Every fixture goes through the REAL pipeline: `build_semantic_ledger_shadow`
(real `extract_claims`, real Ledger reconstruction) -> `resolve_orphan_
realizations_shadow` (PATH A first, unconditionally, then PATH B only when
PATH A found nothing). No Video00-specific text/ids -- generic fixtures
matching D-067-D-071's own shapes, per D-073 Section 13's own instruction.

PATH A (lexical) is never touched by this file -- its own tests
(test_cutsell_complete_retry_identity_guard.py) stay green, unchanged,
verified again here in Section 14.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_claims import extract_claims
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import (
    PRE_GROUP_REJECTED,
    REPLACEMENT_VERIFIED_SAFE,
    REPLACEMENT_VERIFIED_SEMANTIC,
    REVIEW_REQUIRED,
    VERIFICATION_METHOD_LEXICAL,
    VERIFICATION_METHOD_SEMANTIC,
    resolve_orphan_realizations_shadow,
)


def _clip(clip_id, text, *, selected, start=0.0, end=1.0):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _draft(*, selected=(), discarded=(), decisions):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"hybrid_editorial_chunks": [{"decisions": list(decisions)}]},
    )


def _orphan_decision(clip_id, *, pre_guard_candidate=None):
    return {
        "clip_id": clip_id, "applied_delete": True,
        "delete_basis": "high_confidence_semantic",
        "later_retry_replacement_id": None,
        "replacement_candidate_clip_id_before_guard": pre_guard_candidate,
    }


def _review_for(reviews, realization_id):
    return next(r for r in reviews if r.realization_id == realization_id)


class _AlwaysConfirmArbiter:
    def __init__(self):
        self.calls = []

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return True, 0.99, "always confirms"


class _AlwaysDeclineArbiter:
    def claim_covered(self, claim_text, candidate_text):
        return False, 0.4, "never confirms"


# --- Section 12: directional test matrix ------------------------------------

def test_superset_replacement_may_verify():
    """discard: 'ordered ultrasound' -> replacement: 'ordered ultrasound and
    found nodule' -- MAY VERIFY."""
    discarded = _clip("c_orphan", "She ordered a thyroid ultrasound.", selected=False, start=0.0, end=2.0)
    kept = _clip(
        "c_kept", "She ordered a thyroid ultrasound and it found a suspicious 3 cm nodule.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC
    assert review.verification_method == VERIFICATION_METHOD_SEMANTIC
    assert review.replacement_realization_id == kept.clip_id
    assert review.semantic_replacement_evidence["same_idea_verified"] is True
    assert review.semantic_replacement_evidence["unique_required_content_preserved"] is True


def test_reverse_direction_never_verifies():
    """discard includes the nodule fact; replacement drops it -- NEVER
    VERIFY, regardless of how the reverse pairing behaves."""
    discarded = _clip(
        "c_orphan", "She ordered a thyroid ultrasound and it found a suspicious 3 cm nodule.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _clip("c_kept", "She ordered a thyroid ultrasound.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED
    # D's realization text carries a number ("3 cm") absent from R's --
    # the realization-level NUMBER hard gate fails this closed before the
    # per-claim preservation loop even runs (a strictly earlier, more
    # precise diagnosis of the same required NEVER-VERIFY outcome).
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "number_mismatch"


def test_same_topic_different_event_never_verifies_founding_guard_case():
    """The complete_retry_identity_guard's own founding failure class,
    re-verified at THIS layer: a same-topic narrative continuation
    (new fact/event) must never be mistaken for a preserving replacement."""
    discarded = _clip("c_orphan", "She ordered a thyroid ultrasound.", selected=False, start=0.0, end=2.0)
    kept = _clip(
        "c_kept", "The ultrasound found a suspicious 3 cm nodule and it was sent for biopsy.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_same_topic_missing_critical_fact_never_verifies():
    discarded = _clip(
        "c_orphan", "The nodule measured 3 centimeters and was sent for biopsy.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _clip("c_kept", "The nodule was sent for biopsy.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_different_number_never_verifies():
    discarded = _clip(
        "c_orphan", "The nodule measured 3 centimeters and was sent for biopsy.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _clip(
        "c_kept", "The nodule measured 5 centimeters and was sent for biopsy.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_additional_number_in_replacement_is_safe_superset_may_verify():
    """R preserving D's own number AND adding a genuinely additional one
    (a different fact, not a substitution) is the safe superset direction
    Section 3 explicitly allows -- must NOT be rejected as a number
    mismatch (subset check, not exact-set equality)."""
    discarded = _clip(
        "c_orphan", "The nodule measured 3 centimeters and was sent for biopsy.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _clip(
        "c_kept",
        "The nodule measured 3 centimeters and was sent for biopsy, and the "
        "results came back after 5 days.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC
    assert review.semantic_replacement_evidence["hard_gate_results"]["realization_number_match"] is True


def test_different_diagnosis_never_verifies_even_with_confirming_arbiter():
    """The D-061-shaped adversarial case: even an arbiter that would
    confirm ANYTHING must never certify a diagnosis substitution."""
    discarded = _clip("c_orphan", "The test confirmed it was gastritis.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "The test confirmed it was an ulcer.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = _AlwaysConfirmArbiter()
    review = _review_for(
        resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter), discarded.clip_id,
    )
    assert review.verdict == REVIEW_REQUIRED


def test_factual_negation_reversal_never_verifies():
    discarded = _clip("c_orphan", "It did not reduce my bloating.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "It reduced my bloating.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_causal_reversal_never_verifies_even_with_confirming_arbiter():
    discarded = _clip("c_orphan", "The medication caused the rash.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "The rash caused the need for medication.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = _AlwaysConfirmArbiter()
    review = _review_for(
        resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter), discarded.clip_id,
    )
    assert review.verdict == REVIEW_REQUIRED
    # Never even asked -- direction-sensitive claims bypass the arbiter path.
    assert arbiter.calls == []


def test_temporal_reversal_never_verifies_even_with_confirming_arbiter():
    discarded = _clip("c_orphan", "First the swelling appeared, then the fever started.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "First the fever started, then the swelling appeared.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    arbiter = _AlwaysConfirmArbiter()
    review = _review_for(
        resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter), discarded.clip_id,
    )
    assert review.verdict == REVIEW_REQUIRED


def test_selected_candidate_incomplete_never_verifies():
    discarded = _clip("c_orphan", "She ordered a thyroid ultrasound.", selected=False, start=0.0, end=2.0)
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0, start=5.0, end=8.0,
        text="She ordered a thyroid ultrasound and found a nodule.",
        caption_text="She ordered a thyroid ultrasound and found a nodule.",
        selected=True, complete_idea=False,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "candidate_incomplete"


# --- Section 13: orphan A/B/C regression, generic shapes --------------------

def test_orphan_a_like_claim_subsumption_certifies_when_selected_and_preserving():
    """A-like: claim-level full subsumption, lexical sequence guard would
    have failed (not exercised here -- this file tests PATH B directly),
    same relation verified via the pre-guard candidate, complete and
    selected -- REPLACEMENT_VERIFIED_SEMANTIC."""
    discarded = _clip(
        "c_orphan", "At the end of my contract, I asked my doctor for tests.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _clip(
        "c_kept",
        "At the end of my contract, I asked my doctor for tests she could think of, "
        "including anything she could recommend.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC


def test_orphan_b_like_no_concrete_candidate_stays_review_required():
    """B-like: a failed/incomplete fragment with no valid concrete
    selected replacement under PATH B evidence (no pre-guard candidate at
    all, matching D-070's own finding for this shape) -- REVIEW_REQUIRED,
    unless a concrete preserving replacement is proven (it isn't here)."""
    discarded = _clip("c_orphan", "That's when they sent me", selected=False, start=0.0, end=2.0)
    kept = _clip(
        "c_kept", "That's when they sent me to get thyroid scans and others.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate=None)],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "no_pre_guard_candidate"


def test_orphan_c_like_truncated_text_stays_review_required():
    """C-like: truncated discarded text with essentially no extractable
    claim content, and/or unresolved semantic equivalence against the
    kept candidate -- REVIEW_REQUIRED."""
    discarded = _clip("c_orphan", "I had stomach problems...", selected=False, start=0.0, end=2.0)
    kept = _clip(
        "c_kept",
        "I had digestion problems where they did an endoscopy and said it was gastritis.",
        selected=True, start=5.0, end=8.0,
    )
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


# --- Section 14: historical guard regression --------------------------------

def test_path_a_still_wins_when_it_already_verified_a_replacement():
    """PATH B is never even attempted when PATH A (lexical) already
    verified a replacement -- verification_method stays 'lexical'."""
    discarded = _clip("c_orphan", "Original realization text now gone.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "Replacement realization text.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[{
            "clip_id": "c_orphan", "applied_delete": True, "delete_basis": "semantic_failed",
            "later_retry_replacement_id": "c_kept",
            "replacement_candidate_clip_id_before_guard": "c_kept",
        }],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REPLACEMENT_VERIFIED_SAFE
    assert review.verification_method == VERIFICATION_METHOD_LEXICAL


def test_pre_group_rejected_never_reaches_path_b():
    discarded = _clip("c_orphan", "Some ordinary rejected clip.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "Some ordinary rejected clip, restated more fully.", selected=True, start=5.0, end=8.0)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(discarded,), diagnostics={},
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == PRE_GROUP_REJECTED
    assert review.verification_method == ""
    assert review.semantic_replacement_evidence == {}


def test_arbiter_failure_fails_closed_never_certifies():
    """Provider failure/exception during arbiter consultation must fail
    closed -- REVIEW_REQUIRED, never a partial certification."""
    discarded = _clip("c_orphan", "Symptoms that did not seem suspicious, looking back now they were.",
                       selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "Symptoms I had, which in my view were symptomatic.",
                 selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)

    class _RaisingArbiter:
        def claim_covered(self, claim_text, candidate_text):
            raise RuntimeError("provider timeout")

    review = _review_for(
        resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=_RaisingArbiter()),
        discarded.clip_id,
    )
    assert review.verdict != REPLACEMENT_VERIFIED_SEMANTIC


def test_no_arbiter_declines_still_stays_review_required_for_ambiguous_cases():
    discarded = _clip("c_orphan", "Symptoms that did not seem suspicious, looking back now they were.",
                       selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "Symptoms I had, which in my view were symptomatic.",
                 selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(
        resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=_AlwaysDeclineArbiter()),
        discarded.clip_id,
    )
    assert review.verdict != REPLACEMENT_VERIFIED_SEMANTIC


def test_candidate_realization_not_found_fails_closed():
    discarded = _clip("c_orphan", "She ordered a thyroid ultrasound.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "She ordered a thyroid ultrasound and found a nodule.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="clip_does_not_exist")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "candidate_realization_not_found"


def test_candidate_not_selected_fails_closed():
    discarded = _clip("c_orphan", "She ordered a thyroid ultrasound.", selected=False, start=0.0, end=2.0)
    alt = _clip("c_alt", "She ordered a thyroid ultrasound and found a nodule.", selected=False, start=5.0, end=8.0)
    draft = _draft(
        selected=(), discarded=(discarded, alt),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_alt")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "candidate_not_selected"


def test_no_extractable_claims_never_verifies():
    discarded = _clip("c_orphan", "uh...", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "She ordered a thyroid ultrasound and found a nodule.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_contradiction_between_orphan_and_candidate_never_verifies():
    discarded = _clip("c_orphan", "It was not 5 percent.", selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "It was 10 percent.", selected=True, start=5.0, end=8.0)
    draft = _draft(
        selected=(kept,), discarded=(discarded,),
        decisions=[_orphan_decision("c_orphan", pre_guard_candidate="c_kept")],
    )
    ledger = build_semantic_ledger_shadow(draft)
    review = _review_for(resolve_orphan_realizations_shadow(ledger), discarded.clip_id)
    assert review.verdict == REVIEW_REQUIRED


def test_founding_guard_tests_unaffected_by_path_b():
    """Direct proof that PATH B changes nothing about complete_retry_
    identity_guard's own decisions -- imported and re-run here for
    visibility in this file's own suite."""
    from cutsell_worker.contracts import CandidateTake
    from cutsell_worker.hybrid_session_cleanup import _later_semantic_retry_replacement

    def take(clip_id, start, end, text, *, complete_idea=True):
        return CandidateTake(
            clip_id=clip_id, source_asset_id="src", source_order=0,
            start=start, end=end, text=text, complete_idea=complete_idea,
        )

    failed = take("failed", 10.0, 14.0, "a hacer sonografia de tiroides y otras sonografias", complete_idea=True)
    continuation = take(
        "continuation", 18.0, 24.0,
        "en la sonografia de tiroides aparecio un nodulo sospechoso de 3 centimetros que se mando a biopsia",
        complete_idea=True,
    )
    decisions = {"failed": ("failed", 0.85), "continuation": ("winner", 0.95)}
    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, continuation), decisions)
    assert overlap == 0.0
    assert replacement is None
