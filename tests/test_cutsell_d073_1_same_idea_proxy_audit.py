"""D-073.1: SAME-IDEA PROXY SAFETY AUDIT.

Offline-only audit of D-073's own `candidate.state == "selected"` proxy for
"verified same semantic idea/retry relation" (the only available signal for
a TRUE orphan, which has no `semantic_idea_id` by definition). Every
fixture goes through the REAL pipeline exactly like
`test_cutsell_d073_semantic_replacement_certification.py`.

This audit found and fixed ONE concrete, provable unsafe case: R's own
matching claim carrying REPORTED/ATTRIBUTED speech ("Some customers said it
did not work for them...") could satisfy preservation for a D claim
asserting the identical words directly and unattributed ("It did not work
for me.") -- even though R's own net assertion, in a contrastive-rebuttal
frame, is the OPPOSITE of D's. Fixed via
`_preservation_blocked_by_attribution_asymmetry` in realization_resolver.py
(see that function's own docstring). This file locks the fix in and proves
every other adversarial construction attempted during the audit does NOT
defeat the selected-state proxy once all of PATH B's remaining preservation
gates are applied.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import (
    REPLACEMENT_VERIFIED_SEMANTIC,
    REVIEW_REQUIRED,
    resolve_orphan_realizations_shadow,
)


def _clip(clip_id, text, *, selected, start=0.0, end=1.0):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _draft(*, selected, discarded, pre_guard_candidate):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(selected,), alternates=(), discarded=(discarded,),
        diagnostics={"hybrid_editorial_chunks": [{"decisions": [{
            "clip_id": discarded.clip_id, "applied_delete": True,
            "delete_basis": "high_confidence_semantic",
            "later_retry_replacement_id": None,
            "replacement_candidate_clip_id_before_guard": pre_guard_candidate,
        }]}]},
    )


def _resolve(discard_text, kept_text, *, arbiter=None):
    discarded = _clip("c_orphan", discard_text, selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", kept_text, selected=True, start=5.0, end=8.0)
    draft = _draft(selected=kept, discarded=discarded, pre_guard_candidate="c_kept")
    ledger = build_semantic_ledger_shadow(draft)
    kwargs = {"claim_equivalence_arbiter": arbiter} if arbiter is not None else {}
    return next(
        r for r in resolve_orphan_realizations_shadow(ledger, **kwargs) if r.realization_id == "c_orphan"
    )


class _AlwaysYesArbiter:
    def __init__(self):
        self.calls = []

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return True, 0.99, "always yes"


# --- Section 3: founding historical case (exact audit text) -----------------

def test_founding_historical_case_exact_audit_text_never_verifies():
    review = _resolve(
        "She was sent for a thyroid ultrasound.",
        "The thyroid ultrasound found a suspicious 3 cm nodule and it was sent for biopsy.",
    )
    assert review.verdict == REVIEW_REQUIRED
    assert review.semantic_replacement_evidence["semantic_replacement_reason"] == "required_claim_not_preserved"


# --- Section 4: same-topic-different-event matrix ---------------------------

def test_same_topic_matrix_a_test_ordered_vs_result_shown():
    review = _resolve("She ordered the test.", "The test showed cancer.")
    assert review.verdict == REVIEW_REQUIRED


def test_same_topic_matrix_b_started_product_vs_felt_better():
    review = _resolve("I started taking the product.", "After a week I felt better.")
    assert review.verdict == REVIEW_REQUIRED


def test_same_topic_matrix_c_showed_pouch_vs_gummies_tasted():
    review = _resolve("I showed the pouch.", "The gummies tasted good.")
    assert review.verdict == REVIEW_REQUIRED


def test_same_topic_matrix_d_doctor_ordered_vs_ultrasound_found():
    review = _resolve("The doctor ordered an ultrasound.", "The ultrasound found a nodule.")
    assert review.verdict == REVIEW_REQUIRED


# --- Section 5: valid superset matrix (must still certify) ------------------

def test_valid_superset_ultrasound_and_later_nodule_found():
    review = _resolve(
        "She ordered an ultrasound.",
        "She ordered an ultrasound and later they found a nodule.",
    )
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC


def test_valid_superset_gummies_routine():
    review = _resolve(
        "I take two gummies every morning.",
        "I take two gummies every morning and they helped my routine.",
    )
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC


# --- Section 6: always-YES arbiter attack ------------------------------------

def test_always_yes_arbiter_cannot_certify_same_topic_different_event():
    arbiter = _AlwaysYesArbiter()
    review = _resolve("She ordered the test.", "The test showed cancer.", arbiter=arbiter)
    assert review.verdict == REVIEW_REQUIRED
    # Deterministic content-token/type mismatch rejects it before the
    # arbiter branch (D-066 hindsight path only) is ever reachable.
    assert arbiter.calls == []


def test_always_yes_arbiter_cannot_certify_founding_case():
    arbiter = _AlwaysYesArbiter()
    review = _resolve(
        "She was sent for a thyroid ultrasound.",
        "The thyroid ultrasound found a suspicious 3 cm nodule and it was sent for biopsy.",
        arbiter=arbiter,
    )
    assert review.verdict == REVIEW_REQUIRED
    assert arbiter.calls == []


# --- D-073.1 discovered defect: reported/attributed speech asymmetry --------

def test_reported_attribution_in_replacement_never_verifies_direct_claim():
    """D-073.1's own discovered adversarial case: R's matching claim
    attributes D's exact words to a third party as the SETUP of a
    contrastive rebuttal ("...but it worked great for me") -- R's own net
    assertion is the OPPOSITE of D's direct claim. Must NEVER certify,
    even though every deterministic content/negation/digit gate would
    otherwise pass."""
    review = _resolve(
        "It did not work for me.",
        "Some customers said it did not work for them but it worked great for me.",
    )
    assert review.verdict == REVIEW_REQUIRED


def test_reported_attribution_never_verifies_even_with_always_yes_arbiter():
    arbiter = _AlwaysYesArbiter()
    review = _resolve(
        "It did not work for me.",
        "Some customers said it did not work for them but it worked great for me.",
        arbiter=arbiter,
    )
    assert review.verdict == REVIEW_REQUIRED


def test_symmetric_attribution_may_still_certify():
    """When D's OWN claim already carries the same attribution framing
    (both quote the same source), the asymmetry guard must not fire --
    this is an ordinary safe superset, not the adversarial case above."""
    review = _resolve(
        "The doctor said I have gastritis.",
        "The doctor said I have gastritis and prescribed medication.",
    )
    assert review.verdict == REPLACEMENT_VERIFIED_SEMANTIC
