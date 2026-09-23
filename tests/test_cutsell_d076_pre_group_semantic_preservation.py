"""D-076: PRE_GROUP_SEMANTIC_PRESERVATION -- implements D-075's accepted
design. Extends the Unified Resolver's own evaluation to two discard
populations previously invisible to any semantic authority
(`clean_cut_or_composite_resolution`, `draft_review_removed`), and wires a
new, unifying SEMANTIC_PRESERVATION_PROOF into StoryValidator's consumption
alongside the existing PATH A/PATH B verdicts. PATH A/PATH B's own gates
and StoryValidator's own GROUPED_SAME_IDEA mechanism are reused verbatim,
never modified -- see realization_resolver.py's own docstrings for exactly
which functions are shared and which are new.

Every fixture goes through the REAL pipeline: `build_semantic_ledger_
shadow` (real `extract_claims`, real Ledger reconstruction, real
`DraftClip.attempt_id`/`.source_asset_id` provenance) -> `resolve_pre_
group_semantic_preservation_shadow`. No Video00-specific text/ids.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import (
    PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
    build_semantic_preservation_proofs,
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
    resolution`, exactly D-076 Section 3's target population."""
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


# --- Section 8: D-074 generic regression (nonrequired-omission year) -------

def test_d074_generic_case_verifies_with_nonrequired_year_omission():
    """A generic stand-in for D-074's own real finding: the required
    proposition is a strict literal subset of the selected realization's
    own words; the only D-side content absent from R is a bare incidental
    year, already classified CONTEXTUAL by the existing deterministic
    classifier -- recorded as a nonrequired omission, not a blocker."""
    proof = _single_proof(
        "I had gastritis in 2023.",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
    )
    assert proof.verified is True
    assert proof.proof_method == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert proof.relationship_evidence == "attempt_relation"
    assert len(proof.nonrequired_omissions) == 1
    assert proof.nonrequired_omissions[0]["atom"] == "2023"
    assert proof.nonrequired_omissions[0]["importance"] == "CONTEXTUAL"


def test_valid_superset_no_omission_still_verifies():
    proof = _single_proof(
        "She ordered an ultrasound.",
        "She ordered an ultrasound and later they found a nodule.",
    )
    assert proof.verified is True
    assert proof.nonrequired_omissions == ()


# --- Section 9: no-relation negative control (mandatory) --------------------

def test_no_relation_different_attempt_same_source_never_verifies():
    proof = _single_proof(
        "I had gastritis in 2023.",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        d_attempt="att_1", r_attempt="att_2",
    )
    assert proof.verified is False
    assert proof.rejection_reason == "no_strong_relation_candidate"


def test_no_relation_different_source_even_with_same_attempt_id_never_verifies():
    """Same source_asset_id is an ADDITIONAL requirement -- a coincidental
    attempt_id match across two different sources must never qualify."""
    proof = _single_proof(
        "I had gastritis in 2023.",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        d_attempt="att_1", r_attempt="att_1", d_source="srcA", r_source="srcB",
    )
    assert proof.verified is False
    assert proof.rejection_reason == "no_strong_relation_candidate"


def test_no_relation_topically_similar_and_temporally_close_never_verifies():
    """Same source, close start/end times, topically related -- but with
    NO attempt/source-span relation supplied -- must still produce NO
    proof. This is the mandatory 'temporal proximity alone is never
    sufficient' control."""
    discarded = _clip("c_orphan", "I had gastritis in 2023.", selected=False, start=4.9, end=5.0, source_asset_id="src")
    kept = _clip(
        "c_kept",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        selected=True, start=5.0, end=8.0, source_asset_id="src",
    )
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    proofs = resolve_pre_group_semantic_preservation_shadow(ledger)
    assert len(proofs) == 1
    assert proofs[0].verified is False
    assert proofs[0].rejection_reason == "no_strong_relation_candidate"


# --- Section 10: historical continuation safety, even WITH a relation ------

def test_same_topic_continuation_with_relation_never_verifies():
    proof = _single_proof(
        "Doctor ordered a thyroid ultrasound.",
        "The ultrasound found a suspicious 3 cm nodule and it was sent for biopsy.",
    )
    assert proof.verified is False
    assert proof.rejection_reason == "required_claim_not_preserved"


def test_number_mismatch_with_relation_never_verifies():
    proof = _single_proof("The nodule measured 5 cm.", "The nodule was found.")
    assert proof.verified is False


# --- Section 11: sales/UGC matrix -------------------------------------------

def test_sales_positive_superset_verifies():
    proof = _single_proof(
        "I had bloating.",
        "I had bloating, and it completely went away, plus my energy improved a lot.",
    )
    assert proof.verified is True


def test_sales_negative_dosage_never_verifies():
    proof = _single_proof("I take two gummies every morning.", "They helped my bloating.")
    assert proof.verified is False


def test_sales_negative_negation_flip_never_verifies():
    proof = _single_proof("This serum did not irritate my skin.", "This serum irritated my skin.")
    assert proof.verified is False


def test_sales_same_topic_product_continuation_never_verifies():
    proof = _single_proof("I opened the pouch.", "The gummies tasted strawberry.")
    assert proof.verified is False


# --- Section 12: attribution regression (re-run through the new path) ------

def test_reported_attribution_asymmetry_never_verifies_pre_group():
    proof = _single_proof(
        "It did not work for me.",
        "Some customers said it did not work for them but it worked great for me.",
    )
    assert proof.verified is False


def test_symmetric_attribution_may_still_verify_pre_group():
    proof = _single_proof(
        "The doctor said I have gastritis.",
        "The doctor said I have gastritis and prescribed medication.",
    )
    assert proof.verified is True


# --- Section 13: arbiter attack ---------------------------------------------

def test_always_yes_arbiter_cannot_create_candidate_relation():
    """No relation supplied -- the arbiter must never be consulted, let
    alone substitute for a missing relation."""
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "I had gastritis in 2023.",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        d_attempt="att_1", r_attempt="att_2", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


def test_always_yes_arbiter_cannot_override_same_topic_continuation():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "Doctor ordered a thyroid ultrasound.",
        "The ultrasound found a suspicious 3 cm nodule and it was sent for biopsy.",
        arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


def test_always_yes_arbiter_cannot_override_causal_reversal():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "The medication caused the rash.", "The rash caused the need for medication.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


def test_always_yes_arbiter_cannot_override_diagnosis_substitution():
    arbiter = _AlwaysYesArbiter()
    proof = _single_proof(
        "The test confirmed it was gastritis.", "The test confirmed it was an ulcer.", arbiter=arbiter,
    )
    assert proof.verified is False
    assert arbiter.calls == []


# --- Mechanical garbage / no extractable claims -----------------------------

def test_no_extractable_claims_never_verifies():
    proof = _single_proof("uh...", "She ordered a thyroid ultrasound and found a nodule.")
    assert proof.verified is False
    assert proof.rejection_reason == "no_extractable_claims_uncertain_content"


# --- StoryValidator consumption (Section 7/10/11/15) ------------------------

def test_storyvalidator_suppresses_when_proof_verified():
    discarded = _clip("c_orphan", "I had gastritis in 2023.", selected=False, start=0.0, end=2.0, attempt_id="att_1")
    kept = _clip(
        "c_kept",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        selected=True, start=5.0, end=8.0, attempt_id="att_1",
    )
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    proofs = build_semantic_preservation_proofs(ledger)
    assert "c_orphan" in proofs

    validated = apply_final_story_coherence_validation(draft, semantic_preservation_proofs=proofs)
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
    row = next((f for f in findings if f["clip_id"] == "c_orphan"), None)
    assert row is not None
    assert row["blocking"] is False
    assert row["classification"] == "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION"
    assert row["content_loss_suppressed_by"] == PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION
    assert row["preserving_realization_id"] == "c_kept"
    assert row["nonrequired_omissions"][0]["atom"] == "2023"


def test_storyvalidator_stays_fail_closed_with_no_proof():
    """Same shape, but no relation -- StoryValidator's existing behavior
    (whatever it already was) must be completely unaffected."""
    discarded = _clip("c_orphan", "I had gastritis in 2023.", selected=False, start=0.0, end=2.0, attempt_id="att_1")
    kept = _clip(
        "c_kept",
        "I had digestive problems, had an endoscopy, was diagnosed with gastritis, "
        "and took medication for three months.",
        selected=True, start=5.0, end=8.0, attempt_id="att_2",
    )
    draft = _draft(selected=kept, discarded=discarded)
    ledger = build_semantic_ledger_shadow(draft)
    proofs = build_semantic_preservation_proofs(ledger)
    assert "c_orphan" not in proofs

    without_proofs = apply_final_story_coherence_validation(draft)
    with_empty_lookup = apply_final_story_coherence_validation(draft, semantic_preservation_proofs=proofs)
    findings_a = without_proofs.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
    findings_b = with_empty_lookup.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
    assert findings_a == findings_b


def test_storyvalidator_default_none_byte_identical_to_before_d076():
    """`semantic_preservation_proofs` omitted entirely (every pre-D-076
    call site) must produce a BYTE-IDENTICAL row to explicitly passing
    `None` -- proving the new parameter changes nothing when a caller
    simply doesn't supply it (every existing call site, and the first
    LEGACY/SHADOW-mode StoryValidator pass in universal_clean_cut.py,
    which has no Ledger built yet to source proofs from)."""
    discarded = _clip("c_orphan", "Some ordinary discarded content nobody credits.",
                       selected=False, start=0.0, end=2.0)
    kept = _clip("c_kept", "Something entirely unrelated that was kept.",
                 selected=True, start=5.0, end=8.0)
    draft = _draft(selected=kept, discarded=discarded)
    omitted = apply_final_story_coherence_validation(draft)
    explicit_none = apply_final_story_coherence_validation(draft, semantic_preservation_proofs=None)
    assert (
        omitted.diagnostics["final_story_coherence_validation"]
        == explicit_none.diagnostics["final_story_coherence_validation"]
    )
    row = next(
        f for f in omitted.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
        if f["clip_id"] == "c_orphan"
    )
    assert row["blocking"] is True
    assert "preserving_realization_id" not in row
    assert "preserving_realization_id" not in row
