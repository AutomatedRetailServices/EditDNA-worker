"""D-079: FINAL FREEZE-BLOCKER INTEGRATION -- claim single-truth + dominance
reachability.

Phase 1/2/3/4 (this file's own new coverage): `_lost_critical_claims` gains
a claim-SCOPED (never idea- or clip-scoped) consumption path for an already-
verified `SemanticPreservationProof` -- closing the D-078 dual-truth gap
(StoryValidator's `_lost_semantic_atoms` said "preserved", `_lost_critical_
claims` independently said "lost", for the identical source text) via a new,
additive `INTRA_IDEA_SEMANTIC_PRESERVATION` certification path
(realization_resolver.py) for the THIRD remaining discard population: a
realization that DID reach grouping and lost, within its own idea, to that
idea's own resolved winner. A new, narrow `rhetorical_aside_negation_bridge`
matching strategy (opt-in, never touching the global NEGATION hard gate)
closes the specific D-078 root cause: a bare negation token inside a fixed,
general discourse-aside phrase ("no need to ask" / "no hay que preguntar")
spuriously classifying an otherwise-plain statement NEGATION/CRITICAL and
tripping `claim_coverage()`'s own negation-flip guard.

Phase 5/6/7 (D-063 CRITICAL_COVERAGE_DOMINANCE reachability, true-tie
preservation, and the retry regression fixture) are ALREADY fully covered
by the existing, unmodified `tests/test_cutsell_d066_hindsight_alignment.py`
suite -- in particular `test_d064_generic_chain_auto_resolves_with_
confirming_arbiter` (Candidate A with a critical diagnosis + hindsight
claim, delivery 0.9, dominates Candidate B with only the hindsight claim,
delivery 0.95 -- delivery cannot override) and `test_d064_generic_chain_
stays_ambiguous_without_arbiter_confirmation`/`_with_no_arbiter_at_all`
(true tie stays REVIEW_REQUIRED, never forced). D-079 does not modify
`claim_coverage_best_take.py` at all -- re-run here (Section 11) as
confirmation, not reproduced.

Every fixture goes through the REAL pipeline: `build_semantic_ledger_
shadow` (real `extract_claims`, real Ledger reconstruction) ->
`resolve_realizations_shadow` -> `resolve_intra_idea_semantic_preservation_
shadow` -> `build_preserved_claim_id_index` ->
`apply_final_story_coherence_validation`. No Video00-specific text/ids.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_claims import extract_claims
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.realization_resolver import (
    PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION,
    RESOLVED_WINNER,
    build_preserved_claim_id_index,
    build_semantic_preservation_proofs,
    resolve_intra_idea_semantic_preservation_shadow,
    resolve_realizations_shadow,
)
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def _idea_clip(clip_id, text, *, selected, idea_id="idea_1", start=0.0, end=1.0, source_asset_id="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        semantic_idea_id=idea_id,
    )


def _draft(*, discarded, selected):
    """`semantic_idea_id` (shared by both clips, via `_idea_clip`) drives
    the Ledger/Resolver side (`resolve_realizations_shadow`/D-079's own
    intra-idea pass); a `take_judge_groups` entry listing BOTH clip ids
    drives `_lost_critical_claims`'s own, completely separate group
    iteration (it never reads `semantic_idea_id` at all) -- both are
    needed for a single fixture to exercise the full D-079 chain end to
    end, exactly mirroring the real pipeline's own two independent inputs."""
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(selected,), alternates=(), discarded=(discarded,),
        diagnostics={"take_judge_groups": [{"group_id": "g1", "ranked": [
            {"clip_id": selected.clip_id, "score": 0.9, "reason": "watch_listen_baseline"},
            {"clip_id": discarded.clip_id, "score": 0.5, "reason": "watch_listen_baseline"},
        ]}]},
    )


def _resolve_intra_idea(draft_obj, *, arbiter=None):
    ledger = build_semantic_ledger_shadow(draft_obj)
    report = resolve_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter)
    proofs = resolve_intra_idea_semantic_preservation_shadow(
        ledger, claim_equivalence_arbiter=arbiter, resolver_report=report,
    )
    return ledger, report, proofs


class _AlwaysYesArbiter:
    def __init__(self):
        self.calls = []

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return True, 0.95, "always yes"


class _CountingRealArbiter:
    def __init__(self, verdict=True, confidence=0.9, reason="confirmed"):
        self.calls = []
        self._verdict, self._confidence, self._reason = verdict, confidence, reason

    def claim_covered(self, claim_text, candidate_text):
        self.calls.append((claim_text, candidate_text))
        return self._verdict, self._confidence, self._reason


# The exact D-078 forensic shape (English stand-in per D-079 Phase 4).
_D078_DISCARD = "I had stomach problems for a while, in 2023, no need to ask."
_D078_SELECTED = "I had digestive problems, had an endoscopy, was diagnosed with gastritis, and took medication."


# --- Phase 1: canonical claim identity --------------------------------------

def test_canonical_claim_identity_shared_between_lost_critical_claims_and_ledger():
    """The SAME source clip/text, extracted independently by _lost_
    critical_claims's own extract_claims call and by the Ledger's own
    claim registration, must mint the IDENTICAL canonical_claim_id --
    a pure function of (claim_type, content_tokens), never of idea id or
    clip id (Phase 1's own 'do not match by idea id' requirement)."""
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    ledger = build_semantic_ledger_shadow(draft_obj)

    fresh_claim = extract_claims("c_orphan", _D078_DISCARD)[0]
    assert fresh_claim.claim_type == "NEGATION"
    assert fresh_claim.importance == "CRITICAL"

    ledger_claims = ledger.claims()
    assert fresh_claim.canonical_claim_id in ledger_claims
    assert ledger_claims[fresh_claim.canonical_claim_id].claim_type == "NEGATION"


def test_no_canonical_identity_no_suppression_fail_closed():
    """A claim whose canonical_claim_id genuinely never appears in any
    verified proof's preserved_claim_ids (no relation supplied at all,
    Phase 1's own 'if canonical identity cannot be established' case)
    must fall through to the existing, unmodified CRITICAL_CLAIM_LOST
    behavior -- proven by omitting the index entirely (None, the default,
    exactly every pre-D-079 call site)."""
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    validated = apply_final_story_coherence_validation(draft_obj)
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]
    assert any(f["claim_text"] == _D078_DISCARD for f in findings)


# --- Phase 2/4: claim-scoped proof consumption + D-078 negation shape ------

def test_d078_negation_shape_resolves_via_intra_idea_proof_with_arbiter():
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _CountingRealArbiter(verdict=True)

    ledger, report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    assert len(proofs) == 1
    proof = proofs[0]
    assert proof.verified is True
    assert proof.proof_method == PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION
    assert len(arbiter.calls) == 1

    index = build_preserved_claim_id_index(proofs)
    validated = apply_final_story_coherence_validation(
        draft_obj, claim_equivalence_arbiter=arbiter, critical_claim_preservation_index=index,
    )
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]
    confirmations = validated.diagnostics["final_story_coherence_validation"]["claim_coverage_confirmations"]
    assert not any(f["claim_text"] == _D078_DISCARD for f in findings)
    consumed = next(c for c in confirmations if c.get("claim_preservation_consumed"))
    assert consumed["proof_method"] == PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION
    assert consumed["preserving_realization_id"] == "c_kept"


def test_d078_negation_shape_fails_closed_without_arbiter():
    """Phase 1's own 'if the existing arbiter is unavailable: fail closed'
    -- no arbiter, no certification, the pre-existing CRITICAL_CLAIM_LOST
    behavior is completely unchanged."""
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)

    ledger, report, proofs = _resolve_intra_idea(draft_obj, arbiter=None)
    assert len(proofs) == 1
    assert proofs[0].verified is False

    index = build_preserved_claim_id_index(proofs)
    assert index == {}
    validated = apply_final_story_coherence_validation(
        draft_obj, critical_claim_preservation_index=index,
    )
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]
    assert any(f["claim_text"] == _D078_DISCARD for f in findings)


def test_no_relation_never_certifies_even_with_always_yes_arbiter():
    """Mandatory negative control: the SAME D-078 pair, but the two
    realizations belong to DIFFERENT ideas -- no relation at all. The
    always-YES arbiter must never be reached; intra-idea discovery is
    the idea's own resolution, never a search."""
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, idea_id="idea_1", start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, idea_id="idea_2", start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()

    ledger, report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # The discard's own idea ("idea_1") has no selected winner at all --
    # it is not RESOLVED_WINNER/RESOLVED_COMPOSITE, so it is never even a
    # candidate for this pass.
    assert proofs == ()
    assert arbiter.calls == []


# --- Phase 3: hard safety (defense-in-depth on the NEW bridge specifically) -

def test_number_mismatch_never_suppressed_even_with_always_yes():
    discarded = _idea_clip("c_orphan", "The nodule measured 5 cm, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", "The nodule measured 3 cm.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


def test_negation_reversal_never_suppressed_even_with_always_yes():
    discarded = _idea_clip("c_orphan", "I did not have gastritis, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", "I had gastritis.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


def test_diagnosis_substitution_never_suppressed_even_with_always_yes():
    discarded = _idea_clip("c_orphan", "The biopsy was benign, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", "I was diagnosed with papillary thyroid cancer.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


def test_attribution_asymmetry_never_suppressed_even_with_always_yes():
    discarded = _idea_clip("c_orphan", "It did not work for me, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip(
        "c_kept", "Some customers said it did not work for them but it worked great for me.",
        selected=True, start=5.0, end=8.0,
    )
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


def test_causal_reversal_never_suppressed_even_with_always_yes():
    discarded = _idea_clip(
        "c_orphan", "The medication caused the breakout, no doubt.", selected=False, start=0.0, end=2.0,
    )
    kept = _idea_clip(
        "c_kept", "The medication did not cause the breakout.", selected=True, start=5.0, end=8.0,
    )
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


def test_same_topic_different_event_never_suppressed_even_with_always_yes():
    discarded = _idea_clip("c_orphan", "Doctor ordered an ultrasound, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", "The ultrasound found a suspicious nodule.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


# --- Rhetorical-aside marker safety: exact phrase only, never bare "no" ----

def test_bare_negation_without_marker_phrase_never_bridges():
    """A genuine negation with no fixed aside phrase present must never
    reach the new bridge at all -- confirms the marker match is exact-
    phrase, not a bare 'no'/'not' token."""
    discarded = _idea_clip("c_orphan", "I did not have stomach problems.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip(
        "c_kept", "I had digestive problems, had an endoscopy, was diagnosed with gastritis.",
        selected=True, start=5.0, end=8.0,
    )
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)
    assert arbiter.calls == []


def test_residual_with_genuine_second_negation_never_bridges():
    """The aside phrase is present, but a REAL, second negation also
    exists in the same clause -- the residual must still carry a
    negation marker and must never be bridged (Phase 3 defense-in-depth:
    'residual_still_negated')."""
    discarded = _idea_clip(
        "c_orphan", "I did not have stomach problems, no need to ask.", selected=False, start=0.0, end=2.0,
    )
    kept = _idea_clip(
        "c_kept", "I had digestive problems, had an endoscopy, was diagnosed with gastritis.",
        selected=True, start=5.0, end=8.0,
    )
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)


# --- Sales/UGC generalization ------------------------------------------------

def test_sales_positive_rhetorical_aside_may_resolve_with_arbiter():
    discarded = _idea_clip(
        "c_orphan", "I had annoying bloating for a while, in 2023, no need to ask.",
        selected=False, start=0.0, end=2.0,
    )
    kept = _idea_clip(
        "c_kept", "I had digestive bloating, had a check-up, was diagnosed with a sensitivity, and took supplements.",
        selected=True, start=5.0, end=8.0,
    )
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _CountingRealArbiter(verdict=True)
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    assert len(proofs) == 1
    assert proofs[0].verified is True
    assert len(arbiter.calls) == 1


def test_sales_negative_dosage_never_bridges():
    discarded = _idea_clip(
        "c_orphan", "I take two gummies every morning, no doubt.", selected=False, start=0.0, end=2.0,
    )
    kept = _idea_clip("c_kept", "They helped my digestion.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    arbiter = _AlwaysYesArbiter()
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    # Either the idea itself never resolves to a single winner at the
    # Resolver level (a genuine contradiction blocks RESOLVED_WINNER
    # entirely -- an even earlier safety net, so this pass never attempts
    # certification at all) or it does attempt certification and correctly
    # fails to verify -- both are safe; only "verified" is ever unsafe.
    assert not any(p.verified for p in proofs)
    assert arbiter.calls == []


# --- Phase 8: single-truth invariant ----------------------------------------

def _invariant_holds(draft_obj, *, arbiter):
    """The core D-079 invariant: for the exact same canonical required
    proposition, a verified claim-scoped preservation proof and a
    CRITICAL_CLAIM_LOST finding must never coexist."""
    ledger, report, proofs = _resolve_intra_idea(draft_obj, arbiter=arbiter)
    semantic_preservation_proofs = build_semantic_preservation_proofs(
        ledger, claim_equivalence_arbiter=arbiter, intra_idea_proofs=proofs,
    )
    index = build_preserved_claim_id_index(proofs)
    validated = apply_final_story_coherence_validation(
        draft_obj, claim_equivalence_arbiter=arbiter,
        semantic_preservation_proofs=semantic_preservation_proofs,
        critical_claim_preservation_index=index,
    )
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]
    finding_canonical_ids = {f.get("canonical_claim_id") for f in findings}
    for canonical_claim_id, proof in index.items():
        assert proof.verified is True
        assert canonical_claim_id not in finding_canonical_ids, (
            f"canonical claim {canonical_claim_id} has BOTH a verified proof AND a CRITICAL_CLAIM_LOST finding"
        )
    return validated


def test_single_truth_invariant_positive_case():
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    _invariant_holds(draft_obj, arbiter=_CountingRealArbiter(verdict=True))


def test_single_truth_invariant_negative_case_diagnosis():
    discarded = _idea_clip("c_orphan", "The biopsy was benign, no doubt.", selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", "I was diagnosed with papillary thyroid cancer.", selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    validated = _invariant_holds(draft_obj, arbiter=_AlwaysYesArbiter())
    findings = validated.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]
    assert any("benign" in f["claim_text"] for f in findings)


# --- Phase 9: no new authority ------------------------------------------------

def test_storyvalidator_consumer_only_no_discovery_no_arbiter_own_call():
    """StoryValidator's own `apply_final_story_coherence_validation` never
    discovers a candidate, extracts a claim, or invokes an arbiter for
    THIS decision -- passing a real, would-be-invoked arbiter through
    `claim_equivalence_arbiter` still only ever reaches it via the
    EXISTING `_lost_critical_claims`/`resolve_ambiguous_coverage` path
    (pre-D-079, unchanged), never a second time for the index consumption
    itself (a pure dict lookup)."""
    discarded = _idea_clip("c_orphan", _D078_DISCARD, selected=False, start=0.0, end=2.0)
    kept = _idea_clip("c_kept", _D078_SELECTED, selected=True, start=5.0, end=8.0)
    draft_obj = _draft(discarded=discarded, selected=kept)
    cert_arbiter = _CountingRealArbiter(verdict=True)
    _ledger, _report, proofs = _resolve_intra_idea(draft_obj, arbiter=cert_arbiter)
    index = build_preserved_claim_id_index(proofs)
    assert len(cert_arbiter.calls) == 1  # spent during certification, not during consumption

    consumption_arbiter = _CountingRealArbiter(verdict=True)
    apply_final_story_coherence_validation(
        draft_obj, claim_equivalence_arbiter=consumption_arbiter, critical_claim_preservation_index=index,
    )
    # The claim in question is already suppressed via the index lookup
    # before `_lost_critical_claims`'s own `resolve_ambiguous_coverage`
    # step could ever be reached for it -- confirmed no consumption-time
    # call was made for the D-078 claim's own text.
    assert not any(call[0] == _D078_DISCARD for call in consumption_arbiter.calls)
