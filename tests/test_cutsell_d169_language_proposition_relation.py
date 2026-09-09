"""D-169: Language / Transcript Spine, Phase C -- PropositionCandidate +
RelationEvidence. Covers every directive-required fixture category plus
the proposition-vs-family test requirements and no-authority-change proofs.

See cutsell_worker/language_proposition_relation.py's own module docstring
for the full design rationale (including the conflation forensic) this
suite verifies against.
"""
from __future__ import annotations

import inspect

import pytest

from cutsell_worker.language_spine import normalize_language_text
from cutsell_worker.language_utterance_attempt import (
    CONFIDENCE_SUPPORTED,
    LanguageAttempt,
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
)
from cutsell_worker import language_proposition_relation as m
from cutsell_worker.language_proposition_relation import (
    CONFLICT,
    PropositionCandidate,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    RelationEvidence,
    SLOT_CONCLUSION,
    SLOT_CTA,
    SUPPORT,
    UNKNOWN,
    build_claim_signature,
    build_proposition_candidates,
    build_relation_evidence,
    claim_signatures_conflict,
    classify_relation_candidate,
    language_proposition_relation_diagnostics,
    raw_understanding_proposition_reference,
    signatures_describe_same_proposition,
    watch_listen_proposition_reference,
)


def _code_body_excluding_module_docstring(module) -> str:
    source = inspect.getsource(module)
    first = source.find('"""')
    if first == -1:
        return source
    second = source.find('"""', first + 3)
    return source[second + 3:] if second != -1 else source


def _mk(
    text: str, start: float, end: float, *, aid: str,
    meaning: str = MEANING_COMPLETE, restart: bool = False,
    correction: bool = False, continuation: bool = False,
    confidence: str = CONFIDENCE_SUPPORTED,
) -> LanguageAttempt:
    return LanguageAttempt(
        source_asset_id="src1", attempt_id=aid, utterance_ids=(f"u_{aid}",),
        source_start=start, source_end=end, text_raw=text,
        text_normalized=normalize_language_text(text),
        attempt_state="CLEAN_ATTEMPT", meaning_completion=meaning,
        restart_evidence=restart, correction_evidence=correction,
        continuation_evidence=continuation, recording_process_evidence=False,
        confidence=confidence, provenance="ATTEMPT_RECONSTRUCTION",
    )


def _build(attempts):
    props = build_proposition_candidates(tuple(attempts))
    by_id = {a.attempt_id: a for a in attempts}
    relations = build_relation_evidence(props, by_id)
    return props, relations


# ---------------------------------------------------------------------------
# 1. One clean proposition
# ---------------------------------------------------------------------------
def test_01_one_clean_proposition():
    props, relations = _build([_mk("This works great.", 0, 2, aid="a1")])
    assert len(props) == 1
    assert relations == ()
    assert props[0].meaning_completion == MEANING_COMPLETE


# ---------------------------------------------------------------------------
# 2. Same proposition two attempts (retry)
# ---------------------------------------------------------------------------
def test_02_same_proposition_two_attempts_retry():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("This cream removes wrinkles fast, really works.", 3, 5, aid="a2", restart=True),
    ])
    assert len(props) == 2
    assert props[0].attempt_ids != props[1].attempt_ids
    assert signatures_describe_same_proposition(props[0].claim_signature, props[1].claim_signature)
    assert relations[0].relation_candidate == RELATION_RETRY
    # No retry_family_id/family id minted anywhere.
    assert not hasattr(props[0], "retry_family_id")


# ---------------------------------------------------------------------------
# 3. Same topic different proposition
# ---------------------------------------------------------------------------
def test_03_same_topic_different_proposition():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("This cream costs 50 dollars only.", 3, 5, aid="a2"),
    ])
    assert relations[0].relation_candidate == RELATION_DISTINCT_PROPOSITION


# ---------------------------------------------------------------------------
# 4. Same product different claims
# ---------------------------------------------------------------------------
def test_04_same_product_different_claims():
    props, relations = _build([
        _mk("The blender crushes ice easily.", 0, 2, aid="a1"),
        _mk("The blender weighs 5 pounds total.", 3, 5, aid="a2"),
    ])
    assert relations[0].relation_candidate == RELATION_DISTINCT_PROPOSITION


# ---------------------------------------------------------------------------
# 5. Same opener different proposition
# ---------------------------------------------------------------------------
def test_05_same_opener_different_proposition():
    props, relations = _build([
        _mk("Another symptom was fatigue every morning.", 0, 2, aid="a1"),
        _mk("Another problem was rising costs entirely.", 3, 5, aid="a2"),
    ])
    assert relations[0].relation_candidate in (RELATION_DISTINCT_PROPOSITION, RELATION_COMPLEMENTARY)
    assert relations[0].relation_candidate != RELATION_RETRY


# ---------------------------------------------------------------------------
# 6. Retry evidence
# ---------------------------------------------------------------------------
def test_06_retry_evidence():
    props, relations = _build([
        _mk("I really wanted to explain this properly.", 0, 2, aid="a1"),
        _mk("I really wanted to explain this properly, clearly.", 3, 5, aid="a2", restart=True),
    ])
    assert relations[0].relation_candidate == RELATION_RETRY
    assert relations[0].support_status == SUPPORT
    assert relations[0].meaning_conflict is False


# ---------------------------------------------------------------------------
# 7. Continuation evidence
# ---------------------------------------------------------------------------
def test_07_continuation_evidence():
    props, relations = _build([
        _mk("And then we went to the", 0, 2, aid="a1", meaning=MEANING_INCOMPLETE),
        _mk("store to buy everything we needed.", 3, 5, aid="a2", continuation=True),
    ])
    assert relations[0].relation_candidate == RELATION_CONTINUATION
    # Attempts stay physically distinct -- no composition performed.
    assert props[0].attempt_ids != props[1].attempt_ids


# ---------------------------------------------------------------------------
# 8. Correction evidence
# ---------------------------------------------------------------------------
def test_08_correction_evidence():
    props, relations = _build([
        _mk("It costs 50 dollars total price.", 0, 2, aid="a1"),
        _mk("It costs 40 dollars total price.", 3, 5, aid="a2", restart=True, correction=True),
    ])
    assert relations[0].relation_candidate == RELATION_CORRECTION
    assert relations[0].meaning_conflict is True
    # Both before/after claim signatures preserved (not flattened away).
    assert "50" in props[0].claim_signature.numbers
    assert "40" in props[1].claim_signature.numbers


# ---------------------------------------------------------------------------
# 9. Complementary evidence
# ---------------------------------------------------------------------------
def test_09_complementary_evidence():
    props, relations = _build([
        _mk("First symptom was fatigue every morning.", 0, 2, aid="a0"),
        _mk("Second symptom was headaches sometimes.", 3, 5, aid="a1"),
        _mk("Third symptom was nausea too.", 6, 8, aid="a2"),
        _mk("That is everything I noticed overall.", 9, 11, aid="a3"),
    ])
    middle = relations[1]
    assert middle.relation_candidate == RELATION_COMPLEMENTARY
    assert middle.meaning_conflict is False


# ---------------------------------------------------------------------------
# 10. New audience beat
# ---------------------------------------------------------------------------
def test_10_new_audience_beat():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("Now let me show you our second amazing product line.", 20, 22, aid="a2"),
    ])
    assert relations[0].relation_candidate == RELATION_NEW_AUDIENCE_BEAT


# ---------------------------------------------------------------------------
# 11. Distinct proposition
# ---------------------------------------------------------------------------
def test_11_distinct_proposition():
    props, relations = _build([
        _mk("The blender crushes ice easily.", 0, 2, aid="a1"),
        _mk("The warranty lasts 2 years total.", 3, 5, aid="a2"),
    ])
    assert relations[0].relation_candidate == RELATION_DISTINCT_PROPOSITION


# ---------------------------------------------------------------------------
# 12. Uncertain
# ---------------------------------------------------------------------------
def test_12_uncertain():
    props, relations = _build([
        _mk("", 0, 1, aid="a1", meaning=MEANING_UNCERTAIN),
        _mk("Something normal happens next.", 2, 4, aid="a2"),
    ])
    assert relations[0].relation_candidate == RELATION_UNCERTAIN


def test_12b_uncertain_never_forced_retry_or_distinct():
    left = PropositionCandidate(
        source_asset_id="src1", proposition_candidate_id="prop_left", attempt_ids=("a1",),
        source_start=0, source_end=1, text_raw="", text_normalized="",
        claim_signature=build_claim_signature("src1", ""), meaning_completion=MEANING_UNCERTAIN,
        editorial_slot_evidence=SLOT_CTA, confidence="UNKNOWN", provenance="LANGUAGE_ATTEMPT",
    )
    right = PropositionCandidate(
        source_asset_id="src1", proposition_candidate_id="prop_right", attempt_ids=("a2",),
        source_start=2, source_end=3, text_raw="A clean statement here.", text_normalized="a clean statement here.",
        claim_signature=build_claim_signature("src1", "A clean statement here."), meaning_completion=MEANING_COMPLETE,
        editorial_slot_evidence=SLOT_CTA, confidence="SUPPORTED", provenance="LANGUAGE_ATTEMPT",
    )
    evidence = classify_relation_candidate(left, right, semantic_support=SUPPORT, watch_listen_support=CONFLICT)
    assert evidence.relation_candidate == RELATION_UNCERTAIN


# ---------------------------------------------------------------------------
# 13. Negation conflict
# ---------------------------------------------------------------------------
def test_13_negation_conflict():
    props, relations = _build([
        _mk("The treatment works for everyone.", 0, 2, aid="a1"),
        _mk("The treatment does not work for everyone.", 3, 5, aid="a2"),
    ])
    assert relations[0].meaning_conflict is True
    assert relations[0].relation_candidate != RELATION_RETRY


# ---------------------------------------------------------------------------
# 14. Numbers conflict
# ---------------------------------------------------------------------------
def test_14_numbers_conflict():
    left = build_claim_signature("src1", "The price is 50 dollars.")
    right = build_claim_signature("src1", "The price is 40 dollars.")
    assert claim_signatures_conflict(left, right) is True


# ---------------------------------------------------------------------------
# 15. Factual-term difference
# ---------------------------------------------------------------------------
def test_15_factual_term_difference():
    left = build_claim_signature("src1", "The biopsy confirmed a benign tumor.")
    right = build_claim_signature("src1", "The biopsy confirmed a malignant tumor.")
    # Different factual term ("benign" vs "malignant") -- content differs,
    # so these are not even judged the "same proposition" at the coarse
    # token level (an honest V1 limitation, never silently equated).
    assert left.content_tokens != right.content_tokens


# ---------------------------------------------------------------------------
# 16. Same meaning safe normalization
# ---------------------------------------------------------------------------
def test_16_same_meaning_safe_normalization():
    sig = build_claim_signature("src1", "It did NOT cost 50 dollars, ever!!")
    assert sig.negation_present is True
    assert "50" in sig.numbers


# ---------------------------------------------------------------------------
# 17. Conclusion continuation
# ---------------------------------------------------------------------------
def test_17_conclusion_continuation():
    props, relations = _build([
        _mk("So in the end it all comes down to the", 0, 2, aid="a1", meaning=MEANING_INCOMPLETE),
        _mk("consistency of your daily habits.", 3, 5, aid="a2", continuation=True),
    ])
    assert relations[0].relation_candidate == RELATION_CONTINUATION
    assert relations[0].relation_candidate != RELATION_RETRY


# ---------------------------------------------------------------------------
# 18. CTA advisory
# ---------------------------------------------------------------------------
def test_18_cta_advisory():
    props, _ = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("Get yours today and save.", 3, 5, aid="a2"),
    ])
    assert props[-1].editorial_slot_evidence == SLOT_CTA


# ---------------------------------------------------------------------------
# 19. Conclusion advisory
# ---------------------------------------------------------------------------
def test_19_conclusion_advisory():
    # "only" + a number -- semantic_claims.classify_claim's own generic,
    # already-existing UNIQUE_CONCLUSION marker (never Video00-specific).
    props, _ = _build([
        _mk("Only 5 percent of people ever succeed.", 0, 2, aid="a1"),
    ])
    assert props[0].editorial_slot_evidence == SLOT_CONCLUSION


# ---------------------------------------------------------------------------
# 20. Correction with numbers
# ---------------------------------------------------------------------------
def test_20_correction_with_numbers():
    props, relations = _build([
        _mk("It costs 50 dollars total price.", 0, 2, aid="a1"),
        _mk("It costs 40 dollars total price.", 3, 5, aid="a2", restart=True, correction=True),
    ])
    assert relations[0].relation_candidate == RELATION_CORRECTION
    assert relations[0].meaning_conflict is True
    assert props[0].claim_signature.numbers != props[1].claim_signature.numbers


# ---------------------------------------------------------------------------
# 21. One proposition no family
# ---------------------------------------------------------------------------
def test_21_one_proposition_no_family():
    props, relations = _build([_mk("A single clean statement.", 0, 2, aid="a1")])
    assert len(props) == 1
    assert relations == ()
    assert not hasattr(props[0], "retry_family_id")


# ---------------------------------------------------------------------------
# 22. No retry_family_id minted
# ---------------------------------------------------------------------------
def test_22_no_retry_family_id_minted():
    fields = set(PropositionCandidate.__dataclass_fields__) | set(RelationEvidence.__dataclass_fields__)
    assert "retry_family_id" not in fields
    body = _code_body_excluding_module_docstring(m)
    assert "mint_retry_family_id" not in body


# ---------------------------------------------------------------------------
# 23. Proposition id separate namespace
# ---------------------------------------------------------------------------
def test_23_proposition_id_separate_namespace():
    props, _ = _build([_mk("A single clean statement.", 0, 2, aid="a1")])
    assert props[0].proposition_candidate_id.startswith("prop_")
    assert not props[0].proposition_candidate_id.startswith(("idea_", "att_", "latt_", "lutt_", "span_", "real_"))


# ---------------------------------------------------------------------------
# 24. Proposition id deterministic
# ---------------------------------------------------------------------------
def test_24_proposition_id_deterministic():
    attempts1 = [_mk("A single clean statement.", 0, 2, aid="a1")]
    attempts2 = [_mk("A single clean statement.", 0, 2, aid="a1")]
    props1, _ = _build(attempts1)
    props2, _ = _build(attempts2)
    assert props1[0].proposition_candidate_id == props2[0].proposition_candidate_id


# ---------------------------------------------------------------------------
# 25. Relation ordering deterministic
# ---------------------------------------------------------------------------
def test_25_relation_ordering_deterministic():
    attempts = [
        _mk("First statement here today.", 0, 2, aid="a1"),
        _mk("Second statement here today.", 3, 5, aid="a2"),
        _mk("Third statement here today.", 6, 8, aid="a3"),
    ]
    _, r1 = _build(attempts)
    _, r2 = _build(list(reversed(attempts)))
    assert [ (r.left_proposition_candidate_id, r.right_proposition_candidate_id) for r in r1 ] == \
           [ (r.left_proposition_candidate_id, r.right_proposition_candidate_id) for r in r2 ]


# ---------------------------------------------------------------------------
# 26. Source identity preserved
# ---------------------------------------------------------------------------
def test_26_source_identity_preserved():
    props, _ = _build([_mk("A single clean statement.", 0, 2, aid="a1")])
    assert props[0].source_asset_id == "src1"


# ---------------------------------------------------------------------------
# 27. Timeline preserved
# ---------------------------------------------------------------------------
def test_27_timeline_preserved():
    attempt = _mk("A single clean statement.", 1.5, 3.5, aid="a1")
    props, _ = _build([attempt])
    assert props[0].source_start == 1.5
    assert props[0].source_end == 3.5


# ---------------------------------------------------------------------------
# 28. Confidence categorical
# ---------------------------------------------------------------------------
def test_28_confidence_categorical():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("This cream removes wrinkles fast, really works.", 3, 5, aid="a2", restart=True),
    ])
    for p in props:
        assert p.confidence in ("SUPPORTED", "WEAK", "UNKNOWN", "MIXED")
    for r in relations:
        assert r.confidence in ("SUPPORTED", "WEAK", "UNKNOWN", "MIXED")


# ---------------------------------------------------------------------------
# 29. Provenance retained
# ---------------------------------------------------------------------------
def test_29_provenance_retained():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("This cream removes wrinkles fast, really works.", 3, 5, aid="a2", restart=True),
    ])
    assert props[0].provenance == m.PROVENANCE_LANGUAGE_ATTEMPT
    assert relations[0].provenance


# ---------------------------------------------------------------------------
# 30. Old serialized ids unaffected
# ---------------------------------------------------------------------------
def test_30_old_serialized_ids_unaffected():
    import subprocess
    for path in ("cutsell_worker/canonical_identity.py", "cutsell_worker/contracts.py", "cutsell_worker/pipeline.py"):
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", path],
            cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
        )
        assert result.stdout.strip() == "", path


# ---------------------------------------------------------------------------
# 31. No Family Formation change
# ---------------------------------------------------------------------------
def test_31_no_family_formation_change():
    for name in ("take_grouping", "take_grouping_provider", "hybrid_session_cleanup"):
        source = inspect.getsource(__import__(f"cutsell_worker.{name}", fromlist=["_"]))
        assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 32. No D-150 change
# ---------------------------------------------------------------------------
def test_32_no_d150_change():
    source = inspect.getsource(__import__("cutsell_worker.semantic_authority_observability", fromlist=["_"]))
    assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 33. No D-158 change
# ---------------------------------------------------------------------------
def test_33_no_d158_change():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/attempt_relationship_authority.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 34. No D-161 change
# ---------------------------------------------------------------------------
def test_34_no_d161_change():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_relation_discovery.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 35. No DeliveryScorer change
# ---------------------------------------------------------------------------
def test_35_no_deliveryscorer_change():
    source = inspect.getsource(__import__("cutsell_worker.take_judge", fromlist=["_"]))
    assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 36. No BestTake change
# ---------------------------------------------------------------------------
def test_36_no_besttake_change():
    for name in ("deterministic_best_take_authority", "take_judge"):
        source = inspect.getsource(__import__(f"cutsell_worker.{name}", fromlist=["_"]))
        assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 37. D-163 regression
# ---------------------------------------------------------------------------
def test_37_d163_regression():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_besttake_evidence.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 38. D-167 regression
# ---------------------------------------------------------------------------
def test_38_d167_regression():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/watch_listen_zone_usability_v2.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 39. No Boundary change
# ---------------------------------------------------------------------------
def test_39_no_boundary_change():
    source = inspect.getsource(__import__("cutsell_worker.boundary_engine_pass", fromlist=["_"]))
    assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 40. No Pacing change
# ---------------------------------------------------------------------------
def test_40_no_pacing_change():
    source = inspect.getsource(__import__("cutsell_worker.dialogue_pacing_transition", fromlist=["_"]))
    assert "language_proposition_relation" not in source


# ---------------------------------------------------------------------------
# 41. No provider/network
# ---------------------------------------------------------------------------
def test_41_no_provider_network():
    body = _code_body_excluding_module_docstring(m)
    for forbidden in ("openai", "gemini", "requests.", "httpx.", "urllib.request", "socket."):
        assert forbidden not in body.lower()


# ---------------------------------------------------------------------------
# 42. No composite implementation (attempts never physically merged)
# ---------------------------------------------------------------------------
def test_42_no_composite_implementation():
    body = _code_body_excluding_module_docstring(m)
    for forbidden in ("compose", "composite", "merge_attempt", "physically join"):
        assert forbidden not in body.lower()


# ---------------------------------------------------------------------------
# Additional structural/contract tests beyond the 42-item matrix.
# ---------------------------------------------------------------------------
def test_language_utterance_attempt_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/language_utterance_attempt.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_language_spine_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/language_spine.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_semantic_claims_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/semantic_claims.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_semantic_idea_equivalence_untouched():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/semantic_idea_equivalence.py"],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_not_imported_by_any_production_call_site():
    call_sites = (
        "pipeline", "flow_b", "take_grouping", "take_grouping_provider",
        "hybrid_session_cleanup", "semantic_idea_equivalence",
        "attempt_relationship_authority", "deterministic_best_take_authority",
        "take_judge", "watch_listen_besttake_evidence",
        "watch_listen_zone_usability_v2", "boundary_engine_pass",
        "dialogue_pacing_transition", "semantic_authority_observability",
    )
    for name in call_sites:
        source = inspect.getsource(__import__(f"cutsell_worker.{name}", fromlist=["_"]))
        assert "language_proposition_relation" not in source, name


def test_empty_input_returns_empty():
    assert build_proposition_candidates(()) == ()
    assert build_relation_evidence((), {}) == ()


def test_diagnostics_shape():
    props, relations = _build([
        _mk("This cream removes wrinkles fast.", 0, 2, aid="a1"),
        _mk("This cream removes wrinkles fast, really works.", 3, 5, aid="a2", restart=True),
    ])
    diag = language_proposition_relation_diagnostics(props, relations)
    required = {
        "language_proposition_candidate_count", "language_relation_evidence_count",
        "retry_evidence_count", "continuation_evidence_count", "correction_evidence_count",
        "complementary_evidence_count", "new_beat_evidence_count",
        "distinct_proposition_evidence_count", "uncertain_relation_evidence_count",
        "cta_slot_candidate_count", "conclusion_slot_candidate_count",
        "proposition_conflict_count", "meaning_conflict_count",
    }
    assert required.issubset(diag.keys())
    assert diag["language_proposition_candidate_count"] == 2
    assert diag["language_relation_evidence_count"] == 1
    assert diag["retry_evidence_count"] == 1


def test_raw_understanding_proposition_reference_additive_only():
    row = raw_understanding_proposition_reference("span_abc", proposition_candidate_id="prop_x")
    assert row == {"span_id": "span_abc", "proposition_candidate_id": "prop_x"}


def test_watch_listen_proposition_reference_additive_only():
    row = watch_listen_proposition_reference("uspan_abc", proposition_candidate_id="prop_y")
    assert row == {"understanding_span_id": "uspan_abc", "proposition_candidate_id": "prop_y"}


def test_semantic_support_caller_supplied_never_computed_here():
    props, _ = _build([
        _mk("First statement here today.", 0, 2, aid="a1"),
        _mk("Second statement here today.", 3, 5, aid="a2"),
    ])
    key = (props[0].proposition_candidate_id, props[1].proposition_candidate_id)
    by_id = {"a1": _mk("First statement here today.", 0, 2, aid="a1"), "a2": _mk("Second statement here today.", 3, 5, aid="a2")}
    relations = build_relation_evidence(props, by_id, semantic_support_by_pair={key: SUPPORT})
    assert relations[0].semantic_support == SUPPORT


def test_no_family_or_authority_field_on_propositioncandidate():
    forbidden = {"retry_family_id", "take_group_id", "selected", "winner"}
    assert forbidden.isdisjoint(set(PropositionCandidate.__dataclass_fields__))
    assert forbidden.isdisjoint(set(RelationEvidence.__dataclass_fields__))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
