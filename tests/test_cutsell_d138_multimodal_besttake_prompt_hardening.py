"""D-138 -- Phase 2B prompt/decision-contract hardening, offline markers.

docs/CUTSELL_DECISIONS.md D-138 (post D-137 verdict D). These tests prove
the hardened `INSTRUCTION` constant in `multimodal_besttake_openai.py`
carries the required GENERAL decision-contract doctrine -- never exact
prose, always semantic contract markers -- and never leaks a fixture-
specific phrase, a Video00-specific detail, a reference/oracle value, or a
false audio-perception claim. No provider call, no network, no eval
harness invocation -- pure static-string assertions on the production
prompt constant, exactly like D-136's own `test_no_video00_specific_
language_in_prompt`-shaped tests, extended for D-138's new doctrine.
"""
from __future__ import annotations

from cutsell_worker.multimodal_besttake_openai import INSTRUCTION

# ---------------------------------------------------------------------------
# 1) abstention allowed
# ---------------------------------------------------------------------------

def test_prompt_states_abstention_is_not_required_to_pick_a_winner():
    assert "NOT required to pick a winner" in INSTRUCTION


def test_prompt_states_uncertain_is_preferred_over_manufactured_certainty():
    lowered = INSTRUCTION.lower()
    assert "uncertain" in lowered and "preferred" in lowered
    assert "manufacturing certainty" in lowered


def test_prompt_states_uncertain_may_carry_high_confidence():
    normalized = " ".join(INSTRUCTION.split())
    assert "confident UNCERTAIN" in normalized


# ---------------------------------------------------------------------------
# 2) equivalent allowed
# ---------------------------------------------------------------------------

def test_prompt_states_equivalent_is_a_valid_non_forced_outcome():
    assert "EQUIVALENT" in INSTRUCTION
    assert "do not force a preference" in INSTRUCTION.lower() or "do not force" in INSTRUCTION.lower()


def test_prompt_gives_explicit_equivalent_rule_conditions():
    assert "Return EQUIVALENT when" in INSTRUCTION


# ---------------------------------------------------------------------------
# 3) complementary ambiguity -> uncertain
# ---------------------------------------------------------------------------

def test_prompt_requires_uncertain_for_complementary_content_shape():
    lowered = INSTRUCTION.lower()
    assert "complementary material" in lowered
    assert "you must return uncertain" in lowered


def test_prompt_forbids_collapsing_complementary_content_or_compositing():
    lowered = INSTRUCTION.lower()
    assert "never collapse complementary content" in lowered
    assert "never invent a composite" in lowered
    assert "never decide which unique fact could be" in lowered


# ---------------------------------------------------------------------------
# 4) Boundary ENTRY/EXIT distinction
# ---------------------------------------------------------------------------

def test_prompt_teaches_entry_exit_defect_is_not_a_bad_take():
    normalized = " ".join(INSTRUCTION.split())
    assert "GOOD_TAKE_TRIM_ENTRY" in normalized
    assert "GOOD_TAKE_TRIM_EXIT" in normalized
    assert "does NOT make an otherwise-good take globally worse" in normalized


def test_prompt_references_boundary_editability_note_field():
    assert "boundary_editability_note" in INSTRUCTION


# ---------------------------------------------------------------------------
# 5) D-123 structured-conflict respect
# ---------------------------------------------------------------------------

def test_prompt_names_structured_conflict_check_step():
    assert "STRUCTURED-CONFLICT CHECK" in INSTRUCTION


def test_prompt_teaches_semantic_vs_deliveryscore_disagreement_shape():
    assert 'semantic_label' in INSTRUCTION
    assert "deliveryscore_summary" in INSTRUCTION
    assert "not yours to re-open" in INSTRUCTION


def test_prompt_requires_respecting_semantic_winner_or_uncertain_on_conflict():
    normalized = " ".join(INSTRUCTION.split())
    assert 'respect the semantic_label="winner" candidate, or return UNCERTAIN' in normalized


def test_prompt_requires_overwhelming_evidence_to_override_structured_conflict():
    assert "overwhelming visual evidence" in INSTRUCTION
    assert "subjectively prefer" in INSTRUCTION


# ---------------------------------------------------------------------------
# 6) no forced BEST_TAKE
# ---------------------------------------------------------------------------

def test_prompt_states_best_take_is_not_the_default():
    assert "do not default to BEST_TAKE merely" in INSTRUCTION


def test_prompt_gives_explicit_best_take_validity_conditions():
    assert "BEST_TAKE is valid ONLY when ALL of the following hold" in INSTRUCTION
    assert "If any of these do not clearly\n    \"hold, do not return BEST_TAKE.".replace("\n    \"", " ") in INSTRUCTION.replace("\n", " ") \
        or "do not return BEST_TAKE" in INSTRUCTION


# ---------------------------------------------------------------------------
# 7) no fixture-specific words
# ---------------------------------------------------------------------------

_FIXTURE_SPECIFIC_TERMS = (
    "pimples", "papillary", "stomach", "gynaecolog", "sonography",
    "vamos", "diagnosis", "hereditary", "acné", "acne",
    "tg_eval_", "tg_ef754f8f610ab360df", "tg_dfa8f59296237ae030",
)


def test_prompt_contains_no_fixture_specific_terms():
    lowered = INSTRUCTION.lower()
    for term in _FIXTURE_SPECIFIC_TERMS:
        assert term not in lowered, f"fixture-specific term {term!r} leaked into production prompt"


# ---------------------------------------------------------------------------
# 8) no Video00-specific language
# ---------------------------------------------------------------------------

_VIDEO00_SPECIFIC_TERMS = (
    "video00", "raw run", "modal", "runpod", "d-097", "d-123", "d-126",
    "d-127", "d-128", "d-136", "d-137", "d-138",
)


def test_prompt_contains_no_video00_or_decision_number_language():
    lowered = INSTRUCTION.lower()
    for term in _VIDEO00_SPECIFIC_TERMS:
        assert term not in lowered, f"Video00/decision-id-specific term {term!r} leaked into production prompt"


# ---------------------------------------------------------------------------
# 9) no reference oracle leakage
# ---------------------------------------------------------------------------

_ORACLE_LEAKAGE_TERMS = (
    "should_select", "should_equivalent", "should_uncertain", "should_trim",
    "expected_outcome", "human gold", "cut.ai", "cutai", "oracle",
    "reference winner", "expected answer",
)


def test_prompt_contains_no_oracle_or_expected_answer_leakage():
    lowered = INSTRUCTION.lower()
    for term in _ORACLE_LEAKAGE_TERMS:
        assert term not in lowered, f"oracle/expected-answer term {term!r} leaked into production prompt"


# ---------------------------------------------------------------------------
# 10) no claim of audio hearing
# ---------------------------------------------------------------------------

def test_prompt_explicitly_disclaims_audio_perception():
    lowered = INSTRUCTION.lower()
    assert "you cannot hear tone, cadence" in lowered
    assert "audio clipping" in lowered


def test_prompt_never_claims_to_hear_or_listen():
    lowered = INSTRUCTION.lower()
    assert "you can hear" not in lowered
    assert "you can listen" not in lowered
    assert "listening to" not in lowered
    assert "you hear" not in lowered


# ---------------------------------------------------------------------------
# Bonus: canonical decision hierarchy ordering markers (STEP 1..5 present
# and in the directive-required order: meaning sufficiency -> relationship
# -> performance -> editability/boundary -> abstention).
# ---------------------------------------------------------------------------

def test_prompt_states_the_canonical_decision_hierarchy_in_order():
    step1 = INSTRUCTION.index("STEP 1 MEANING SUFFICIENCY")
    step2 = INSTRUCTION.index("STEP 2 RELATIONSHIP CHECK")
    step2b = INSTRUCTION.index("STEP 2B STRUCTURED-CONFLICT CHECK")
    step3 = INSTRUCTION.index("STEP 3 PERFORMANCE QUALITY")
    step4 = INSTRUCTION.index("STEP 4 EDITABILITY / BOUNDARY")
    step5 = INSTRUCTION.index("STEP 5 ABSTENTION")
    assert step1 < step2 < step2b < step3 < step4 < step5


def test_prompt_output_vocabulary_unchanged_from_d136():
    # D-138 is prompt-only -- the bounded output vocabulary and JSON schema
    # must remain byte-for-byte identical to D-136's own contract.
    assert (
        'Return JSON only, no prose: {"outcome":"BEST_TAKE|EQUIVALENT|'
        'GOOD_TAKE_TRIM_ENTRY|GOOD_TAKE_TRIM_EXIT|UNCERTAIN",'
        '"best_take_candidate_id":"<id or null>","confidence":0.0,'
    ) in INSTRUCTION


def test_prompt_reason_field_guidance_names_general_bases_not_mandatory_schema():
    # Directive: strengthen reason guidance without an unnecessary schema
    # break -- the JSON schema's "reason" value stays a single free-text
    # string (no new field), but the prompt names the 8 general bases.
    for basis in (
        "CLEAR_PERFORMANCE_SUPERIORITY",
        "EQUIVALENT_PERFORMANCE",
        "REMOVABLE_ENTRY_DEFECT",
        "REMOVABLE_EXIT_DEFECT",
        "COMPLEMENTARY_OR_RELATIONSHIP_AMBIGUITY",
        "MIXED_PERFORMANCE_EVIDENCE",
        "INSUFFICIENT_VISUAL_EVIDENCE",
        "STRUCTURED_CONFLICT_UNRESOLVED",
    ):
        assert basis in INSTRUCTION
    assert '"reason":"<one short sentence>"' not in INSTRUCTION  # old D-136 schema string replaced
    assert '"reason":"<BASIS_LABEL: one short sentence>"' in INSTRUCTION


def test_prompt_energy_motion_rule_present():
    lowered = INSTRUCTION.lower()
    assert "more motion with worse" in lowered
    assert "more energy with better" in lowered


def test_prompt_case_b_evidence_not_treated_as_absolute_score():
    lowered = INSTRUCTION.lower()
    assert "never a pre-computed" in lowered
    assert "higher event count does not by itself" in lowered
