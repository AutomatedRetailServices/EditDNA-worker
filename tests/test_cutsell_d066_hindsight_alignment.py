"""D-065/D-066: claim-vs-claim hindsight equivalence -- claim_coverage_best_take.py.

Generic (English + Spanish) fixtures only -- no Video00-specific fact,
disease, or product name (the D-064 generic-chain fixture reproduces the
GENERAL pattern D-064 traced, never the literal clause). Covers the
mandatory adversarial safety suite, the positive paraphrase suite, sales/UGC
regression shapes, and the D-064/D-063 auto-resolve chain end to end.
"""
from cutsell_worker.claim_coverage_best_take import (
    _find_hindsight_alignment,
    _hindsight_alignment_hard_gates_pass,
    apply_claim_coverage_best_take,
)
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_claims import extract_claims


def clip(clip_id, start, end, text, *, selected=True, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def draft(selected, take_judge_groups):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=(),
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


class _AlwaysConfirmArbiter:
    """An arbiter that would confirm ANY claim-vs-claim pairing -- used to
    prove the deterministic hard gates (not the arbiter) are what actually
    keep a dangerous case from merging."""

    def __init__(self):
        self.calls = []

    def claim_covered(self, claim_text, winning_realization_text):
        self.calls.append((claim_text, winning_realization_text))
        return True, 0.99, "always confirms"


class _AlwaysDeclineArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        return False, 0.4, "never confirms"


def _claims_for(text):
    return extract_claims("x", text)


# --- Direct hard-gate unit tests -----------------------------------------

def test_hard_gate_rejects_diagnosis_candidate():
    negation = _claims_for(
        "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos."
    )[0]
    diagnosis = _claims_for("La biopsia confirmó que era un cáncer papilar de tiroides.")[0]
    assert diagnosis.claim_type == "DIAGNOSIS_IDENTIFICATION"
    assert _hindsight_alignment_hard_gates_pass(negation, diagnosis) is False


def test_hard_gate_rejects_number_bearing_candidate():
    negation = _claims_for(
        "I did not think it was serious, but now I realize it was."
    )[0]
    numeric = _claims_for("It happened over about 10 days.")[0]
    assert _hindsight_alignment_hard_gates_pass(negation, numeric) is False


def test_hard_gate_rejects_correction_candidate():
    negation = _claims_for(
        "I did not think it was serious, but now I realize it was."
    )[0]
    correction = _claims_for("Actually, it was fine after all.")
    correction_claim = next((c for c in correction if c.claim_type == "CORRECTION"), None)
    if correction_claim is not None:
        assert _hindsight_alignment_hard_gates_pass(negation, correction_claim) is False


def test_hard_gate_accepts_plain_reflective_candidate():
    negation = _claims_for(
        "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos."
    )[0]
    reflective = _claims_for(
        "Síntomas que tuve según yo era sintomática."
    )[0]
    assert reflective.claim_type in ("ACTION_EVENT", "STATE_RESULT")
    assert _hindsight_alignment_hard_gates_pass(negation, reflective) is True


def test_arbiter_never_reaches_protected_candidate_even_if_it_would_confirm():
    # The single most important safety proof (D-066 Section 14's mandatory
    # QA question): an arbiter that would confirm ANYTHING must never even
    # be asked about a protected (diagnosis) candidate.
    negation = _claims_for(
        "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos."
    )[0]
    all_claims = (
        _claims_for("La biopsia confirmó que era un cáncer papilar de tiroides.")
        + _claims_for("Síntomas que tuve según yo era sintomática.")
    )
    arbiter = _AlwaysConfirmArbiter()
    aligned = _find_hindsight_alignment(negation, all_claims, claim_equivalence_arbiter=arbiter)
    # The diagnosis claim must never be the alignment target, and the
    # arbiter must never have been asked about it.
    assert aligned is not None
    assert aligned.claim_type != "DIAGNOSIS_IDENTIFICATION"
    assert all("biopsia" not in call[1].casefold() for call in arbiter.calls)


# --- D-064/D-063 generic auto-resolve chain, end to end -------------------

_A_TEXT = (
    "La biopsia confirmó que era un cáncer papilar de tiroides. "
    "Síntomas que tuve según yo era sintomática pero si hubo indicios ahora mirándose atrás."
)
_B_TEXT = "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos."


def test_d064_generic_chain_auto_resolves_with_confirming_arbiter():
    a = clip("A", 0.0, 5.0, _A_TEXT)
    b = clip("B", 5.0, 10.0, _B_TEXT)
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysConfirmArbiter())

    assert [c.clip_id for c in out.selected] == ["A"]
    assert [c.clip_id for c in out.discarded] == ["B"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["dominance_resolutions"][0]["winner_clip_id"] == "A"
    alignments = diag["hindsight_alignments"]
    assert len(alignments) == 1
    assert alignments[0]["coverage_unit_relation"] == "merged"
    assert alignments[0]["negation_role"] == "CONTRASTIVE_HINDSIGHT_NEGATION"


def test_d064_generic_chain_stays_ambiguous_without_arbiter_confirmation():
    a = clip("A", 0.0, 5.0, _A_TEXT)
    b = clip("B", 5.0, 10.0, _B_TEXT)
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysDeclineArbiter())

    assert sorted(c.clip_id for c in out.selected) == ["A", "B"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["dominance_resolutions"] == []
    assert diag["hindsight_alignments"][0]["coverage_unit_relation"] == "unmerged"


def test_d064_generic_chain_stays_ambiguous_with_no_arbiter_at_all():
    a = clip("A", 0.0, 5.0, _A_TEXT)
    b = clip("B", 5.0, 10.0, _B_TEXT)
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=None)

    assert sorted(c.clip_id for c in out.selected) == ["A", "B"]


# --- Sales/UGC: the mandatory "never merge" factual-negation counter-example --

def test_bloating_factual_negation_never_merges_even_with_confirming_arbiter():
    # "It reduced bloating." vs "It did not reduce bloating." -- a direct
    # factual contradiction, not a hindsight paraphrase. Must never resolve
    # to either side auto-winning via this mechanism, even with an arbiter
    # that would confirm anything: the negation clause never even becomes
    # CONTRASTIVE_HINDSIGHT_NEGATION-eligible (no belief/perception verb),
    # so no alignment is ever attempted, and any_pair_contradicts's own
    # safety gate blocks dominance regardless.
    a = clip("A", 0.0, 5.0, "It reduced bloating within a week.")
    b = clip("B", 5.0, 10.0, "It did not reduce bloating within a week.")
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysConfirmArbiter())

    assert sorted(c.clip_id for c in out.selected) == ["A", "B"]
    diag = out.diagnostics.get("claim_coverage_best_take") or {}
    assert diag.get("dominance_resolutions", []) == []


def test_diagnosis_substitution_never_merges_even_with_confirming_arbiter():
    # The D-061 QA_ENGINE-flagged shape (diagnosis substitution) -- must
    # stay unaffected and unmerged by D-066.
    a = clip("A", 0.0, 5.0, "The test confirmed it was gastritis.")
    b = clip("B", 5.0, 10.0, "The test confirmed it was an ulcer.")
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.85)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysConfirmArbiter())
    diag = out.diagnostics.get("claim_coverage_best_take") or {}
    assert diag.get("dominance_resolutions", []) == []


# --- Sales/UGC positive shapes (beauty/wellness/consumer/storytelling) ----

def test_beauty_shape_auto_resolves_with_confirming_arbiter():
    a = clip(
        "A", 0.0, 5.0,
        "The dermatologist confirmed it was a mild reaction. "
        "I did not notice any difference in my skin at first, but after two weeks it looked clearer.",
    )
    b = clip(
        "B", 5.0, 10.0,
        "I did not notice any difference in my skin at first, but after two weeks it looked clearer.",
    )
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysConfirmArbiter())
    assert [c.clip_id for c in out.selected] == ["A"]


def test_consumer_product_shape_auto_resolves_with_confirming_arbiter():
    a = clip(
        "A", 0.0, 5.0,
        "The store confirmed it was the 1200-watt model. "
        "I did not think the blender was powerful enough, but it crushed the ice easily.",
    )
    b = clip(
        "B", 5.0, 10.0,
        "I did not think the blender was powerful enough, but it crushed the ice easily.",
    )
    d = draft((a, b), [{"group_id": "g1", "ranked": [ranked_row("A", 0.9), ranked_row("B", 0.95)]}])

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysConfirmArbiter())
    assert [c.clip_id for c in out.selected] == ["A"]
