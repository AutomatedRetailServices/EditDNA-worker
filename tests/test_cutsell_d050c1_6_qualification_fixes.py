"""D-050C1.6: shadow resolver qualification fixes (F1-F7), found by the
D-050C1.5 full 54-fixture CleanCutBench sweep. See
docs/CUTSELL_DECISIONS.md D-050C1.5 and D-050C1.6.

Every test proves one of the six phases:
  Phase 1 (F6/F7): the engine's own FINAL resolution shape (single winner /
      composite / unresolved) is computed from ground-truth realization
      state, never a stale decision-event snapshot, and the parity
      comparison never penalizes a shadow resolver that converges where
      the engine deliberately didn't.
  Phase 2 (F1): a bare negation/number riding on an incidental temporal
      aside no longer forces a composite or disqualifies an otherwise-
      complete winner, UNLESS corroborated by a second realization.
  Phase 3 (F2): negation detection survives `_content`'s short-token
      floor ("no", "sin") on both the dedup gate and contradiction
      detection, including across a claim_type mismatch (NEGATION vs a
      differently-typed positive counterpart).
  Phase 4 (F3): a CORRECTION-typed claim only ever supersedes a
      conflicting prior claim when it explicitly names and rejects that
      claim's own specific value -- never merely because a generic
      correction marker is present.
  Phase 5 (F5): composite formation is bounded (2 members), rejects
      temporally overlapping members, and never silently assembles an
      arbitrary N-fragment set just to satisfy coverage.
  Phase 6 (F4): the existing, provider-neutral `ClaimEquivalenceArbiter`
      contract is consulted ONLY for the genuinely ambiguous dedup band,
      bounded, fail-open, and its usage is recorded in diagnostics.
"""
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord,
    DELIVERY_SCORE_WINNER,
    ENGINE_BLOCKED_UNRESOLVED,
    ENGINE_RESOLVED_COMPOSITE,
    ENGINE_RESOLVED_WINNER,
    ENGINE_REVIEW_REQUIRED,
    RealizationRecord,
    SemanticIdeaRecord,
    SemanticLedger,
)
from cutsell_worker.realization_resolver import (
    RESOLVED_COMPOSITE,
    RESOLVED_WINNER,
    REVIEW_REQUIRED,
    _claim_has_negation,
    _claims_dedup_equivalent,
    _correction_explicitly_supersedes,
    _detect_contradiction_signals,
    _effective_importance,
    _find_minimal_composite,
    build_requirement_groups,
    resolve_realizations_shadow,
)


# ---------------------------------------------------------------------------
# Fixture helpers (same convention as test_cutsell_d050c1_realization_resolver.py)
# ---------------------------------------------------------------------------

def _claim(canonical_claim_id, claim_type, tokens, importance="CRITICAL", text="", source_realization_ids=()):
    return CanonicalClaimRecord(
        canonical_claim_id=canonical_claim_id, claim_type=claim_type,
        content_tokens=frozenset(tokens), importance=importance,
        source_realization_ids=source_realization_ids, covered_by_realization_ids=(),
        coverage_state="unresolved", text=text,
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


class _StubArbiter:
    """Minimal `semantic_claims.ClaimEquivalenceArbiter` fake -- returns a
    fixed verdict regardless of input, and counts calls (bounded-call
    proof)."""

    def __init__(self, verdict=True, raise_error=False):
        self.verdict = verdict
        self.raise_error = raise_error
        self.calls = 0

    def claim_covered(self, claim_text, winning_realization_text):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("simulated arbiter failure")
        return self.verdict, 0.9, "stub"


# ---------------------------------------------------------------------------
# Phase 1 (F6/F7): engine resolution finalization + parity comparison
# ---------------------------------------------------------------------------

def test_f6_composite_finalization_not_stale_after_earlier_winner_decision():
    """F6: a `record_winner_decision` recorded BEFORE a later
    `record_composite` call must not leave `current_winner_realization_id`
    stale -- `finalize_idea_engine_resolution` (called by
    `build_semantic_ledger_shadow`, exercised here directly) always
    reflects ground-truth realization state."""
    from cutsell_worker.semantic_ledger import CompositeRecord, DELIVERY_SCORE_WINNER as _DSW

    ledger = SemanticLedger()
    ledger.register_realization(_realization("r_a", semantic_idea_id="idea_1", state="selected"))
    ledger.register_realization(_realization("r_b", semantic_idea_id="idea_1", state="selected"))
    ledger.register_semantic_idea(_idea("idea_1", ["r_a", "r_b"]))
    ledger.record_winner_decision(
        semantic_idea_id="idea_1", realization_id="r_a", stage="take_judge_provider",
        decision_type=_DSW, reason="local_top_score",
    )
    ledger.record_composite(
        CompositeRecord(semantic_idea_id="idea_1", member_realization_ids=("r_a", "r_b"),
                         composite_kind="claim_coverage_composite", reason="test"),
        stage="claim_coverage_best_take",
    )
    ledger.finalize_idea_engine_resolution(
        "idea_1", status=ENGINE_RESOLVED_COMPOSITE, winner_realization_id=None,
        composite_realization_ids=("r_a", "r_b"),
    )
    idea = ledger.ideas()["idea_1"]
    assert idea.engine_resolution_status == ENGINE_RESOLVED_COMPOSITE
    assert idea.current_winner_realization_id is None
    assert set(idea.composite_realization_ids) == {"r_a", "r_b"}


def test_f7_blocked_unresolved_never_penalized_as_regression():
    """F7: an idea with >1 selected realizations and no composite is
    ENGINE_BLOCKED_UNRESOLVED; the shadow resolver reaching a confident
    answer there must never be classified POTENTIAL_REGRESSION."""
    from cutsell_worker.realization_resolver import build_resolver_parity_report

    claim = _claim("c1", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    r_a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c1",))
    r_b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=())
    ledger = _ledger([r_a, r_b], [claim], [_idea("idea_1", ["r_a", "r_b"])])
    ledger.finalize_idea_engine_resolution(
        "idea_1", status=ENGINE_BLOCKED_UNRESOLVED, winner_realization_id=None, composite_realization_ids=(),
    )
    report = resolve_realizations_shadow(ledger)
    entries = build_resolver_parity_report(report, ledger)
    assert len(entries) == 1
    assert entries[0].category != "POTENTIAL_REGRESSION"
    assert entries[0].category == "CONTENT_SAFETY_IMPROVEMENT"


def test_f7_both_unresolved_is_same():
    """F7: engine BLOCKED_UNRESOLVED + shadow REVIEW_REQUIRED (genuinely
    contradictory candidates) is SAME, not a difference."""
    from cutsell_worker.realization_resolver import build_resolver_parity_report

    claim_5 = _claim("c_5", "MEASUREMENT_QUANTITY", {"5%", "hereditario", "carcinoma"}, text="Es un 5%.")
    claim_10 = _claim("c_10", "MEASUREMENT_QUANTITY", {"10%", "hereditario", "carcinoma"}, text="Es un 10%.")
    r_5 = _realization("r_5", semantic_idea_id="idea_1", claim_ids=("c_5",))
    r_10 = _realization("r_10", semantic_idea_id="idea_1", claim_ids=("c_10",))
    ledger = _ledger([r_5, r_10], [claim_5, claim_10], [_idea("idea_1", ["r_5", "r_10"])])
    ledger.finalize_idea_engine_resolution(
        "idea_1", status=ENGINE_BLOCKED_UNRESOLVED, winner_realization_id=None, composite_realization_ids=(),
    )
    report = resolve_realizations_shadow(ledger)
    entries = build_resolver_parity_report(report, ledger)
    assert len(entries) == 1
    assert entries[0].category == "SAME"


# ---------------------------------------------------------------------------
# Phase 2 (F1): incidental/low-information claim downgrade
# ---------------------------------------------------------------------------

def test_f1_bare_year_temporal_aside_does_not_force_composite():
    rich_text = "I had digestion problems and it turned out to be gastritis, nothing severe."
    thin_text = "During one period in 2023 I had digestion problems and it turned out to be gastritis, nothing severe."
    rich_claim = _claim(
        "c_rich", "DIAGNOSIS_IDENTIFICATION",
        {"digestion", "gastritis", "had", "nothing", "out", "problems", "severe", "turned"},
        text=rich_text, source_realization_ids=("r_rich",),
    )
    thin_claim = _claim(
        "c_thin", "DIAGNOSIS_IDENTIFICATION",
        {"2023", "digestion", "during", "gastritis", "had", "nothing", "one", "out", "period", "problems", "severe", "turned"},
        text=thin_text, source_realization_ids=("r_thin",),
    )
    r_rich = _realization("r_rich", semantic_idea_id="idea_1", claim_ids=("c_rich",))
    r_thin = _realization("r_thin", semantic_idea_id="idea_1", claim_ids=("c_thin",))
    ledger = _ledger([r_rich, r_thin], [rich_claim, thin_claim], [_idea("idea_1", ["r_rich", "r_thin"])])
    report = resolve_realizations_shadow(ledger)
    res = report.idea_resolutions["idea_1"]
    assert res.decision_status == RESOLVED_WINNER
    assert res.missing_critical_claim_ids == ()


def test_f1_uncorroborated_incidental_claim_downgraded():
    text = "During one period in 2023 I had digestion problems."
    claim = _claim(
        "c_a", "DIAGNOSIS_IDENTIFICATION", {"2023", "during", "digestion", "had", "one", "period", "problems"},
        text=text, source_realization_ids=("r_a",),
    )
    assert _effective_importance("CRITICAL", [claim]) == "SUPPORTING"


def test_f1_corroborated_incidental_claim_stays_critical():
    """When TWO independent realizations raise the exact same nominally-
    incidental fact, it is no longer treated as a fluke -- stays CRITICAL."""
    text = "During one period in 2023 I had digestion problems."
    claim_a = _claim(
        "c_a", "DIAGNOSIS_IDENTIFICATION", {"2023", "during", "digestion", "had", "one", "period", "problems"},
        text=text, source_realization_ids=("r_a",),
    )
    claim_b = _claim(
        "c_b", "DIAGNOSIS_IDENTIFICATION", {"2023", "during", "digestion", "had", "one", "period", "problems"},
        text=text, source_realization_ids=("r_b",),
    )
    assert _effective_importance("CRITICAL", [claim_a, claim_b]) == "CRITICAL"


# ---------------------------------------------------------------------------
# Phase 3 (F2): negation detection survives the short-token floor
# ---------------------------------------------------------------------------

def test_f2_positive_vs_negative_same_fact_never_dedup():
    positive = _claim("c_pos", "DIAGNOSIS_IDENTIFICATION", {"tiene", "cancer", "tiroides"}, text="Tiene cancer de tiroides.")
    negative = _claim("c_neg", "NEGATION", {"cancer", "tiroides"}, text="No tiene cancer de tiroides.")
    assert not _claims_dedup_equivalent(positive, negative)


def test_f2_spanish_no_survives_short_token_floor():
    assert _claim_has_negation(_claim("c1", "NEGATION", set(), text="No soy la unica en mi familia."))


def test_f2_english_not_survives():
    assert _claim_has_negation(_claim("c1", "NEGATION", set(), text="This is not correct."))


def test_f2_sin_survives():
    assert _claim_has_negation(_claim("c1", "NEGATION", set(), text="Vivo sin problemas de salud."))


def test_f2_contradiction_detector_catches_negation_vs_positive_cross_type():
    """The real D-050C1.5 finding: "no soy la unica" (NEGATION) vs "soy la
    unica" (a differently-typed positive counterpart) must be caught as a
    contradiction even though `classify_claim` gives them different
    claim_types by construction."""
    negative = _claim(
        "c_neg", "NEGATION", {"soy", "unica", "familia", "problema"},
        text="No soy la unica en mi familia con este problema.",
    )
    positive = _claim(
        "c_pos", "UNIQUE_CONCLUSION", {"soy", "unica", "familia", "problema"},
        text="Soy la unica en mi familia con este problema.",
    )
    signals = _detect_contradiction_signals({"r_a": (negative,), "r_b": (positive,)})
    assert len(signals) == 1
    assert signals[0].reason == "negation_polarity_conflict"


def test_f2_double_negative_ambiguous_case_fails_safe():
    """Two claims that both carry negation markers (same polarity) but
    are NOT genuinely equivalent (low content overlap) must never dedup
    just because both are negated -- polarity match is necessary, not
    sufficient."""
    neg_a = _claim("c_a", "NEGATION", {"never", "biopsy", "confirmed"}, text="The biopsy never confirmed anything.")
    neg_b = _claim("c_b", "NEGATION", {"never", "took", "medication"}, text="I never took the medication.")
    assert not _claims_dedup_equivalent(neg_a, neg_b)


# ---------------------------------------------------------------------------
# Phase 4 (F3): CORRECTION marker safety
# ---------------------------------------------------------------------------

def test_f3_explicit_same_proposition_correction_supersedes():
    prior = _claim("c_prior", "MEASUREMENT_QUANTITY", {"5%", "hereditario"}, text="Es un 5% de los casos.")
    correction = _claim(
        "c_fix", "CORRECTION", {"10%", "hereditario", "actually"},
        text="Actually it was 10%, not 5%, of the cases.",
    )
    assert _correction_explicitly_supersedes(correction, prior)
    signals = _detect_contradiction_signals({"r_prior": (prior,), "r_fix": (correction,)})
    assert signals == ()  # explicit correction -- not flagged as an unresolved contradiction


def test_f3_quantity_conflict_without_explicit_correction_blocks():
    claim_5 = _claim("c_5", "MEASUREMENT_QUANTITY", {"5%", "hereditario", "carcinoma"}, text="Solo un 5% de los casos.")
    claim_10 = _claim("c_10", "MEASUREMENT_QUANTITY", {"10%", "hereditario", "carcinoma"}, text="Solo un 10% de los casos.")
    signals = _detect_contradiction_signals({"r_5": (claim_5,), "r_10": (claim_10,)})
    assert len(signals) == 1
    assert signals[0].reason == "quantitative_value_conflict"


def test_f3_ambiguous_correction_marker_with_no_named_value_blocks():
    """"Actually, I checked, and it was 2020" carries a correction marker
    but never names/rejects "2019" -- must NOT be treated as an explicit
    correction, and the conflict must still block."""
    prior = _claim("c_2019", "MEASUREMENT_QUANTITY", {"2019", "event", "happened"}, text="The event happened in 2019.")
    ambiguous = _claim(
        "c_2020", "CORRECTION", {"2020", "actually", "checked"}, text="Actually, I checked, and it was 2020.",
    )
    assert not _correction_explicitly_supersedes(ambiguous, prior)
    signals = _detect_contradiction_signals({"r_2019": (prior,), "r_2020": (ambiguous,)})
    assert len(signals) == 1


def test_f3_correction_of_a_different_entity_does_not_overwrite():
    """A correction that names a value belonging to an UNRELATED claim
    must never be treated as superseding it."""
    unrelated = _claim("c_dose", "MEASUREMENT_QUANTITY", {"5", "pills", "morning"}, text="I take 5 pills every morning.")
    correction = _claim(
        "c_fix", "CORRECTION", {"10%", "actually", "hereditary"},
        text="Actually, it's 10% hereditary, not 5% as I said before.",
    )
    # The correction mentions "5%" (a different proposition -- hereditary
    # rate, not dose) near a negation marker, but `unrelated`'s own value
    # ("5", dose) is a different figure/topic entirely -- never supersedes.
    assert not _correction_explicitly_supersedes(correction, unrelated)


# ---------------------------------------------------------------------------
# Phase 5 (F5): composite completeness / bounded size / order safety
# ---------------------------------------------------------------------------

def test_f5_valid_2piece_complementary_composite_still_works():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    claim_b = _claim("c_b", "STATE_RESULT", {"resultado", "positivo", "temprano"})
    realizations = {
        "r_a": _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=2.0),
        "r_b": _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=4.0),
    }
    groups = build_requirement_groups([claim_a, claim_b])
    critical_ids = frozenset(g.group_id for g in groups if g.importance == "CRITICAL")
    composite = _find_minimal_composite(
        ["r_a", "r_b"], realizations, groups, critical_ids, unsafe_ids=frozenset(), contradiction_pairs=frozenset(),
    )
    assert composite is not None and set(composite) == {"r_a", "r_b"}


def test_f5_invalid_3fragment_assembly_never_silently_composed():
    """The literal D-050C1.5 F5 finding: 3 critical claims split 3 ways
    across 3 candidates must NEVER be silently assembled -- bounded
    composite size (2) makes this structurally impossible, falling to
    REVIEW_REQUIRED instead."""
    claim_a = _claim("c_a", "STATE_RESULT", {"test", "positive", "condition"})
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsy", "confirmed", "concerning"})
    claim_c = _claim("c_c", "UNIQUE_CONCLUSION", {"percent", "people", "reaction"})
    realizations = {
        "r_a": _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=2.0),
        "r_b": _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=4.0),
        "r_c": _realization("r_c", semantic_idea_id="idea_1", claim_ids=("c_c",), start=4.0, end=6.0),
    }
    groups = build_requirement_groups([claim_a, claim_b, claim_c])
    critical_ids = frozenset(g.group_id for g in groups if g.importance == "CRITICAL")
    composite = _find_minimal_composite(
        ["r_a", "r_b", "r_c"], realizations, groups, critical_ids,
        unsafe_ids=frozenset(), contradiction_pairs=frozenset(),
    )
    assert composite is None


def test_f5_temporally_overlapping_members_rejected():
    """Wrong-order / causal-order-inversion proxy: two members whose
    physical windows overlap can never be genuine sequential composite
    pieces, whatever their claim coverage says."""
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    claim_b = _claim("c_b", "STATE_RESULT", {"resultado", "positivo", "temprano"})
    realizations = {
        "r_a": _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=5.0),
        "r_b": _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=7.0),
    }
    groups = build_requirement_groups([claim_a, claim_b])
    critical_ids = frozenset(g.group_id for g in groups if g.importance == "CRITICAL")
    composite = _find_minimal_composite(
        ["r_a", "r_b"], realizations, groups, critical_ids, unsafe_ids=frozenset(), contradiction_pairs=frozenset(),
    )
    assert composite is None


def test_f5_incomplete_sentence_fragments_can_still_compose():
    """`complete_idea=False` (each piece incomplete on its own) does NOT
    exclude a realization from composite membership -- production's own
    real composite formation (claim_coverage_best_take.py) intentionally
    combines complementary incomplete fragments; excluding them here would
    regress that validated behavior (empirically found via the
    D-050C1.5 full sweep's `test_complementary_critical_claims_require_a_
    composite` fixture)."""
    claim_a = _claim("c_a", "STATE_RESULT", {"test", "positive", "condition"})
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsy", "confirmed", "concerning"})
    realizations = {
        "r_a": _realization(
            "r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=2.0, complete_idea=False,
        ),
        "r_b": _realization(
            "r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=4.0, complete_idea=False,
        ),
    }
    groups = build_requirement_groups([claim_a, claim_b])
    critical_ids = frozenset(g.group_id for g in groups if g.importance == "CRITICAL")
    composite = _find_minimal_composite(
        ["r_a", "r_b"], realizations, groups, critical_ids, unsafe_ids=frozenset(), contradiction_pairs=frozenset(),
    )
    assert composite is not None and set(composite) == {"r_a", "r_b"}


def test_f5_redundant_third_fragment_excluded():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"})
    claim_b = _claim("c_b", "STATE_RESULT", {"resultado", "positivo", "temprano"})
    realizations = {
        "r_a": _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",), start=0.0, end=2.0),
        "r_b": _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), start=2.0, end=4.0),
        "r_redundant": _realization("r_redundant", semantic_idea_id="idea_1", claim_ids=("c_a",), start=4.0, end=6.0),
    }
    groups = build_requirement_groups([claim_a, claim_b])
    critical_ids = frozenset(g.group_id for g in groups if g.importance == "CRITICAL")
    composite = _find_minimal_composite(
        ["r_a", "r_b", "r_redundant"], realizations, groups, critical_ids,
        unsafe_ids=frozenset(), contradiction_pairs=frozenset(),
    )
    assert composite is not None and set(composite) == {"r_a", "r_b"}


# ---------------------------------------------------------------------------
# Phase 6 (F4): claim-equivalence arbiter, bounded and fail-open
# ---------------------------------------------------------------------------

def test_f4_deterministic_equivalence_needs_no_arbiter():
    """Clear overlap (>= threshold): no arbiter call at all."""
    arbiter = _StubArbiter(verdict=False)  # would say NOT equivalent if consulted
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico", "hoy"}, text="hoy")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "diagnostico"}, text="hoy")
    assert _claims_dedup_equivalent(claim_a, claim_b, claim_equivalence_arbiter=arbiter)
    assert arbiter.calls == 0


def test_f4_ambiguous_band_consults_arbiter_and_can_confirm_equivalence():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    arbiter = _StubArbiter(verdict=True)
    log: list = []
    result = _claims_dedup_equivalent(claim_a, claim_b, claim_equivalence_arbiter=arbiter, arbiter_log=log)
    assert result is True
    assert arbiter.calls == 1
    assert len(log) == 1 and log[0]["verdict"] is True


def test_f4_arbiter_false_verdict_fails_open_to_distinct():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    arbiter = _StubArbiter(verdict=False)
    assert not _claims_dedup_equivalent(claim_a, claim_b, claim_equivalence_arbiter=arbiter)


def test_f4_no_arbiter_in_ambiguous_band_fails_open_never_silently_collapses():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    assert not _claims_dedup_equivalent(claim_a, claim_b, claim_equivalence_arbiter=None)


def test_f4_arbiter_exception_fails_open():
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    arbiter = _StubArbiter(raise_error=True)
    assert not _claims_dedup_equivalent(claim_a, claim_b, claim_equivalence_arbiter=arbiter)


def test_f4_arbiter_usage_recorded_in_resolver_diagnostics():
    from cutsell_worker.realization_resolver import build_realization_resolver_diagnostics

    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    r_a = _realization("r_a", semantic_idea_id="idea_1", claim_ids=("c_a",))
    r_b = _realization("r_b", semantic_idea_id="idea_1", claim_ids=("c_b",), state="discarded")
    ledger = _ledger([r_a, r_b], [claim_a, claim_b], [_idea("idea_1", ["r_a", "r_b"])])
    arbiter = _StubArbiter(verdict=True)
    report = resolve_realizations_shadow(ledger, claim_equivalence_arbiter=arbiter)
    assert len(report.arbiter_consultations) == 1
    diagnostics = build_realization_resolver_diagnostics(report)
    assert diagnostics["arbiter_consultation_count"] == 1
    assert len(diagnostics["arbiter_consultations"]) == 1


def test_f4_bounded_calls_not_reconsulted_across_group_membership_checks():
    """Once claim_a joins claim_b's group, later claims are only compared
    against the group's representative (`group[0]`) -- not every prior
    member -- keeping arbiter calls bounded and linear, never quadratic
    in a way that re-asks the same settled question repeatedly."""
    claim_a = _claim("c_a", "ENTITY_RELATION", {"biopsia", "confirmo", "algo", "mas"}, text="biopsia confirmo algo mas")
    claim_b = _claim("c_b", "ENTITY_RELATION", {"biopsia", "confirmo", "otra", "cosa"}, text="biopsia confirmo otra cosa")
    claim_c = _claim("c_c", "ENTITY_RELATION", {"biopsia", "confirmo", "tercera", "cosa"}, text="biopsia confirmo tercera cosa")
    arbiter = _StubArbiter(verdict=True)
    log: list = []
    build_requirement_groups([claim_a, claim_b, claim_c], claim_equivalence_arbiter=arbiter, arbiter_log=log)
    assert arbiter.calls <= 3  # bounded -- never one call per (n choose 2) pair
