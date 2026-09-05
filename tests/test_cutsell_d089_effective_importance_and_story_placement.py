"""D-089 -- EFFECTIVE CLAIM IMPORTANCE SINGLE TRUTH + AUTHORITATIVE STORY PLACEMENT.

Part A: the Ledger/Resolver requirement-group effective importance of the
EXACT canonical claim id outranks StoryValidator's raw re-extraction
importance -- one canonical proposition can never be non-required upstream
and CRITICAL_CLAIM_LOST at the Freeze gate. Genuine critical losses stay
blocking; any missing identity fails closed.

Part B: restorations are placed as deterministic units -- an authoritative
composite is one contiguous block in the resolver's order; a winner
replacement takes the departed winner's actual slot (before its surviving
original successor); nothing is appended after the CTA when an anchor
exists; membership never changes.

No Video00 ids/text/timestamps in production logic; every fixture is generic.
"""
from types import SimpleNamespace as NS

import pytest

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION,
)
from cutsell_worker.final_story_coherence_validation import _lost_critical_claims, apply_final_story_coherence_validation
from cutsell_worker.realization_resolver import (
    EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT,
    EFFECTIVE_IMPORTANCE_INCIDENTAL_SOURCE_EXCLUSIVE,
    PLACEMENT_UNIT_COMPOSITE_BLOCK,
    PLACEMENT_UNIT_WINNER_REPLACEMENT,
    RESOLVED_COMPOSITE,
    SEMANTICALLY_RESOLVED,
    EffectiveClaimImportance,
    _place_restored_clips_at_story_position,
    apply_authoritative_realization_resolution,
    build_effective_claim_importance_diagnostics,
    build_effective_claim_importance_index,
    build_preserved_claim_id_index,
    build_story_placement_diagnostics,
    resolve_intra_idea_semantic_preservation_shadow,
    resolve_realizations_shadow,
)
from cutsell_worker.resolver_mode import ENV_VAR_NAME, RESOLVER_MODE_AUTHORITATIVE, RESOLVER_MODE_LEGACY
from cutsell_worker.semantic_claims import extract_claims
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord, RealizationRecord, SemanticIdeaRecord, SemanticLedger, build_semantic_ledger_shadow,
)


# ---------------------------------------------------------------------------
# Part A fixtures
# ---------------------------------------------------------------------------

IDEA = "idea_generic_family"
GROUP = "g_generic"

# D-088 generic shape: a trailing rhetorical aside makes the WHOLE sentence a
# raw NEGATION/CRITICAL claim, but it is incidental (bare year + temporal
# aside, no substantive marker) and source-exclusive.
LOSER_ASIDE = "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar."
WINNER_FULL = "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero tenía gastritis."


def _clip(cid, text, *, start, end, selected, idea=IDEA):
    return DraftClip(
        clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected,
        realization_id=f"real_{cid}", semantic_idea_id=idea, retry_family_id=idea, complete_idea=True,
    )


def _family_draft(loser_text, winner_text, *, groups=None, extra_selected=()):
    winner = _clip("W", winner_text, start=20.0, end=30.0, selected=True)
    loser = _clip("L", loser_text, start=10.0, end=16.0, selected=False)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner, *extra_selected), alternates=(), discarded=(loser,),
        diagnostics={
            "take_judge_groups": groups if groups is not None else [
                {"group_id": GROUP, "ranked": [
                    {"clip_id": "W", "score": 0.7, "reason": "x"}, {"clip_id": "L", "score": 0.6, "reason": "x"},
                ]},
            ],
            "final_story_coherence_validation": {},
        },
    )


def _index_for(draft):
    ledger = build_semantic_ledger_shadow(draft)
    return ledger, build_effective_claim_importance_index(ledger)


def _loser_canonical_id(text):
    return extract_claims("L", text)[0].canonical_claim_id


def _findings(draft, index):
    findings, confirmations = _lost_critical_claims(draft, canonical_effective_importance_index=index)
    return findings, confirmations


# ---------------------------------------------------------------------------
# Part A tests (Section 14)
# ---------------------------------------------------------------------------

def test_a1_exact_canonical_id_effective_supporting_suppresses_critical_loss():
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    ledger, index = _index_for(draft)
    cid = _loser_canonical_id(LOSER_ASIDE)
    entry = index[cid]
    assert entry.raw_importance == "CRITICAL"
    assert entry.effective_importance == "SUPPORTING"
    assert entry.reason == EFFECTIVE_IMPORTANCE_INCIDENTAL_SOURCE_EXCLUSIVE
    assert entry.source_exclusive is True and entry.semantic_idea_id == IDEA

    # Without the index: the pre-D-089 finding (fail-closed baseline).
    baseline, _ = _lost_critical_claims(draft)
    assert [f["canonical_claim_id"] for f in baseline] == [cid]

    findings, confirmations = _findings(draft, index)
    assert findings == []
    row = next(c for c in confirmations if c["canonical_claim_id"] == cid)
    assert row["critical_loss_suppressed_by"] == "canonical_effective_importance"
    assert row["raw_importance"] == "CRITICAL" and row["effective_importance"] == "SUPPORTING"
    assert row["importance_resolution_reason"] == EFFECTIVE_IMPORTANCE_INCIDENTAL_SOURCE_EXCLUSIVE
    assert row["semantic_idea_id"] == IDEA and row["source_realization_ids"] == ["real_L"]


def test_a2_exact_canonical_id_effective_critical_keeps_finding():
    loser = "No tuve gastritis."
    draft = _family_draft(loser, "Tuve gastritis y me mandaron pastillas.")
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(loser)
    assert index[cid].effective_importance == "CRITICAL"
    findings, _ = _findings(draft, index)
    assert [f["canonical_claim_id"] for f in findings] == [cid]


def test_a3_missing_canonical_id_fails_closed():
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    findings, _ = _findings(draft, {})  # index present but the exact id absent
    assert len(findings) == 1
    findings_none, _ = _lost_critical_claims(draft, canonical_effective_importance_index=None)
    assert len(findings_none) == 1


def test_a4_wrong_canonical_id_never_suppresses():
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(LOSER_ASIDE)
    wrong = {"cclaim_other": index[cid]}  # the right entry filed under a different key
    findings, _ = _findings(draft, wrong)
    assert len(findings) == 1
    # And an entry filed under the right key but naming another canonical claim.
    mismatched = {cid: EffectiveClaimImportance(
        canonical_claim_id="cclaim_other", claim_type="NEGATION", text="x", raw_importance="CRITICAL",
        effective_importance="SUPPORTING", reason="x", semantic_idea_id=IDEA, requirement_group_id="g",
        source_realization_ids=("real_L",), source_exclusive=True,
    )}
    findings, _ = _findings(draft, mismatched)
    assert len(findings) == 1


def test_a5_same_text_different_canonical_proposition_no_suppression():
    """A SUPPORTING entry for a different proposition (different type/tokens,
    hence different canonical id) can never downgrade this claim."""
    loser = "No tuve gastritis."
    draft = _family_draft(loser, "Tuve gastritis y me mandaron pastillas.")
    aside_draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    _l, aside_index = _index_for(aside_draft)
    aside_cid = _loser_canonical_id(LOSER_ASIDE)
    assert aside_cid != _loser_canonical_id(loser)
    findings, _ = _findings(draft, {aside_cid: aside_index[aside_cid]})
    assert len(findings) == 1


def test_a6_source_exclusive_incidental_negation_respected_but_corroborated_stays_critical():
    # Corroborated: a SECOND realization in the idea raises the same proposition
    # -> `_effective_importance` keeps CRITICAL (not a fluke) -> finding remains.
    second = _clip("L2", LOSER_ASIDE, start=1.0, end=5.0, selected=False)
    winner = _clip("W", WINNER_FULL, start=20.0, end=30.0, selected=True)
    loser = _clip("L", LOSER_ASIDE, start=10.0, end=16.0, selected=False)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(loser, second),
        diagnostics={"take_judge_groups": [{"group_id": GROUP, "ranked": [
            {"clip_id": "W", "score": 0.7, "reason": "x"}, {"clip_id": "L", "score": 0.6, "reason": "x"}, {"clip_id": "L2", "score": 0.5, "reason": "x"},
        ]}], "final_story_coherence_validation": {}},
    )
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(LOSER_ASIDE)
    assert index[cid].effective_importance == "CRITICAL"
    assert index[cid].source_exclusive is False
    findings, _ = _findings(draft, index)
    assert [f["canonical_claim_id"] for f in findings] == [cid]


UNRELATED_WINNER = "Después de eso me sentí mucho mejor y seguí con mi rutina."


@pytest.mark.parametrize("loser", [
    "No tuve gastritis.",                                      # genuine factual negation
    "La biopsia no era cáncer.",                               # diagnosis negation
    "El nódulo medía 5 centímetros.",                          # number / measurement
    "La biopsia confirmó que era gastritis.",                  # diagnosis / entity identification
    "El producto no me funcionó.",                             # negation the attribution test below reuses
])
def test_a7_to_a10_genuine_critical_losses_remain_blocking(loser):
    """A genuinely critical proposition (negation, diagnosis, number, entity)
    keeps effective importance CRITICAL in the index and its loss keeps
    blocking exactly as without the index."""
    draft = _family_draft(loser, UNRELATED_WINNER)
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(loser)
    assert index[cid].effective_importance == "CRITICAL"
    assert index[cid].reason == "raw_importance_retained"
    baseline, _ = _lost_critical_claims(draft)
    assert [f["canonical_claim_id"] for f in baseline] == [cid]
    findings, _ = _findings(draft, index)
    assert [f["canonical_claim_id"] for f in findings] == [cid]


@pytest.mark.parametrize("loser,winner", [
    ("El nódulo medía 5 centímetros.", "El nódulo medía 3 centímetros."),                       # number mismatch
    ("La biopsia confirmó que era gastritis.", "La biopsia confirmó que era colitis."),           # entity mismatch
    ("Dejé el medicamento porque me daba náuseas.", "Me daba náuseas porque dejé el medicamento."),  # causal reversal
])
def test_a10b_index_never_downgrades_a_mismatching_proposition(loser, winner):
    """The index can only ever REPORT the Ledger's own requirement-group
    importance: a number/entity/causal proposition is never downgraded
    below its raw importance, so nothing new can suppress its loss --
    whichever independent gate (D-038 coverage, D-056 contradiction,
    D-073 direction-sensitivity) judges that mismatch keeps doing so."""
    draft = _family_draft(loser, winner)
    _ledger, index = _index_for(draft)
    for claim in extract_claims("L", loser):
        entry = index[claim.canonical_claim_id]
        assert entry.effective_importance == entry.raw_importance == claim.importance
    baseline, _ = _lost_critical_claims(draft)
    findings, _ = _findings(draft, index)
    assert [f["canonical_claim_id"] for f in findings] == [f["canonical_claim_id"] for f in baseline]


def test_a11_attribution_asymmetry_remains_blocking():
    loser = "El producto no me funcionó."
    winner = "Algunos clientes dijeron que el producto no les funcionó, pero a mí me funcionó perfecto."
    draft = _family_draft(loser, winner)
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(loser)
    assert index[cid].effective_importance == "CRITICAL"
    baseline, _ = _lost_critical_claims(draft)
    findings, _ = _findings(draft, index)
    assert [f["canonical_claim_id"] for f in findings] == [f["canonical_claim_id"] for f in baseline]
    # The intra-idea proof chain (D-073.1 attribution gate) still refuses to
    # certify the reported-speech candidate as preserving the direct claim.
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    proofs = resolve_intra_idea_semantic_preservation_shadow(ledger, resolver_report=report)
    assert all(not p.verified for p in proofs)


def test_a12_d079_claim_scoped_proof_behavior_unchanged():
    """A verified D-079 proof still suppresses on its own (consumed BEFORE the
    effective-importance lookup), and the proof index shape is untouched."""
    class Proof:
        verified = True
        preserved_claim_ids = ()
        proof_method = "INTRA_IDEA_SEMANTIC_PRESERVATION"
        preserving_id = "real_W"
    loser = "No tuve gastritis."
    draft = _family_draft(loser, "Tuve gastritis.")
    cid = _loser_canonical_id(loser)
    proof = Proof(); proof.preserved_claim_ids = (cid,)
    findings, confirmations = _lost_critical_claims(
        draft, critical_claim_preservation_index={cid: proof}, canonical_effective_importance_index={},
    )
    assert findings == []
    assert confirmations[0]["resolution"] == "semantic_preservation_proof_consumed"
    # Real D-079 machinery still produces an unverified proof for the aside
    # shape (D-088 finding) -- the index is what closes it, not the proof.
    aside = _family_draft(LOSER_ASIDE, WINNER_FULL)
    ledger = build_semantic_ledger_shadow(aside)
    report = resolve_realizations_shadow(ledger)
    proofs = resolve_intra_idea_semantic_preservation_shadow(ledger, resolver_report=report)
    assert all(not p.verified for p in proofs)
    assert build_preserved_claim_id_index(proofs) == {}


def test_a13_entry_from_another_idea_never_downgrades_this_group():
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    _ledger, index = _index_for(draft)
    cid = _loser_canonical_id(LOSER_ASIDE)
    from dataclasses import replace
    foreign = {cid: replace(index[cid], semantic_idea_id="idea_other", source_realization_ids=("real_X",))}
    findings, _ = _findings(draft, foreign)
    assert len(findings) == 1


def test_a14_cross_idea_conflict_fails_closed_to_critical(monkeypatch):
    """Section 6: if the same canonical id ever receives two different
    authoritative answers (one per idea), the index keeps CRITICAL -- never
    the weaker one. The requirement-group call is stubbed to force the
    disagreement, since a real Ledger merges a claim's source realizations
    globally and normally cannot produce one."""
    import cutsell_worker.realization_resolver as rr
    ledger = SemanticLedger()
    claim = extract_claims("L", LOSER_ASIDE)[0]
    def realization(rid, idea, state, claims):
        return RealizationRecord(
            realization_id=rid, semantic_idea_id=idea, retry_family_id=idea, source_span_ids=(), attempt_id=None,
            clip_ids=(rid,), text=LOSER_ASIDE if claims else WINNER_FULL, start=0.0, end=1.0, delivery_score=None,
            state=state, discard_reason=None, replacement_realization_id=None, claim_ids=tuple(claims),
            render_fragment_ids=(), complete_idea=True,
        )
    ledger.register_realization(realization("real_a", "idea_1", "discarded", (claim.canonical_claim_id,)))
    ledger.register_realization(realization("real_w1", "idea_1", "selected", ()))
    ledger.register_realization(realization("real_b", "idea_2", "discarded", (claim.canonical_claim_id,)))
    ledger.register_realization(realization("real_c", "idea_2", "selected", ()))
    ledger.register_claim(CanonicalClaimRecord(
        canonical_claim_id=claim.canonical_claim_id, claim_type=claim.claim_type, content_tokens=claim.content_tokens,
        importance=claim.importance, source_realization_ids=("real_a", "real_b"), covered_by_realization_ids=(),
        coverage_state="unresolved", text=claim.text, negation_role=claim.negation_role,
    ))
    for idea, rids in (("idea_1", ("real_a", "real_w1")), ("idea_2", ("real_b", "real_c"))):
        ledger.register_semantic_idea(SemanticIdeaRecord(
            semantic_idea_id=idea, retry_family_ids=(), realization_ids=rids, canonical_claim_ids=(claim.canonical_claim_id,),
            current_winner_realization_id=None, composite_realization_ids=(), coverage_status="unknown", story_order_position=None,
        ))
    real_groups = rr.build_requirement_groups
    calls = {"n": 0}

    def disagreeing_groups(claims, **kw):
        groups = real_groups(claims, **kw)
        calls["n"] += 1
        forced = "SUPPORTING" if calls["n"] == 1 else "CRITICAL"
        return tuple(rr.RequirementGroup(group_id=g.group_id, claim_type=g.claim_type, importance=forced, member_claim_ids=g.member_claim_ids) for g in groups)
    monkeypatch.setattr(rr, "build_requirement_groups", disagreeing_groups)
    index = build_effective_claim_importance_index(ledger)
    entry = index[claim.canonical_claim_id]
    assert entry.effective_importance == "CRITICAL"
    assert entry.reason == EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT
    diag = build_effective_claim_importance_diagnostics(index)
    assert diag["cross_idea_conflict_count"] == 1


def test_a15_story_validator_pass_threads_index_and_unblocks_freeze_for_aside_shape():
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    _ledger, index = _index_for(draft)
    blocked = apply_final_story_coherence_validation(draft)
    assert blocked.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is True
    unblocked = apply_final_story_coherence_validation(draft, canonical_effective_importance_index=index)
    coherence = unblocked.diagnostics["final_story_coherence_validation"]
    assert coherence["lost_critical_claims"] == []
    assert coherence["claim_coverage_confirmations"][0]["critical_loss_suppressed_by"] == "canonical_effective_importance"
    # Whatever else StoryValidator still flags on this synthetic draft (no
    # semantic arbiter here, so the discarded clip's own vocabulary loss may
    # stay a blocking lost_semantic_atoms row) is independent of this claim:
    # freeze_blocked is now exactly the atoms/contradiction/coverage verdict.
    assert coherence["freeze_blocked"] == (
        bool(coherence["contradiction_findings"]) or bool(coherence["missing_idea_coverage"])
        or any(row.get("blocking", True) for row in coherence["lost_semantic_atoms"])
    )


# ---------------------------------------------------------------------------
# Part B fixtures
# ---------------------------------------------------------------------------

IDEA_AB = "idea_diag_hindsight"
IDEA_ACNE = "idea_acne"
IDEA_RES = "idea_resorcinol"


def _ns(cid, start, idea):
    return NS(clip_id=cid, start=start, realization_id=f"real_{cid}", semantic_idea_id=idea)


def _d088_shape():
    """legacy: hook, A(diag+hindsight), old acne winner, "resorcinol." continuation, later, CTA.
    authoritative: A+B composite, new acne winner replacing the old one."""
    hook = _ns("hook", 0.0, "idea_hook")
    A = _ns("A", 135.0, IDEA_AB)
    old = _ns("old_acne", 166.0, IDEA_ACNE)
    res = _ns("res", 191.1, IDEA_RES)
    later = _ns("later", 200.0, "idea_later")
    cta = _ns("cta", 300.0, "idea_cta")
    legacy = [hook, A, old, res, later, cta]
    B = _ns("B", 150.7, IDEA_AB)
    new = _ns("new_acne", 185.2, IDEA_ACNE)
    kept = [hook, A, res, later, cta]
    ideas = {"real_A": IDEA_AB, "real_B": IDEA_AB, "real_old_acne": IDEA_ACNE, "real_new_acne": IDEA_ACNE, "real_res": IDEA_RES}
    return legacy, kept, B, new, ideas


def _place(kept, restored, legacy, ideas, **kw):
    log = []
    out = _place_restored_clips_at_story_position(list(kept), list(restored), tuple(legacy), ideas_by_realization=ideas, placement_log=log, **kw)
    return [c.clip_id for c in out], log


# ---------------------------------------------------------------------------
# Part B tests (Section 15)
# ---------------------------------------------------------------------------

def test_b1_d088_exact_generic_order_regression():
    legacy, kept, B, new, ideas = _d088_shape()
    for restored in ([B, new], [new, B]):
        ids, log = _place(kept, restored, legacy, ideas, composite_order_by_idea={IDEA_AB: ("real_A", "real_B")})
        assert ids == ["hook", "A", "B", "new_acne", "res", "later", "cta"], restored
        assert ids != ["hook", "A", "new_acne", "B", "res", "later", "cta"]
    kinds = [row["unit_type"] for row in log]
    assert kinds == [PLACEMENT_UNIT_COMPOSITE_BLOCK, PLACEMENT_UNIT_WINNER_REPLACEMENT]


def test_b2_composite_members_remain_contiguous_and_b3_order_preserved():
    legacy, kept, B, new, ideas = _d088_shape()
    # Explicit resolver order B before A must outrank recording time.
    ids, log = _place(kept, [B, new], legacy, ideas, composite_order_by_idea={IDEA_AB: ("real_B", "real_A")})
    assert ids[1:3] == ["B", "A"]
    assert log[0]["authoritative_member_order"] == ["real_B", "real_A"]
    assert log[0]["contiguity_validated"] is True
    # No explicit order: recording time orders the block.
    ids, _ = _place(kept, [B, new], legacy, ideas)
    assert ids[1:3] == ["A", "B"]


def test_b4_winner_replacement_inserted_before_surviving_successor():
    legacy, kept, B, new, ideas = _d088_shape()
    ids, log = _place(kept, [new], legacy, ideas)
    assert ids == ["hook", "A", "new_acne", "res", "later", "cta"]
    row = log[0]
    assert row["unit_type"] == PLACEMENT_UNIT_WINNER_REPLACEMENT
    assert row["departed_clip_ids"] == ["old_acne"] and row["departed_original_index"] == 2
    assert row["successor_anchor"] == "res" and row["predecessor_fallback"] is None
    assert row["placement_reason"] == "before_surviving_original_successor"


def test_b5_predecessor_fallback_when_no_successor_survives():
    hook = _ns("hook", 0.0, "idea_hook"); A = _ns("A", 10.0, "idea_a"); old = _ns("old", 20.0, "idea_x")
    legacy = [hook, A, old]
    new = _ns("new", 25.0, "idea_x")
    ids, log = _place([hook, A], [new], legacy, {"real_old": "idea_x", "real_new": "idea_x"})
    assert ids == ["hook", "A", "new"]
    assert log[0]["predecessor_fallback"] == "A" and log[0]["successor_anchor"] is None
    assert log[0]["placement_reason"] == "after_last_surviving_original_predecessor_fallback"


def test_b6_two_replacements_sharing_one_predecessor_keep_original_relative_order():
    hook = _ns("hook", 0.0, "idea_hook"); x_old = _ns("x_old", 10.0, "idea_x"); y_old = _ns("y_old", 20.0, "idea_y"); z = _ns("z", 30.0, "idea_z")
    legacy = [hook, x_old, y_old, z]
    x_new = _ns("x_new", 12.0, "idea_x"); y_new = _ns("y_new", 22.0, "idea_y")
    ideas = {"real_x_old": "idea_x", "real_x_new": "idea_x", "real_y_old": "idea_y", "real_y_new": "idea_y"}
    for restored in ([x_new, y_new], [y_new, x_new]):
        ids, _ = _place([hook, z], restored, legacy, ideas)
        assert ids == ["hook", "x_new", "y_new", "z"], restored


def test_b7_multiple_composite_blocks_placed_atomically():
    hook = _ns("hook", 0.0, "idea_hook"); A = _ns("A", 10.0, "idea_p"); C = _ns("C", 30.0, "idea_q"); cta = _ns("cta", 60.0, "idea_cta")
    legacy = [hook, A, C, cta]
    B = _ns("B", 15.0, "idea_p"); D = _ns("D", 35.0, "idea_q"); D2 = _ns("D2", 38.0, "idea_q")
    ideas = {"real_A": "idea_p", "real_B": "idea_p", "real_C": "idea_q", "real_D": "idea_q", "real_D2": "idea_q"}
    for restored in ([B, D, D2], [D2, B, D], [D, D2, B]):
        ids, log = _place([hook, A, C, cta], restored, legacy, ideas,
                          composite_order_by_idea={"idea_p": ("real_A", "real_B"), "idea_q": ("real_C", "real_D", "real_D2")})
        assert ids == ["hook", "A", "B", "C", "D", "D2", "cta"], restored
        assert all(r["contiguity_validated"] for r in log)


def test_b8_replacement_before_and_after_a_composite_never_splits_it():
    hook = _ns("hook", 0.0, "idea_hook"); pre_old = _ns("pre_old", 5.0, "idea_pre"); A = _ns("A", 10.0, "idea_p"); post_old = _ns("post_old", 20.0, "idea_post"); cta = _ns("cta", 60.0, "idea_cta")
    legacy = [hook, pre_old, A, post_old, cta]
    B = _ns("B", 15.0, "idea_p"); pre_new = _ns("pre_new", 6.0, "idea_pre"); post_new = _ns("post_new", 21.0, "idea_post")
    ideas = {"real_A": "idea_p", "real_B": "idea_p", "real_pre_old": "idea_pre", "real_pre_new": "idea_pre",
             "real_post_old": "idea_post", "real_post_new": "idea_post"}
    for restored in ([B, pre_new, post_new], [post_new, B, pre_new], [pre_new, post_new, B]):
        ids, _ = _place([hook, A, cta], restored, legacy, ideas, composite_order_by_idea={"idea_p": ("real_A", "real_B")})
        assert ids == ["hook", "pre_new", "A", "B", "post_new", "cta"], restored


def test_b9_physical_continuation_remains_adjacent():
    legacy, kept, B, new, ideas = _d088_shape()
    ids, _ = _place(kept, [B, new], legacy, ideas, composite_order_by_idea={IDEA_AB: ("real_A", "real_B")})
    assert ids.index("res") == ids.index("new_acne") + 1


def test_b10_iteration_permutation_yields_same_sequence():
    import itertools
    legacy, kept, B, new, ideas = _d088_shape()
    extra = _ns("lone", 250.0, None)
    outputs = set()
    for perm in itertools.permutations([B, new, extra]):
        ids, _ = _place(kept, list(perm), legacy, ideas, composite_order_by_idea={IDEA_AB: ("real_A", "real_B")})
        outputs.add(tuple(ids))
    assert outputs == {("hook", "A", "B", "new_acne", "res", "later", "cta", "lone")}


def test_b11_no_semantic_membership_change():
    legacy, kept, B, new, ideas = _d088_shape()
    ids, _ = _place(kept, [B, new], legacy, ideas, composite_order_by_idea={IDEA_AB: ("real_A", "real_B")})
    assert sorted(ids) == sorted([c.clip_id for c in kept] + ["B", "new_acne"])
    assert len(ids) == len(set(ids))


def test_b12_end_to_end_apply_places_composite_and_replacement_and_logs_units():
    """Through the real Ledger -> resolver -> application: A+B composite and a
    winner replacement in one draft."""
    diag = CanonicalClaimRecord(canonical_claim_id="cclaim_diag", claim_type="STATE_RESULT", content_tokens=frozenset({"scan", "confirmed", "nodule"}), importance="CRITICAL", source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved")
    hind = CanonicalClaimRecord(canonical_claim_id="cclaim_hind", claim_type="NEGATION", content_tokens=frozenset({"symptoms", "suspicious", "analyze"}), importance="CRITICAL", source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved")
    acne = CanonicalClaimRecord(canonical_claim_id="cclaim_acne", claim_type="STATE_RESULT", content_tokens=frozenset({"acne", "back", "resorcinol"}), importance="CRITICAL", source_realization_ids=(), covered_by_realization_ids=(), coverage_state="unresolved")

    def rec(rid, idea, claims, state, text, start, end, complete=True):
        return RealizationRecord(realization_id=rid, semantic_idea_id=idea, retry_family_id=idea, source_span_ids=(), attempt_id=None, clip_ids=(rid.replace("real_", ""),), text=text, start=start, end=end, delivery_score=None, state=state, discard_reason=None, replacement_realization_id=None, claim_ids=claims, render_fragment_ids=(), complete_idea=complete)
    ledger = SemanticLedger()
    for r in (
        rec("real_A", IDEA_AB, ("cclaim_diag",), "selected", "The scan confirmed a nodule. Looking back there were signs.", 135.0, 149.0),
        rec("real_B", IDEA_AB, ("cclaim_hind",), "discarded", "Symptoms that did not seem suspicious but now I analyze them they were.", 150.7, 158.0),
        rec("real_old_acne", IDEA_ACNE, (), "selected", "Seasonal back acne that I that I treated with resor", 166.0, 182.0, complete=False),
        rec("real_new_acne", IDEA_ACNE, ("cclaim_acne",), "discarded", "Seasonal back acne that I treated with resorcinol", 185.2, 189.8),
    ):
        ledger.register_realization(r)
    for c in (diag, hind, acne):
        ledger.register_claim(c)
    for idea, rids, claims in ((IDEA_AB, ("real_A", "real_B"), ("cclaim_diag", "cclaim_hind")), (IDEA_ACNE, ("real_old_acne", "real_new_acne"), ("cclaim_acne",))):
        ledger.register_semantic_idea(SemanticIdeaRecord(semantic_idea_id=idea, retry_family_ids=(), realization_ids=rids, canonical_claim_ids=claims, current_winner_realization_id=None, composite_realization_ids=(), coverage_status="unknown", story_order_position=None))
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions[IDEA_AB].decision_status == RESOLVED_COMPOSITE
    assert report.idea_resolutions[IDEA_ACNE].winner_realization_id == "real_new_acne"

    def clip(cid, text, start, end, sel, idea):
        return DraftClip(clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end, text=text, caption_text=text, selected=sel, realization_id=f"real_{cid}", semantic_idea_id=idea, retry_family_id=idea, complete_idea=cid != "old_acne")
    hook = clip("hook", "hook", 0.0, 2.0, True, "idea_hook")
    A = clip("A", "The scan confirmed a nodule. Looking back there were signs.", 135.0, 149.0, True, IDEA_AB)
    B = clip("B", "Symptoms that did not seem suspicious but now I analyze them they were.", 150.7, 158.0, False, IDEA_AB)
    old = clip("old_acne", "Seasonal back acne that I that I treated with resor", 166.0, 182.0, True, IDEA_ACNE)
    new = clip("new_acne", "Seasonal back acne that I treated with resorcinol", 185.2, 189.8, False, IDEA_ACNE)
    res = clip("res", "resorcinol.", 191.1, 191.7, True, IDEA_RES)
    cta = clip("cta", "take care of yourself", 300.0, 305.0, True, "idea_cta")
    draft = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING, selected=(hook, A, old, res, cta), alternates=(), discarded=(B, new), diagnostics={})
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    assert applied.status == SEMANTICALLY_RESOLVED
    assert [c.clip_id for c in applied.draft.selected] == ["hook", "A", "B", "new_acne", "res", "cta"]
    units = build_story_placement_diagnostics(applied.story_placement)
    assert units["composite_block_count"] == 1 and units["winner_replacement_count"] == 1
    assert units["all_blocks_contiguous"] is True
    block = next(u for u in units["units"] if u["unit_type"] == PLACEMENT_UNIT_COMPOSITE_BLOCK)
    assert block["authoritative_member_order"] == ["real_A", "real_B"]
    repl = next(u for u in units["units"] if u["unit_type"] == PLACEMENT_UNIT_WINNER_REPLACEMENT)
    assert repl["successor_anchor"] == "res" and repl["departed_clip_ids"] == ["old_acne"]


# ---------------------------------------------------------------------------
# Modes end to end
# ---------------------------------------------------------------------------

def _run(monkeypatch, draft, *, env):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})
    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    return universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)


def test_modes_authoritative_carries_index_and_placement_diagnostics_legacy_does_not(monkeypatch):
    draft = _family_draft(LOSER_ASIDE, WINNER_FULL)
    auth = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    diag = auth.draft.diagnostics
    assert diag["canonical_effective_importance"]["schema_version"] == "cutsell.canonical_effective_importance.v1"
    assert diag["canonical_effective_importance"]["downgraded_count"] >= 1
    assert diag["authoritative_story_placement"]["schema_version"] == "cutsell.authoritative_story_placement.v1"
    assert diag["final_story_coherence_validation"]["lost_critical_claims"] == []
    assert not any(f["kind"] == "CRITICAL_CLAIM_LOST" for f in diag["final_edit_reviewer"]["findings"])
    assert diag["final_story_coherence_validation"]["claim_coverage_confirmations"][0]["critical_loss_suppressed_by"] == "canonical_effective_importance"

    legacy = _run(monkeypatch, draft, env=RESOLVER_MODE_LEGACY)
    assert "canonical_effective_importance" not in legacy.draft.diagnostics
    assert "authoritative_story_placement" not in legacy.draft.diagnostics
    # LEGACY keeps its pre-D-089 fail-closed finding on the same draft.
    assert len(legacy.draft.diagnostics["final_story_coherence_validation"]["lost_critical_claims"]) == 1
