"""D-093 -- INCIDENTAL-OMISSION PERMIT for `_lost_semantic_atoms` (post-authority only).

Origin: D-092 canary run 33969388042 (engine e4cd508). StoryValidator's
vocabulary `content_loss` rule blocked Freeze on a discarded incidental
aside whose only missing atom was CONTEXTUAL, whose canonical effective
importance was SUPPORTING (`incidental_source_exclusive_downgrade`, the very
row D-089 already suppressed at claim level) and which the Resolver itself
had ruled non-required (`retained_for_contextual_value`). Product decision
(D-093): an omission the canonical authority has explicitly classified as
incidental and non-required must not block purely on low vocabulary overlap.

Recovered case: the live aside/winner SHAPE (an incidental temporal aside
carrying a bare year + rhetorical negation, losing to a complete delivery of
the same idea). Generic controls cover every guard. Physical stages are
stubbed in full-path tests; Ledger, Resolver, application, StoryValidator,
plan, reviewer, repair loop and the Freeze gate are real.
"""
from __future__ import annotations

import pytest

import cutsell_worker.final_story_coherence_validation as fscv
import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.canonical_edit_plan import AuthoritativeIdeaDecision, AuthoritativePlanSource
from cutsell_worker.contracts import (
    SCHEMA_VERSION,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
)
from cutsell_worker.final_story_coherence_validation import (
    OMISSION_PERMITTED_BY,
    PERMITTED_INCIDENTAL_OMISSION,
    _lost_semantic_atoms,
    _permit_incidental_omissions,
    apply_final_story_coherence_validation,
)
from cutsell_worker.post_authority_validation import PostAuthorityValidationContext
from cutsell_worker.realization_resolver import EffectiveClaimImportance, RESOLVED_WINNER, SEMANTICALLY_RESOLVED
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME,
    RESOLVER_MODE_AUTHORITATIVE,
    RESOLVER_MODE_LEGACY,
    RESOLVER_MODE_SHADOW,
)
from cutsell_worker.semantic_claims import extract_claims

IDEA = "idea_generic_stomach_story"
GROUP = "g_generic_stomach_story"
WINNER = "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero tenía gastritis."
ASIDE = "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar."
# Hooks: one carries a rhetorical "no" elsewhere in KEEP (the live shape:
# only the bare year was missing); the other does not.
HOOK_WITH_NO = "Hoy no quiero dramatizar, solo contar mi historia."
HOOK_WITHOUT_NO = "Hoy quiero contar mi historia."
CTA = "Cuídate y hazte tus chequeos."


def _clip(cid, text, start, end, selected, idea=IDEA, **extra):
    kwargs = dict(
        clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end, text=text, caption_text=text,
        selected=selected, realization_id=f"real_{cid}", semantic_idea_id=idea, retry_family_id=idea, complete_idea=True,
    )
    kwargs.update(extra)
    return DraftClip(**kwargs)


def _group(group_id, *ranked):
    return {"group_id": group_id, "ranked": [{"clip_id": c, "score": s, "reason": "x"} for c, s in ranked]}


def _draft(loser_text=ASIDE, *, hook_text=HOOK_WITH_NO, loser_extra=None):
    hook = _clip("hook", hook_text, 0.0, 2.0, True, "idea_hook")
    w = _clip("W", WINNER, 20.0, 30.0, True)
    l = _clip("L", loser_text, 10.0, 16.0, False, **(loser_extra or {}))
    cta = _clip("cta", CTA, 40.0, 45.0, True, "idea_cta")
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(hook, w, cta), alternates=(), discarded=(l,),
        diagnostics={"take_judge_groups": [
            _group("g_hook", ("hook", 0.9)), _group(GROUP, ("W", 0.7), ("L", 0.6)), _group("g_cta", ("cta", 0.9)),
        ]},
    )


def _run(monkeypatch, draft, *, env):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)
    monkeypatch.setattr(universal, "process_local_sources", lambda request, local_paths, **kw: ProcessingResult(
        schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={},
    ))
    return universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)


def _fsc(result):
    return result.draft.diagnostics["final_story_coherence_validation"]


def _rows(result):
    return _fsc(result)["lost_semantic_atoms"]


# ---------------------------------------------------------------------------
# RED / GREEN on the recovered case shape (full real path)
# ---------------------------------------------------------------------------

def test_red_legacy_resolving_pass_blocks_the_incidental_aside_on_vocabulary_alone():
    draft = _draft()
    out = apply_final_story_coherence_validation(draft)
    rows = out.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"]
    assert rows and rows[0]["clip_id"] == "L" and rows[0]["blocking"] is True
    assert rows[0]["classification"] == "REAL_CONTENT_LOSS"
    assert [c["importance"] for c in rows[0]["atom_classifications"]] == ["CONTEXTUAL"]
    assert out.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is True


def test_green_post_authority_pass_permits_the_authority_classified_incidental_omission(monkeypatch):
    result = _run(monkeypatch, _draft(), env=RESOLVER_MODE_AUTHORITATIVE)
    d = result.draft.diagnostics
    fam = next(i for i in d["realization_resolver_authority"]["ideas"] if i["semantic_idea_id"] == IDEA)
    assert fam["decision_status"] == RESOLVED_WINNER and fam["winner_realization_id"] == "real_W"
    assert fam["retained_for_contextual_value"] == ["real_L"]
    assert d["post_authority_validation"]["alternates_folded_at_authority_boundary"] == ["L"]  # D-092 still folds
    fsc = _fsc(result)
    assert fsc["validation_mode"] == "post_authority_validation_only"
    row = fsc["lost_semantic_atoms"][0]
    assert row["clip_id"] == "L"
    assert row["blocking"] is False
    assert row["classification"] == PERMITTED_INCIDENTAL_OMISSION
    assert row["omission_permitted_by"] == OMISSION_PERMITTED_BY
    # The row keeps its original evidence: what is missing ...
    assert row["missing_critical_atoms"] == ["2023"]
    assert row["coverage_against_final_keep"] < 0.45 and row["missing_content_token_count"] >= 4
    ev = row["omission_evidence"]
    assert ev["realization_id"] == "real_L" and ev["semantic_idea_id"] == IDEA
    assert ev["resolver_verdict"] == "retained_for_contextual_value"
    assert ev["selected_family_realization_ids"] == ["real_W"]
    # ... what may be omitted and why ...
    assert ev["missing_content_tokens"] and set(ev["omittable_tokens"]) == set(ev["missing_content_tokens"])
    assert len(ev["justified_claims"]) == 1
    jc = ev["justified_claims"][0]
    assert jc["raw_importance"] == "CRITICAL" and jc["effective_importance"] != "CRITICAL"
    assert jc["importance_resolution_reason"] == "incidental_source_exclusive_downgrade"
    assert jc["source_exclusive"] is True and jc["justification"] == "authority_classified_incidental_non_required"
    # ... and it is explicitly NOT a preservation proof.
    assert ev["verified_semantic_preservation"] is False
    assert fsc["permitted_incidental_omission_count"] == 1
    assert fsc["permitted_incidental_omissions"][0]["clip_id"] == "L"
    # D-089 Part A suppression is the SAME truth at claim level.
    assert fsc["lost_critical_claims"] == []
    assert fsc["claim_coverage_confirmations"][0]["critical_loss_suppressed_by"] == "canonical_effective_importance"
    assert fsc["freeze_blocked"] is False
    # Reviewer: warning, not a blocking finding; Freeze proceeds; D-090 invariants intact.
    rev = d["final_edit_reviewer"]
    assert rev["status"] == "PASS" and [w["kind"] for w in rev["warnings"]] == ["UNIQUE_FACT_LOST"]
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
    pav = d["post_authority_validation"]
    assert pav["validation_invariant"]["status"] == "PASS" and pav["repair_invariant"]["status"] == "PASS"
    assert [c.clip_id for c in result.draft.selected] == ["hook", "W", "cta"]


# ---------------------------------------------------------------------------
# Guards that must keep the block (full real path)
# ---------------------------------------------------------------------------

def test_critical_negation_atom_missing_from_keep_keeps_the_block_documented_limit(monkeypatch):
    """When the rhetorical negation token is absent from the whole KEEP the
    atom classifier rates it CRITICAL and the permit never applies (a
    CRITICAL/UNCERTAIN atom loss is never suppressed here). Documented limit:
    atom-level negation classification of a rhetorical aside is not
    reconciled with the claim-level downgrade by D-093."""
    result = _run(monkeypatch, _draft(hook_text=HOOK_WITHOUT_NO), env=RESOLVER_MODE_AUTHORITATIVE)
    row = _rows(result)[0]
    assert row["blocking"] is True and row["classification"] == "REAL_CONTENT_LOSS"
    assert "no" in row["missing_critical_atoms"]
    assert row["omission_permit_denied_reason"] == "critical_or_uncertain_atom_missing"
    assert _fsc(result)["permitted_incidental_omission_count"] == 0
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True


def test_separate_supporting_fact_without_explicit_incidental_justification_keeps_the_block(monkeypatch):
    """Same topic, an additional distinct fact in its own sentence: its claim
    is raw SUPPORTING (never downgraded, so no explicit omission
    justification exists) -> denied, block kept."""
    draft = _draft(ASIDE + " Mi hermana tuvo lo mismo.")
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    row = _rows(result)[0]
    assert row["blocking"] is True
    assert row["omission_permit_denied_reason"].startswith("omission_not_explicitly_justified:")
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True


def test_mixed_clip_with_a_required_dosage_fact_is_never_permitted(monkeypatch):
    """An aside carrying a dosage: the resolver makes it the winner (the
    dosage is CRITICAL and uncovered), so nothing about it is 'retained
    contextual'; whatever is lost stays blocking and no permit is issued."""
    draft = _draft(ASIDE + " Me recetaron omeprazol de 20 miligramos cada mañana.")
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    fam = next(i for i in result.draft.diagnostics["realization_resolver_authority"]["ideas"] if i["semantic_idea_id"] == IDEA)
    assert fam["winner_realization_id"] == "real_L"
    assert _fsc(result)["permitted_incidental_omission_count"] == 0
    assert all(r["blocking"] for r in _rows(result))
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True


def test_missing_identity_keeps_the_block(monkeypatch):
    draft = _draft(loser_extra={"realization_id": None, "semantic_idea_id": None})
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    row = _rows(result)[0]
    assert row["blocking"] is True and row["omission_permit_denied_reason"] == "missing_identity"
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True


def test_merged_clause_limit_is_documented_not_hidden(monkeypatch):
    """LIMIT (documented, not solved by D-093): when the claim extractor folds
    an appended low-information clause into the SAME incidental claim, the
    canonical classifier judges the whole clause incidental and the permit
    follows that single truth. The evidence row lists every omitted token so
    the omission is transparent; the atom-level permit does not invent a
    second classifier to split the clause."""
    draft = _draft(ASIDE[:-1] + ", y mi hermana también.")
    claims = extract_claims("L", draft.discarded[0].text)
    assert len(claims) == 1  # the extractor's segmentation is the boundary
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    row = _rows(result)[0]
    assert row["classification"] == PERMITTED_INCIDENTAL_OMISSION
    assert "hermana" in row["omission_evidence"]["omittable_tokens"]
    assert row["omission_evidence"]["verified_semantic_preservation"] is False


@pytest.mark.parametrize("mode", [RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW])
def test_legacy_and_shadow_never_permit(monkeypatch, mode):
    result = _run(monkeypatch, _draft(), env=mode)
    fsc = _fsc(result)
    assert fsc["validation_mode"] == "legacy_resolving"
    assert "permitted_incidental_omissions" not in fsc
    assert fsc["lost_semantic_atoms"][0]["blocking"] is True
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True


# ---------------------------------------------------------------------------
# Guard matrix at the permit function (forced contexts, generic)
# ---------------------------------------------------------------------------

def _aside_claim():
    return extract_claims("L", ASIDE)[0]


def _entry(claim, *, idea=IDEA, rids=("real_L",), effective="SUPPORTING", reason="incidental_source_exclusive_downgrade",
           source_exclusive=True, canonical_id=None, text=None):
    return EffectiveClaimImportance(
        canonical_claim_id=canonical_id or claim.canonical_claim_id, claim_type=claim.claim_type,
        text=text if text is not None else claim.text, raw_importance=claim.importance,
        effective_importance=effective, reason=reason, semantic_idea_id=idea,
        requirement_group_id=canonical_id or claim.canonical_claim_id,
        source_realization_ids=tuple(rids), source_exclusive=source_exclusive,
    )


def _context(*, retained=("real_L",), resolved=("real_W",), candidates=("real_W", "real_L")):
    decision = AuthoritativeIdeaDecision(
        semantic_idea_id=IDEA, decision_status=RESOLVED_WINNER, winner_realization_id="real_W",
        composite_realization_ids=(), candidate_realization_ids=tuple(candidates),
        covered_canonical_claim_ids=(), missing_critical_claim_ids=(), decision_reason="t",
    )
    return PostAuthorityValidationContext(
        authoritative_status=SEMANTICALLY_RESOLVED,
        plan_source=AuthoritativePlanSource(status=SEMANTICALLY_RESOLVED, decisions={IDEA: decision}),
        source_identity="authsrc_test", decision_count=1,
        retained_contextual_by_idea={IDEA: tuple(retained)},
        resolved_realizations_by_idea={IDEA: tuple(resolved)},
    )


def _permit(draft, context, index):
    rows = _lost_semantic_atoms(draft)
    assert rows and rows[0]["blocking"] is True
    rows, permitted = _permit_incidental_omissions(rows, draft, context=context, canonical_effective_importance_index=index)
    return rows[0], permitted


def test_permit_function_happy_path_generic():
    claim = _aside_claim()
    row, permitted = _permit(_draft(), _context(), {claim.canonical_claim_id: _entry(claim)})
    assert row["blocking"] is False and row["classification"] == PERMITTED_INCIDENTAL_OMISSION
    assert permitted and permitted[0]["clip_id"] == "L"


@pytest.mark.parametrize("case, context_kwargs, index_fn, expected", [
    ("not retained by resolver", dict(retained=()), lambda c: {c.canonical_claim_id: _entry(c)}, "resolver_did_not_retain_as_contextual"),
    ("family winner not selected", dict(resolved=("real_X",)), lambda c: {c.canonical_claim_id: _entry(c)}, "family_winner_not_selected"),
    ("realization not a candidate of the idea", dict(candidates=("real_W",)), lambda c: {c.canonical_claim_id: _entry(c)}, "realization_not_a_candidate_of_idea"),
    ("claim missing from canonical index", {}, lambda c: {}, "claim_not_in_canonical_index:"),
    ("index entry belongs to another idea", {}, lambda c: {c.canonical_claim_id: _entry(c, idea="idea_other")}, "claim_belongs_to_other_idea:"),
    ("index entry not sourced from this realization", {}, lambda c: {c.canonical_claim_id: _entry(c, rids=("real_Z",))}, "claim_not_sourced_from_realization:"),
    ("effective importance still CRITICAL", {}, lambda c: {c.canonical_claim_id: _entry(c, effective="CRITICAL", reason="raw_importance_retained")}, "claim_effective_importance_critical:"),
    ("SUPPORTING without explicit incidental justification", {}, lambda c: {c.canonical_claim_id: _entry(c, reason="raw_importance_retained")}, "omission_not_explicitly_justified:"),
    ("shared claim not carried by a selected realization", {}, lambda c: {c.canonical_claim_id: _entry(c, rids=("real_L", "real_Q"), source_exclusive=False, reason="x")}, "shared_claim_not_carried_by_selected_realization:"),
])
def test_permit_function_denies_and_keeps_the_block(case, context_kwargs, index_fn, expected):
    claim = _aside_claim()
    row, permitted = _permit(_draft(), _context(**context_kwargs), index_fn(claim))
    assert row["blocking"] is True and row["classification"] == "REAL_CONTENT_LOSS", case
    assert row["omission_permit_denied_reason"].startswith(expected), (case, row["omission_permit_denied_reason"])
    assert permitted == []


def test_permit_function_mixed_clip_guard_denies_tokens_outside_justified_claims(monkeypatch):
    """Guard 5: if the extractor's justified claim covers only part of the
    clip's vocabulary, the remaining missing tokens are unclassified
    information and the block is kept."""
    claim = _aside_claim()
    import dataclasses
    narrow = dataclasses.replace(claim, text="en una temporada, en 2023")
    monkeypatch.setattr(fscv, "extract_claims", lambda *a, **k: (narrow,))
    row, permitted = _permit(_draft(), _context(), {claim.canonical_claim_id: _entry(claim)})
    assert row["blocking"] is True
    assert row["omission_permit_denied_reason"].startswith("unclassified_information_in_missing_tokens:")
    assert "preguntar" in row["omission_permit_denied_reason"]
    assert permitted == []


def test_permit_function_requires_extracted_claims(monkeypatch):
    claim = _aside_claim()
    monkeypatch.setattr(fscv, "extract_claims", lambda *a, **k: ())
    row, permitted = _permit(_draft(), _context(), {claim.canonical_claim_id: _entry(claim)})
    assert row["blocking"] is True and row["omission_permit_denied_reason"] == "no_canonical_claims_extracted"


def test_permit_function_never_touches_critical_number_atoms():
    draft = _draft("Tuve problemas de estómago en una temporada y perdí el 15 por ciento de mi peso sin razón.")
    rows = _lost_semantic_atoms(draft)
    assert rows and rows[0]["blocking"] is True
    assert any(c["importance"] != "CONTEXTUAL" for c in rows[0]["atom_classifications"])
    claims = extract_claims("L", draft.discarded[0].text)
    index = {c.canonical_claim_id: _entry(c) for c in claims}
    rows, permitted = _permit_incidental_omissions(rows, draft, context=_context(), canonical_effective_importance_index=index)
    assert rows[0]["blocking"] is True and rows[0]["omission_permit_denied_reason"] == "critical_or_uncertain_atom_missing"
    assert permitted == []


def test_permit_function_leaves_non_blocking_and_non_content_loss_rows_alone():
    claim = _aside_claim()
    draft = _draft()
    rows = _lost_semantic_atoms(draft)
    rows[0]["blocking"] = False
    rows[0]["classification"] = "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION"
    out, permitted = _permit_incidental_omissions(rows, draft, context=_context(), canonical_effective_importance_index={claim.canonical_claim_id: _entry(claim)})
    assert out[0]["classification"] == "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION"
    assert "omission_permit_denied_reason" not in out[0] and permitted == []
