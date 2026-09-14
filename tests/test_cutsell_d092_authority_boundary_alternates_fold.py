"""D-092 -- KEEP/DISCARD normalization at the authority boundary.

D-090's QA_ENGINE acceptance review (docs/CUTSELL_DECISIONS.md, D-090 entry)
found one P2: the AUTHORITATIVE second StoryValidator pass used to fold the
resolver's `retained_for_contextual_value` alternates into `discarded`
(D-019: KEEP/DISCARD only); D-090 made that pass validation-only, so those
clips survived as `alternates` through Freeze and into the result, and the
plan's discard provenance no longer listed them.

D-092 folds them exactly once, in `universal_clean_cut.py`, right after
`apply_authoritative_realization_resolution` and BEFORE the D-090 signature
is captured. The resolver application itself is unchanged (it still reports
and returns its alternates bucket); LEGACY/SHADOW are unchanged.

Fixture: generic three-member retry family -- two same-type CRITICAL claims
the resolver composites plus a third, unrelated-content member it neither
covers nor discard-reasons, so it lands in `retained_for_contextual_value`.
Physical stages (ASR, complete-idea recovery, Boundary polish) are stubbed;
Ledger, Resolver, application, StoryValidator, plan, reviewer, repair loop
and the Freeze gate are real.
"""
from __future__ import annotations

import pytest

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.contracts import (
    SCHEMA_VERSION,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
)
from cutsell_worker.final_story_coherence_validation import fold_alternates_into_discarded
from cutsell_worker.post_authority_validation import semantic_selection_signature
from cutsell_worker.realization_resolver import (
    RESOLVED_COMPOSITE,
    apply_authoritative_realization_resolution,
    resolve_realizations_shadow,
)
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME,
    RESOLVER_MODE_AUTHORITATIVE,
    RESOLVER_MODE_LEGACY,
    RESOLVER_MODE_SHADOW,
)
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow

IDEA = "idea_generic_diagnosis_family"
GROUP = "g_generic_diagnosis_family"
TEXT_A = "The doctors confirmed the diagnosis was thyroid cancer."
TEXT_B = "The biopsy confirmed it was a papillary carcinoma."
TEXT_C = "My mother always said to listen to your body."


class AlwaysMergeArbiter:
    def check(self, request):
        return IdeaEquivalenceResult(
            tuple(IdeaEquivalenceDecision(i, True, 0.9, "same") for i in range(len(request.pairs))),
            "fake", "fake", True, True,
        )


def _clip(cid, text, start, end, selected, idea=IDEA):
    return DraftClip(
        clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected, realization_id=f"real_{cid}",
        semantic_idea_id=idea, retry_family_id=idea, complete_idea=True,
    )


def _group(group_id, *ranked):
    return {"group_id": group_id, "ranked": [{"clip_id": c, "score": s, "reason": "x"} for c, s in ranked]}


def _three_member_draft():
    hook = _clip("hook", "Today I want to tell you my story.", 0.0, 2.0, True, "idea_hook")
    a = _clip("A", TEXT_A, 10.0, 15.0, True)
    b = _clip("B", TEXT_B, 16.0, 21.0, False)
    c = _clip("C", TEXT_C, 22.0, 26.0, False)
    cta = _clip("cta", "Take care of yourself and get checked.", 40.0, 45.0, True, "idea_cta")
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(hook, a, cta), alternates=(), discarded=(b, c),
        diagnostics={"take_judge_groups": [
            _group("g_hook", ("hook", 0.9)), _group(GROUP, ("B", 0.72), ("A", 0.66), ("C", 0.5)), _group("g_cta", ("cta", 0.9)),
        ]},
    )


def _run(monkeypatch, draft, *, env):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)
    monkeypatch.setattr(universal, "process_local_sources", lambda request, local_paths, **kw: ProcessingResult(
        schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={},
    ))
    return universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None, semantic_equivalence_arbiter=AlwaysMergeArbiter(),
    )


def test_resolver_application_itself_still_returns_the_retained_alternate():
    """The resolver's own bucket semantics are NOT changed by D-092."""
    draft = _three_member_draft()
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions[IDEA].decision_status == RESOLVED_COMPOSITE
    assert report.idea_resolutions[IDEA].retained_for_contextual_value == ("real_C",)
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    assert [c.clip_id for c in applied.draft.alternates] == ["C"]
    assert [c.clip_id for c in applied.draft.selected] == ["hook", "A", "B", "cta"]


def test_fold_helper_moves_alternates_to_discarded_and_never_touches_selected():
    draft = _three_member_draft()
    ledger = build_semantic_ledger_shadow(draft)
    applied = apply_authoritative_realization_resolution(draft, ledger, resolve_realizations_shadow(ledger))
    before = semantic_selection_signature(applied.draft)
    folded = fold_alternates_into_discarded(applied.draft)
    assert folded.alternates == ()
    assert "C" in [c.clip_id for c in folded.discarded]
    assert next(c for c in folded.discarded if c.clip_id == "C").selected is False
    assert semantic_selection_signature(folded).ordered_digest == before.ordered_digest


def test_authoritative_full_path_folds_retained_alternates_once_at_the_boundary(monkeypatch):
    result = _run(monkeypatch, _three_member_draft(), env=RESOLVER_MODE_AUTHORITATIVE)
    d = result.draft.diagnostics
    fam = next(i for i in d["realization_resolver_authority"]["ideas"] if i.get("semantic_idea_id") == IDEA)
    # The resolver's reasoning stays visible ...
    assert fam["decision_status"] == RESOLVED_COMPOSITE
    assert fam["retained_for_contextual_value"] == ["real_C"]
    # ... but the final draft is KEEP/DISCARD only (D-019).
    assert result.draft.alternates == ()
    assert [c.clip_id for c in result.draft.selected] == ["hook", "A", "B", "cta"]
    assert "C" in [c.clip_id for c in result.draft.discarded]
    # The fold happened at the boundary: the signature captured after
    # authority already sees zero alternates and the post-authority
    # StoryValidator had nothing left to fold on its working copy.
    pav = d["post_authority_validation"]
    assert pav["alternates_folded_at_authority_boundary"] == ["C"]
    assert pav["signature_after_authority"]["alternates_count"] == 0
    assert pav["signature_after_authority"]["discarded_count"] == 1  # B restored into the composite; only C remains discarded
    assert pav["validation_invariant"]["status"] == "PASS" and pav["repair_invariant"]["status"] == "PASS"
    assert pav["integrity_failed"] is False
    fsc = d["final_story_coherence_validation"]
    assert fsc["alternates_folded_into_discard"] is False and fsc["alternates_folded_clip_ids"] == []
    # Provenance: the folded clip is accounted for in the plan's discard
    # provenance again (the D-090 QA P2 gap), and the coverage checks still
    # evaluate its content as lost.
    assert "C" in [row["clip_id"] for row in d["canonical_edit_plan"]["discard_provenance"]]
    assert any(row.get("clip_id") == "C" for row in fsc["lost_semantic_atoms"])
    assert fsc["freeze_blocked"] is True


def test_fold_is_a_no_op_when_the_resolver_retained_nothing(monkeypatch):
    draft = _three_member_draft()
    two_member = DraftTimeline(**{
        **draft.__dict__,
        "discarded": tuple(c for c in draft.discarded if c.clip_id != "C"),
        "diagnostics": {"take_judge_groups": [
            _group("g_hook", ("hook", 0.9)), _group(GROUP, ("B", 0.72), ("A", 0.66)), _group("g_cta", ("cta", 0.9)),
        ]},
    })
    result = _run(monkeypatch, two_member, env=RESOLVER_MODE_AUTHORITATIVE)
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["alternates_folded_at_authority_boundary"] == []
    assert result.draft.alternates == ()
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False


@pytest.mark.parametrize("mode", [RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW])
def test_legacy_and_shadow_keep_their_own_fold_and_carry_no_boundary_record(monkeypatch, mode):
    result = _run(monkeypatch, _three_member_draft(), env=mode)
    assert "post_authority_validation" not in result.draft.diagnostics
    assert result.draft.alternates == ()
    fsc = result.draft.diagnostics["final_story_coherence_validation"]
    assert fsc["alternates_folded_into_discard"] is True
    assert fsc["validation_mode"] == "legacy_resolving"
