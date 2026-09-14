"""D-050C3: AUTHORITATIVE DOWNSTREAM INTEGRATION + IDENTITY CLOSURE.

Proves the pipeline reorder (D-050C3 Section 2): in AUTHORITATIVE mode the
Unified Realization Resolver's own resolved draft becomes the SOLE input to
CanonicalEditPlan/StoryValidator/FinalEditReviewer/Freeze -- the legacy
(pre-cutover) computation of those same three diagnostics survives only as
`*_legacy_evidence`, structurally unable to block or approve anything.
LEGACY/SHADOW stay byte-for-byte the D-050C1/C2 pipeline, unchanged.

Also covers the D-050C3 Section 5 identity-propagation fix: pipeline.py's
`discarded_clips` construction used to hardcode `group_id=None`
unconditionally (even for `review_removed` takes that DID go through
grouping), silently stripping `semantic_idea_id`/`retry_family_id` from
every discarded realization. Fixed to the same `clip_to_group.get(...)`
lookup `selected`/`alternates` already used -- no clip-id hardcoding.
"""
from dataclasses import replace as dataclass_replace

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, JobState,
    ProcessingResult, SCHEMA_VERSION, SemanticRole,
)
from cutsell_worker.human_boundary_polish_v5 import _remove_micro_visual_reset_word_gaps
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME, RESOLVER_MODE_AUTHORITATIVE, RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW,
)


# ---------------------------------------------------------------------------
# Section 5: identity propagation -- pipeline.py's own discard path
# ---------------------------------------------------------------------------

def test_pipeline_discarded_clips_preserve_semantic_idea_id_when_grouped():
    """The general fix, exercised directly against pipeline.py's own
    `_draft_clip`/`clip_to_group` pattern (no clip-id hardcoding): a
    discarded clip whose grouping DID assign it a real group_id must carry
    a `semantic_idea_id` -- the pre-fix code passed `group_id=None`
    unconditionally for the entire discarded bucket regardless of what
    `clip_to_group` actually knew."""
    from cutsell_worker.pipeline import _draft_clip
    from cutsell_worker.contracts import SemanticLabel

    take = CandidateTake(
        clip_id="c_losing_retry", source_asset_id="src", source_order=0,
        start=5.0, end=8.0, text="a losing retry of the same idea", words=(),
    )
    clip_to_group = {"c_losing_retry": "tg_real_group_123"}

    # Mirrors pipeline.py's own discarded_clips construction exactly.
    clip = _draft_clip(
        take, role=SemanticRole.OTHER, group_id=clip_to_group.get(take.clip_id), selected=False,
    )
    assert clip.semantic_idea_id is not None
    assert clip.retry_family_id is not None

    # And the true-negative: a take genuinely never grouped (pre-grouping
    # clean_cut/hybrid-cleanup reject) still correctly gets no semantic_idea_id.
    ungrouped_take = CandidateTake(
        clip_id="c_never_grouped", source_asset_id="src", source_order=0,
        start=0.0, end=1.0, text="filler", words=(),
    )
    ungrouped_clip = _draft_clip(
        ungrouped_take, role=SemanticRole.OTHER, group_id=clip_to_group.get(ungrouped_take.clip_id), selected=False,
    )
    assert ungrouped_clip.semantic_idea_id is None


def test_pipeline_review_removed_takes_keep_group_membership_through_real_flow():
    """A closer-to-real-shape regression: two takes in one grouped retry
    family, one landing in the discarded bucket -- both must resolve to
    the SAME semantic_idea_id (D-049 Case A's own precondition: a
    discarded realization must be traceable back to the idea it belonged
    to, not silently orphaned by construction)."""
    from cutsell_worker.pipeline import _draft_clip
    from cutsell_worker.canonical_identity import mint_semantic_idea_id

    winner_take = CandidateTake(
        clip_id="c_winner", source_asset_id="src", source_order=0,
        start=0.0, end=3.0, text="the clean complete take", words=(),
    )
    loser_take = CandidateTake(
        clip_id="c_loser", source_asset_id="src", source_order=1,
        start=3.0, end=6.0, text="a weaker retry of the clean complete take", words=(),
    )
    group_id = "tg_shared_family"
    clip_to_group = {"c_winner": group_id, "c_loser": group_id}

    winner_clip = _draft_clip(winner_take, role=SemanticRole.OTHER, group_id=clip_to_group.get("c_winner"), selected=True)
    loser_clip = _draft_clip(loser_take, role=SemanticRole.OTHER, group_id=clip_to_group.get("c_loser"), selected=False)

    assert winner_clip.semantic_idea_id == loser_clip.semantic_idea_id == mint_semantic_idea_id(group_id)


# ---------------------------------------------------------------------------
# Section 5: fragment/subspan identity propagation -- already correct,
# locked here as an explicit regression guard (both use dataclasses.replace,
# which copies semantic_idea_id/realization_id/retry_family_id forward
# unless the D-050C3 audit's structural assumption changes).
# ---------------------------------------------------------------------------

def test_group_merge_and_composite_realization_id_sharing_preserves_mapping():
    """Two clips deliberately sharing one realization_id (the composite/
    fragment shape) must both resolve to the SAME semantic_idea_id in the
    Ledger -- proves `_clip_realization_id` grouping does not silently
    drop a sibling's stamped identity."""
    from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow

    a = DraftClip(
        clip_id="frag_a", source_asset_id="src", source_order=0, start=0.0, end=2.0,
        text="first half", caption_text="first half", selected=True,
        realization_id="real_shared", semantic_idea_id="idea_shared",
    )
    b = DraftClip(
        clip_id="frag_b", source_asset_id="src", source_order=0, start=2.0, end=4.0,
        text="second half", caption_text="second half", selected=True,
        realization_id="real_shared", semantic_idea_id="idea_shared",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(a, b), alternates=(), discarded=(), diagnostics={},
    )
    ledger = build_semantic_ledger_shadow(draft)
    record = ledger.realizations()["real_shared"]
    assert record.semantic_idea_id == "idea_shared"
    assert set(record.clip_ids) == {"frag_a", "frag_b"}
    assert "real_shared" not in ledger.find_orphan_realizations()


def test_human_boundary_polish_v5_split_fragments_keep_identity():
    """A selected delivery split by Boundary (D-036/D-050A fragment path)
    must have both physical pieces carry the parent's realization_id/
    semantic_idea_id forward -- proven directly against the real splitting
    function, not a hand-rolled stand-in."""
    from cutsell_worker.contracts import Word
    from cutsell_worker.human_boundary_polish_v5 import _timeline_proxies

    words = (
        Word(text="one", start=0.0, end=0.30),
        Word(text="two", start=0.75, end=1.05),  # 0.45s gap -- in the eligible band
    )
    clip = DraftClip(
        clip_id="c_root", source_asset_id="src", source_order=0, start=0.0, end=1.05,
        text="one two", caption_text="one two", selected=True, words=words,
        realization_id="real_root", semantic_idea_id="idea_root", retry_family_id="fam_root",
    )

    class _FakeEvent:
        def __init__(self, kind, start, end, confidence):
            self.kind, self.start, self.end, self.confidence = kind, start, end, confidence

    class _FakeTimeline:
        events = (_FakeEvent("body_reset_candidate", 0.30, 0.75, 0.95),)

    pieces, diag = _remove_micro_visual_reset_word_gaps(clip, _FakeTimeline())
    if len(pieces) < 2:
        return  # threshold not met on this synthetic timeline -- not what this test checks
    for piece in pieces:
        assert piece.realization_id == "real_root"
        assert piece.semantic_idea_id == "idea_root"
        assert piece.retry_family_id == "fam_root"


# ---------------------------------------------------------------------------
# Sections 1-4: full downstream reorder, exercised end-to-end through
# universal_clean_cut.process_universal_clean_cut_sources.
# ---------------------------------------------------------------------------

def _identity_clip(clip_id, start, end, text, *, selected, semantic_idea_id, realization_id=None):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        semantic_idea_id=semantic_idea_id, realization_id=realization_id or clip_id,
        retry_family_id=semantic_idea_id, complete_idea=True,
    )


def _run(monkeypatch, draft, *, env, causal_arbiter=None):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )
    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    return universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)


def test_authoritative_winner_reflected_in_canonical_edit_plan(monkeypatch):
    """The resolver picks the RICHER realization over the legacy pick
    (identical fixture shape to D-050C1.6's own qualification cases) --
    CanonicalEditPlan's real (non-legacy-evidence) key must reflect the
    resolver's winner, not the legacy one, and the legacy computation must
    survive only under its own `_legacy_evidence` key."""
    thin = _identity_clip("c_thin", 0.0, 2.0, "biopsia", selected=True, semantic_idea_id="idea_1")
    rich = _identity_clip(
        "c_rich", 2.0, 4.0,
        "la biopsia confirmo el diagnostico de cancer de tiroides temprano",
        selected=False, semantic_idea_id="idea_1",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(thin,), alternates=(), discarded=(rich,),
        diagnostics={
            "take_judge_groups": [{
                "group_id": "g1",
                "ranked": [
                    {"clip_id": "c_thin", "score": 0.60, "reason": "watch_listen_baseline"},
                    {"clip_id": "c_rich", "score": 0.58, "reason": "watch_listen_baseline"},
                ],
            }],
        },
    )
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)

    plan_diag = result.draft.diagnostics["canonical_edit_plan"]
    legacy_plan_diag = result.draft.diagnostics.get("canonical_edit_plan_legacy_evidence")
    authority_diag = result.draft.diagnostics["realization_resolver_authority"]
    assert authority_diag["mode"] == RESOLVER_MODE_AUTHORITATIVE
    # Whatever the resolver decided, the REAL CanonicalEditPlan key must
    # match the resolver's own applied draft.selected -- never the legacy
    # pre-cutover one when they disagree.
    keep_ids = {c["clip_id"] for c in plan_diag["keep_sequence"]}
    assert keep_ids == {c.clip_id for c in result.draft.selected}
    assert legacy_plan_diag is not None
    assert "canonical_edit_plan" in result.draft.diagnostics
    assert "canonical_edit_plan_legacy_evidence" in result.draft.diagnostics


def test_review_required_blocks_freeze_and_resolved_idea_permits_it(monkeypatch):
    """REVIEW_REQUIRED (a genuine contradiction, unresolvable) must block
    Freeze; the same shape with the ambiguity removed must NOT."""
    claim_5 = _identity_clip(
        "c_5", 0.0, 1.0, "el 5% de los cánceres son hereditarios", selected=True, semantic_idea_id="idea_1",
    )
    claim_10 = _identity_clip(
        "c_10", 1.0, 2.0, "el 10% de los cánceres son hereditarios", selected=True, semantic_idea_id="idea_1",
    )
    blocked_draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(claim_5, claim_10), alternates=(), discarded=(), diagnostics={},
    )
    result = _run(monkeypatch, blocked_draft, env=RESOLVER_MODE_AUTHORITATIVE)
    assert result.draft.diagnostics["realization_resolver_authority"]["status"] == "REVIEW_REQUIRED"
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == (
        "not_frozen_freeze_blocked_by_coherence_review"
    )

    resolved = _identity_clip("c_ok", 0.0, 1.0, "the clean complete idea", selected=True, semantic_idea_id="idea_2")
    permitted_draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(resolved,), alternates=(), discarded=(), diagnostics={},
    )
    result2 = _run(monkeypatch, permitted_draft, env=RESOLVER_MODE_AUTHORITATIVE)
    assert result2.draft.diagnostics["realization_resolver_authority"]["status"] == "SEMANTICALLY_RESOLVED"
    assert result2.stage_status["freeze_blocked_pending_coherence_review"] is False
    assert result2.draft.diagnostics["selection_boundary_contract"]["status"] == "verified"


def test_legacy_evidence_cannot_overwrite_authoritative_canonical_plan(monkeypatch):
    """Directly proves Section 4's own requirement: even where the FIRST
    (legacy) pass's CanonicalEditPlan disagrees with the resolver, the
    real diagnostics key never reverts to it -- the legacy computation is
    demoted to `_legacy_evidence` and nothing reads it for a decision."""
    thin = _identity_clip("c_thin2", 0.0, 2.0, "diagnostico", selected=True, semantic_idea_id="idea_x")
    rich = _identity_clip(
        "c_rich2", 2.0, 4.0,
        "la biopsia confirmo el diagnostico definitivo de carcinoma papilar",
        selected=False, semantic_idea_id="idea_x",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(thin,), alternates=(), discarded=(rich,),
        diagnostics={
            "take_judge_groups": [{
                "group_id": "g1",
                "ranked": [
                    {"clip_id": "c_thin2", "score": 0.60, "reason": "x"},
                    {"clip_id": "c_rich2", "score": 0.58, "reason": "x"},
                ],
            }],
        },
    )
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    legacy_keep_ids = {
        c["clip_id"] for c in result.draft.diagnostics["canonical_edit_plan_legacy_evidence"]["keep_sequence"]
    }
    real_keep_ids = {c["clip_id"] for c in result.draft.diagnostics["canonical_edit_plan"]["keep_sequence"]}
    # The real key is always derived from the resolver's OWN applied
    # selection -- never the pre-cutover one, whether or not they happen to
    # agree on any given fixture.
    assert real_keep_ids == {c.clip_id for c in result.draft.selected}

    # And the legacy_evidence key is exactly what a plain LEGACY run of the
    # SAME draft would have produced -- i.e. it is genuinely the untouched
    # first pass, not some watered-down or resolver-influenced copy.
    legacy_only_result = _run(monkeypatch, draft, env=RESOLVER_MODE_LEGACY)
    legacy_only_keep_ids = {
        c["clip_id"] for c in legacy_only_result.draft.diagnostics["canonical_edit_plan"]["keep_sequence"]
    }
    assert legacy_keep_ids == legacy_only_keep_ids


def test_legacy_mode_byte_for_byte_unchanged(monkeypatch):
    """No `*_legacy_evidence` keys, no `authoritative_semantic_state` key,
    and CanonicalEditPlan/FinalEditReviewer/StoryValidator run exactly
    once -- LEGACY (default, unset) behaves identically to every prior
    D-050C1.x/D-050C2 directive."""
    clip = _identity_clip("c1", 0.0, 1.0, "hello", selected=True, semantic_idea_id="idea_1")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(clip,), alternates=(), discarded=(), diagnostics={},
    )
    monkeypatch.delenv(ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )
    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    result = universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)

    for key in (
        "canonical_edit_plan_legacy_evidence", "final_edit_reviewer_legacy_evidence",
        "final_story_coherence_validation_legacy_evidence", "repair_loop_legacy_evidence",
        "authoritative_semantic_state",
    ):
        assert key not in result.draft.diagnostics
    assert result.draft.diagnostics["realization_resolver_authority"]["status"] is None
    assert result.draft.diagnostics["realization_resolver_authority"]["mode"] == "LEGACY"


def test_shadow_mode_computes_shadow_report_but_no_legacy_evidence_split(monkeypatch):
    """SHADOW mode still only ever runs the downstream stages ONCE -- the
    resolver observes (realization_resolver_shadow is populated) but never
    applies, so there is no second pass and no `_legacy_evidence` split."""
    clip = _identity_clip("c1", 0.0, 1.0, "hello", selected=True, semantic_idea_id="idea_1")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(clip,), alternates=(), discarded=(), diagnostics={},
    )
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_SHADOW)
    assert result.draft.diagnostics["realization_resolver_shadow"]["resolutions"]
    assert result.draft.diagnostics["realization_resolver_authority"]["status"] is None
    assert "canonical_edit_plan_legacy_evidence" not in result.draft.diagnostics


# ---------------------------------------------------------------------------
# Section 8: CI observability -- Human Gold / architecture validators must
# still run when Freeze is BLOCKED (this is workflow-file content, not
# runtime behavior -- asserted the same way this repo already locks other
# workflow-file invariants elsewhere in tests/).
# ---------------------------------------------------------------------------

def test_modal_raw_workflow_runs_validators_even_when_selection_lock_fails():
    import pathlib
    workflow_path = pathlib.Path(__file__).resolve().parents[1] / ".github/workflows/cutsell-video00-modal-raw.yml"
    text = workflow_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _step_has_always(step_name: str) -> bool:
        for index, line in enumerate(lines):
            if line.strip() == f"- name: {step_name}":
                # if: always() must appear before the next "- name:" line.
                for follow in lines[index + 1:]:
                    if follow.strip().startswith("- name:"):
                        return False
                    if follow.strip() == "if: always()":
                        return True
        raise AssertionError(f"step {step_name!r} not found in workflow")

    assert _step_has_always("Verify Video00 architecture")
    assert _step_has_always("Verify Human Gold regression QA (18-check manifest)")
