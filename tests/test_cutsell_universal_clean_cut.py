from dataclasses import replace
from types import SimpleNamespace

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
    SCHEMA_VERSION,
)
from cutsell_worker.unified_selection_reasoner import UnifiedSelectionDecision, UnifiedSelectionPlan


def test_universal_clean_cut_disables_sales_layers_but_allows_bounded_hybrid_cleanup(monkeypatch):
    captured = {}

    def fake_process(request, local_paths, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            schema_version="cutsell.v1",
            project_id="p1",
            state="draft_ready",
            draft=SimpleNamespace(),
            stage_status={"clean_cut": "context_aware_deterministic_complete"},
        )

    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    hybrid_judge = object()

    result = universal.process_universal_clean_cut_sources(
        object(),
        {},
        asr_provider=object(),
        visual_provider=object(),
        take_judge_provider=object(),
        clean_cut_provider=object(),
        take_grouping_provider=object(),
        whole_video_provider=object(),
        editorial_judge=hybrid_judge,
    )

    assert isinstance(captured["semantic_provider"], universal.NoopSemanticProvider)
    assert captured["composer_provider"] is None
    assert captured["draft_review_provider"] is None
    assert captured["editorial_judge"] is hybrid_judge
    assert captured["take_grouping_provider"] is not None
    assert captured["take_judge_provider"] is not None
    assert captured["whole_video_provider"] is not None
    assert captured["visual_provider"] is not None
    assert result.stage_status["brain_mode"] == "universal_clean_cut"
    assert result.stage_status["semantic"] == "not_requested_clean_cut_only"
    assert result.stage_status["composer"] == "not_requested_clean_cut_only"
    assert result.stage_status["draft_review"] == "not_requested_clean_cut_only"


# --- Architecture rebalance Phase 0/1: sequencing with Unified Selection ---


def _clip(clip_id, start, end, text, *, selected):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _pre_unified_draft():
    winner = _clip("winner", 0.0, 5.0, "the clean complete take", selected=True)
    # Shares real topic vocabulary with the winner ("clean complete take") the
    # way an actual reworded retry of the same idea would -- the lost-
    # semantic-atoms coverage check (final_story_coherence_validation.py)
    # only flags a discard whose OWN content is largely absent from the
    # final KEEP text, and a synthetic "a weaker retry of the same idea"
    # fixture text with zero real overlap would false-positive there.
    loser = _clip("loser", 5.0, 10.0, "a weaker retry of the same clean complete take", selected=False)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p1",
        strategy=EditStrategy.STORYTELLING,
        selected=(winner,),
        alternates=(loser,),
        discarded=(),
        diagnostics={
            "take_judge_groups": [{
                "group_id": "g1",
                "ranked": [
                    {"clip_id": "winner", "score": 0.94, "reason": "watch_listen_baseline"},
                    {"clip_id": "loser", "score": 0.50, "reason": "watch_listen_baseline"},
                ],
            }],
        },
    )


class _FakeReasonerBothIndependent:
    """Mimics Unified Selection treating two members of one lexical retry
    family as independent beats and selecting both -- the exact failure mode
    the RAW #122 audit found, and what Phase 1's deterministic override
    exists to correct using evidence Unified Selection itself never saw."""

    def reason(self, draft):
        return UnifiedSelectionPlan(
            decisions=(
                UnifiedSelectionDecision("winner", "select", "independent", 1.0, 0, "independent_story_coverage"),
                UnifiedSelectionDecision("loser", "select", "independent", 1.0, 1, "independent_story_coverage"),
            ),
            provider="fake",
            model="fake-model",
        )


def _fake_process_local_sources(request, local_paths, **kwargs):
    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id="p1",
        state=JobState.DRAFT_READY,
        draft=_pre_unified_draft(),
        stage_status={},
    )


def _run_with_reasoner(
    monkeypatch, *, deterministic_best_take_authority_enabled=True, clean_cut_core_v1_enabled=False,
):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    return universal.process_universal_clean_cut_sources(
        object(),
        {},
        asr_provider=object(),
        selection_reasoner=_FakeReasonerBothIndependent(),
        deterministic_best_take_authority_enabled=deterministic_best_take_authority_enabled,
        clean_cut_core_v1_enabled=clean_cut_core_v1_enabled,
    )


# The following three tests exercise the RETIRED pre-Clean-Cut-Core-V1
# architecture (whole-video Unified Selection reasoner + Phase 0/1 sequencing)
# kept only behind clean_cut_core_v1_enabled=False for rollback/regression
# comparison -- see test_clean_cut_core_v1_* below for the active default.


def test_deterministic_best_take_authority_runs_after_unified_selection_by_default(monkeypatch):
    result = _run_with_reasoner(monkeypatch)

    # Unified Selection (the fake) put both clips in SELECT; the deterministic
    # override -- sequential, not exclusive with Unified Selection -- corrects
    # the clear-gap family back to one SELECT + one SWAP afterward. SWAP is
    # only reachable at all via this retired swap_enabled=True rollback path.
    assert [c.clip_id for c in result.draft.selected] == ["winner"]
    assert [c.clip_id for c in result.draft.alternates] == ["loser"]
    assert "deterministic_best_take_authority" in result.draft.diagnostics
    assert result.stage_status["selection_phase_authority"] == (
        "unified_whole_video_selection_applied+deterministic_best_take_authority"
    )


def test_deterministic_best_take_authority_rollback_flag_restores_pure_unified_selection(monkeypatch):
    result = _run_with_reasoner(monkeypatch, deterministic_best_take_authority_enabled=False)

    # With the rollback flag off, Unified Selection's raw (buggy) verdict
    # stands untouched at the SELECTION level -- exactly today's pre-Phase-1
    # production behavior: both members of one retry family remain selected.
    assert sorted(c.clip_id for c in result.draft.selected) == ["loser", "winner"]
    assert "deterministic_best_take_authority" not in result.draft.diagnostics
    # But FinalEditReviewer (D-024) is a general, bounded pre-Freeze check
    # applied regardless of which upstream selection authority ran -- it
    # catches this exact "raw buggy verdict" shape (two members of one idea
    # both in the final KEEP sequence) as DUPLICATE_IDEA/UNRESOLVED_RETRY and
    # blocks Freeze rather than letting Boundary/Renderer ship it silently.
    assert result.stage_status["selection_phase_authority"] == (
        "unified_whole_video_selection_applied+freeze_blocked_pending_human_review"
    )
    assert result.stage_status["final_edit_reviewer"] == "FAIL"


def test_deterministic_best_take_authority_never_invoked_in_legacy_non_unified_path(monkeypatch):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    called = []
    monkeypatch.setattr(
        universal,
        "apply_deterministic_best_take_authority",
        lambda draft, **kwargs: called.append(True) or draft,
    )

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None, clean_cut_core_v1_enabled=False,
    )

    assert not called
    assert result.stage_status["selection_phase_authority"] == "legacy_explicit_final_selection_authority_executed"


# --- Clean Cut Core V1: idea-first deterministic pipeline, active default ---


def test_clean_cut_core_v1_is_the_default_and_never_invokes_unified_selection_reasoner(monkeypatch):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    reasoner = _FakeReasonerBothIndependent()
    invoked = []
    monkeypatch.setattr(universal, "apply_unified_selection_reasoner", lambda draft, r: invoked.append(True) or draft)

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=reasoner,
    )

    # Even though a selection_reasoner instance was passed, Clean Cut Core V1
    # (the default) never invokes it -- Gemini is a bounded arbiter only.
    assert not invoked
    assert result.stage_status["unified_selection_reasoner"] == "disabled_clean_cut_core_v1"
    assert result.stage_status["semantic"] == "clean_cut_core_v1_idea_first"


def test_clean_cut_core_v1_resolves_clear_family_to_keep_discard_no_swap(monkeypatch):
    # _pre_unified_draft already has a decisive take_judge_groups contest
    # (winner 0.94 vs loser 0.50) with "loser" pre-parked in alternates by the
    # upstream (fake) pipeline -- Clean Cut Core V1 must fold that into
    # DISCARD, never leave or create a SWAP/alternates bucket.
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None,
    )

    assert [c.clip_id for c in result.draft.selected] == ["winner"]
    assert result.draft.alternates == ()
    assert [c.clip_id for c in result.draft.discarded] == ["loser"]
    assert result.stage_status["selection_phase_authority"] == "clean_cut_core_v1_idea_first_keep_discard"
    assert "final_story_coherence_validation" in result.draft.diagnostics


def _contradictory_ambiguous_draft():
    # Thin score gap (0.60 vs 0.58, < CLEAR_WINNER_MINIMUM_GAP) so
    # deterministic_best_take_authority leaves both selected; the two texts
    # disagree via an explicit negation, so coherence validation must flag a
    # contradiction rather than silently letting both stand.
    a = _clip("a", 0.0, 5.0, "No soy la unica con este problema.", selected=True)
    b = _clip("b", 5.0, 10.0, "Soy la unica con este problema.", selected=True)
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(a, b), alternates=(), discarded=(),
        diagnostics={"take_judge_groups": [{
            "group_id": "g1",
            "ranked": [
                {"clip_id": "a", "score": 0.60, "reason": "watch_listen_baseline"},
                {"clip_id": "b", "score": 0.58, "reason": "watch_listen_baseline"},
            ],
        }]},
    )


def test_freeze_blocked_by_contradiction_skips_freeze_and_boundary(monkeypatch):
    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=_contradictory_ambiguous_draft(), stage_status={},
        )

    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    freeze_calls = []
    boundary_calls = []
    monkeypatch.setattr(universal, "freeze_selection_contract", lambda draft: freeze_calls.append(True) or draft)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: boundary_calls.append(True) or result)
    monkeypatch.setattr(universal, "enforce_selection_contract", lambda draft: draft)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None,
    )

    assert not freeze_calls
    assert not boundary_calls
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True
    assert "freeze_blocked_pending_human_review" in result.stage_status["selection_phase_authority"]
    # Both contradictory clips remain visible (unresolved), for human review.
    assert sorted(c.clip_id for c in result.draft.selected) == ["a", "b"]


def test_freeze_not_blocked_when_no_coherence_failure(monkeypatch):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None,
        clean_cut_core_v1_enabled=False,
    )
    # Legacy path never populates final_story_coherence_validation, so the
    # gate must be a no-op (never blocks) for it.
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
