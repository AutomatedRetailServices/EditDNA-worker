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
    loser = _clip("loser", 5.0, 10.0, "a weaker retry of the same idea", selected=False)
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


def _run_with_reasoner(monkeypatch, *, deterministic_best_take_authority_enabled=True):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    return universal.process_universal_clean_cut_sources(
        object(),
        {},
        asr_provider=object(),
        selection_reasoner=_FakeReasonerBothIndependent(),
        deterministic_best_take_authority_enabled=deterministic_best_take_authority_enabled,
    )


def test_deterministic_best_take_authority_runs_after_unified_selection_by_default(monkeypatch):
    result = _run_with_reasoner(monkeypatch)

    # Unified Selection (the fake) put both clips in SELECT; the deterministic
    # override -- sequential, not exclusive with Unified Selection -- corrects
    # the clear-gap family back to one SELECT + one SWAP afterward.
    assert [c.clip_id for c in result.draft.selected] == ["winner"]
    assert [c.clip_id for c in result.draft.alternates] == ["loser"]
    assert "deterministic_best_take_authority" in result.draft.diagnostics
    assert result.stage_status["selection_phase_authority"] == (
        "unified_whole_video_selection_applied+deterministic_best_take_authority"
    )


def test_deterministic_best_take_authority_rollback_flag_restores_pure_unified_selection(monkeypatch):
    result = _run_with_reasoner(monkeypatch, deterministic_best_take_authority_enabled=False)

    # With the rollback flag off, Unified Selection's raw (buggy) verdict
    # stands untouched -- exactly today's pre-Phase-1 production behavior.
    assert sorted(c.clip_id for c in result.draft.selected) == ["loser", "winner"]
    assert "deterministic_best_take_authority" not in result.draft.diagnostics
    assert result.stage_status["selection_phase_authority"] == "unified_whole_video_selection_applied"


def test_deterministic_best_take_authority_never_invoked_in_legacy_non_unified_path(monkeypatch):
    monkeypatch.setattr(universal, "process_local_sources", _fake_process_local_sources)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    called = []
    monkeypatch.setattr(
        universal,
        "apply_deterministic_best_take_authority",
        lambda draft: called.append(True) or draft,
    )

    result = universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None,
    )

    assert not called
    assert result.stage_status["selection_phase_authority"] == "legacy_explicit_final_selection_authority_executed"
