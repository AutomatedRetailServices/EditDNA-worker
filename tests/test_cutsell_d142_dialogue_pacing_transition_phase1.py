"""D-142 Phase 1 -- Dialogue/Pacing Transition foundation tests.

Offline only: no ffmpeg, no network, no RAW. `boundary_engine_pass`
diagnostics rows are constructed directly (the same shape `boundary_engine_
pass.apply_post_freeze_boundary_pass` itself writes) so the planner's own
logic is exercised deterministically.
"""
from __future__ import annotations

from cutsell_worker.contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    ProcessingResult,
    ProcessingRequest,
    SemanticRole,
    SourceAsset,
    JobState,
)
from cutsell_worker.dialogue_pacing_transition import (
    FALLBACK_OVERLAP_DISABLED,
    FALLBACK_RENDERER_EXTENSION_REQUIRED,
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    PHASE_1_EXECUTABLE_MODES,
    REQUIRES_RENDERER_EXTENSION,
    RENDERER_SUPPORT_MATRIX,
    SAFETY_SAFE,
    SUPPORTED_NOW,
    TIGHT_CUT,
    apply_dialogue_pacing_transition_pass,
    dialogue_pacing_transition_diagnostics,
    plan_dialogue_pacing_transitions,
)


def _clip(clip_id: str, start: float, end: float, *, role=SemanticRole.OTHER, boundary_reason=None) -> DraftClip:
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=f"text for {clip_id}",
        caption_text="",
        semantic_role=role,
        boundary_reason=boundary_reason,
    )


def _audio_edge_row(clip_id: str, action: str, trim_sec: float) -> dict:
    return {"clip_id": clip_id, "actions": [{"action": action, "trim_sec": trim_sec}]}


def _visual_edge_row(clip_id: str, *, trim_side: str, reason: str, old_start=None, new_start=None, old_end=None, new_end=None) -> dict:
    return {
        "clip_id": clip_id,
        "trim_side": trim_side,
        "trim_applied": True,
        "reason": reason,
        "old_start": old_start,
        "new_start": new_start,
        "old_end": old_end,
        "new_end": new_end,
    }


def _boundary_diagnostics(*, audio_edge_rows=(), visual_edge_rows=()) -> dict:
    return {"boundary_engine_pass": {"audio_edge_rows": list(audio_edge_rows), "visual_edge_rows": list(visual_edge_rows)}}


def _result_with(clips, diagnostics=None) -> ProcessingResult:
    draft = DraftTimeline(
        schema_version="cutsell.v1", project_id="p1", strategy=EditStrategy.MIXED,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=dict(diagnostics or {}),
    )
    return ProcessingResult(
        schema_version="cutsell.v1", project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={},
    )


# --- 1. no gap -> HARD_CUT ---------------------------------------------------

def test_no_gap_produces_hard_cut():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    assert len(plans) == 1
    assert plans[0].mode == HARD_CUT
    assert plans[0].gap_removed_duration == 0.0


# --- 2. safe removable gap -> TIGHT_CUT -------------------------------------

def test_boundary_recorded_exit_trim_produces_tight_cut():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(audio_edge_rows=[_audio_edge_row("a", "tighten_audio_exit", 0.35)])
    plans = plan_dialogue_pacing_transitions(clips, diagnostics, dialogue_overlap_enabled=False)
    assert plans[0].mode == TIGHT_CUT
    assert plans[0].gap_removed_duration == 0.35
    assert "tighten_audio_exit" in plans[0].provenance


def test_boundary_recorded_entry_trim_produces_tight_cut():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(audio_edge_rows=[_audio_edge_row("b", "tighten_audio_entry", 0.22)])
    plans = plan_dialogue_pacing_transitions(clips, diagnostics, dialogue_overlap_enabled=False)
    assert plans[0].mode == TIGHT_CUT
    assert plans[0].gap_removed_duration == 0.22


def test_both_edges_trimmed_sums_gap_removed():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(audio_edge_rows=[
        _audio_edge_row("a", "tighten_audio_exit", 0.30),
        _audio_edge_row("b", "tighten_audio_entry", 0.20),
    ])
    plans = plan_dialogue_pacing_transitions(clips, diagnostics, dialogue_overlap_enabled=False)
    assert plans[0].mode == TIGHT_CUT
    assert plans[0].gap_removed_duration == 0.50


# --- 3. overlap disabled prevents overlap modes -----------------------------

def test_overlap_disabled_never_selects_overlap_mode_and_records_reason():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    assert plans[0].mode in PHASE_1_EXECUTABLE_MODES
    assert plans[0].mode not in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
    assert plans[0].fallback_reason == FALLBACK_OVERLAP_DISABLED
    assert plans[0].overlap_duration == 0.0


# --- 4. overlap enabled does not force overlap ------------------------------

def test_overlap_enabled_still_never_selects_overlap_mode_in_phase1():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=True)
    assert plans[0].mode in PHASE_1_EXECUTABLE_MODES
    assert plans[0].mode not in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
    # Never "forced" -- still HARD_CUT/TIGHT_CUT by the SAME evidence rule,
    # only the diagnostic reason for not overlapping changes.
    assert plans[0].fallback_reason == FALLBACK_RENDERER_EXTENSION_REQUIRED
    assert plans[0].overlap_duration == 0.0
    assert plans[0].dialogue_overlap_enabled is True


# --- 5. required DELIVERY blocks tightening ---------------------------------

def test_delivery_overlap_no_trim_row_never_produces_tight_cut():
    # Boundary itself never writes a trim_applied=True row for a DELIVERY-
    # zone event -- simulate the no-op / no-trim outcome and confirm the
    # planner reads it as "nothing to tighten", never inventing a trim.
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(visual_edge_rows=[{
        "clip_id": "a", "trim_side": None, "trim_applied": False,
        "reason": "visual_event_overlaps_delivery_no_trim",
    }])
    plans = plan_dialogue_pacing_transitions(clips, diagnostics, dialogue_overlap_enabled=False)
    assert plans[0].mode == HARD_CUT
    assert plans[0].gap_removed_duration == 0.0


# --- 6. Boundary-only exit debris tightened only inside approved handles ---

def test_visual_exit_edge_trim_row_produces_tight_cut_with_measured_amount():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(visual_edge_rows=[_visual_edge_row(
        "a", trim_side="EXIT", reason="visual_exit_edge_trim", old_end=2.0, new_end=1.7,
    )])
    plans = plan_dialogue_pacing_transitions(clips, diagnostics, dialogue_overlap_enabled=False)
    assert plans[0].mode == TIGHT_CUT
    assert abs(plans[0].gap_removed_duration - 0.30) < 1e-6


# --- 7. cross-proposition transition remains safe ---------------------------

def test_cross_semantic_role_pair_still_never_overlaps():
    clips = [_clip("a", 0.0, 2.0, role=SemanticRole.STORY), _clip("b", 2.0, 4.0, role=SemanticRole.FEATURES)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=True)
    assert plans[0].mode in PHASE_1_EXECUTABLE_MODES


# --- 8. polarity/number adjacency does not overlap --------------------------

def test_adjacent_clips_with_differing_content_never_overlap_even_when_enabled():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=True)
    assert plans[0].overlap_duration == 0.0
    assert plans[0].mode != MICRO_AUDIO_OVERLAP


# --- 9. CTA adjacency does not overlap unsafely -----------------------------

def test_cta_adjacent_pair_never_overlaps():
    clips = [_clip("a", 0.0, 2.0, role=SemanticRole.PROOF), _clip("b", 2.0, 4.0, role=SemanticRole.CTA)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=True)
    assert plans[0].mode != MICRO_AUDIO_OVERLAP
    assert plans[0].overlap_duration == 0.0


# --- 10. unsupported advanced mode fails to HARD_CUT ------------------------

def test_every_mode_the_planner_can_select_is_phase1_executable():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0), _clip("c", 4.0, 6.0)]
    for overlap in (False, True):
        plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=overlap)
        for plan in plans:
            assert plan.mode in PHASE_1_EXECUTABLE_MODES
            assert RENDERER_SUPPORT_MATRIX[plan.mode] == SUPPORTED_NOW


# --- 11-13. membership / order / source ids unchanged -----------------------

def test_planner_never_mutates_or_reorders_selected_clips():
    clips = (_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0), _clip("c", 4.0, 6.0))
    before = tuple(clips)
    plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    assert clips == before
    assert [c.clip_id for c in clips] == ["a", "b", "c"]


def test_apply_pass_never_changes_selected_tuple():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    result = _result_with(clips)
    original_selected = result.draft.selected
    updated = apply_dialogue_pacing_transition_pass(result, dialogue_overlap_enabled=False)
    assert updated.draft.selected == original_selected
    assert [c.clip_id for c in updated.draft.selected] == ["a", "b"]


# --- 14. source spans stay within handles ------------------------------------

def test_plan_fields_never_exceed_clip_source_spans():
    clips = [_clip("a", 1.0, 3.0), _clip("b", 3.0, 5.5)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    plan = plans[0]
    assert 1.0 <= plan.left_audio_end <= 3.0
    assert 3.0 <= plan.right_audio_start <= 5.5
    assert 1.0 <= plan.visual_cut_time <= 3.0


# --- 15-16. D-123 / D-128 unchanged (independence proof) --------------------

def _import_lines(module) -> list[str]:
    with open(module.__file__, "r", encoding="utf-8") as handle:
        return [line for line in handle if line.lstrip().startswith(("import ", "from "))]


def test_module_never_imports_besttake_or_fallback_authority():
    import cutsell_worker.dialogue_pacing_transition as mod
    imports = "\n".join(_import_lines(mod))
    for forbidden in (
        "deterministic_best_take_authority",
        "multimodal_besttake_fallback",
        "multimodal_besttake_arbiter",
        "realization_resolver",
    ):
        assert forbidden not in imports


# --- 17. Boundary output unchanged ------------------------------------------

def test_apply_pass_never_mutates_boundary_diagnostics_keys():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    diagnostics = _boundary_diagnostics(audio_edge_rows=[_audio_edge_row("a", "tighten_audio_exit", 0.4)])
    result = _result_with(clips, diagnostics)
    before_boundary = dict(result.draft.diagnostics["boundary_engine_pass"])
    updated = apply_dialogue_pacing_transition_pass(result, dialogue_overlap_enabled=False)
    assert updated.draft.diagnostics["boundary_engine_pass"] == before_boundary


# --- 18. render baseline unchanged when no pacing action applies -----------

def test_render_plan_and_render_modules_not_imported_by_dialogue_pacing_transition():
    import cutsell_worker.dialogue_pacing_transition as mod
    imports = "\n".join(_import_lines(mod))
    assert "render_plan" not in imports
    assert "from .render import" not in imports


# --- 19. legacy audio_overlap not directly consumed -------------------------

def test_module_never_references_legacy_audio_overlap_field():
    import ast
    import re
    import cutsell_worker.dialogue_pacing_transition as mod
    with open(mod.__file__, "r", encoding="utf-8") as handle:
        text = handle.read()
    tree = ast.parse(text)
    # Strip the module docstring (prose only) -- check the executable code
    # body never references the legacy field name as its own identifier
    # (not merely as a substring of e.g. "micro_audio_overlap_count").
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    code_only = ast.unparse(tree)
    assert re.search(r"(?<![\w])audio_overlap(?![\w])", code_only) is None


def test_pipeline_call_site_passes_canonical_field_not_legacy():
    import cutsell_worker.universal_clean_cut as mod
    with open(mod.__file__, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert (
        "apply_dialogue_pacing_transition_pass(\n"
        "                result, dialogue_overlap_enabled=getattr(request, \"dialogue_overlap_enabled\", False),"
    ) in text


# --- 20. canonical dialogue_overlap_enabled consumed ------------------------

def test_diagnostics_reflect_the_canonical_overlap_flag_value():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    plans_off = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    plans_on = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=True)
    diag_off = dialogue_pacing_transition_diagnostics(plans_off, dialogue_overlap_enabled=False)
    diag_on = dialogue_pacing_transition_diagnostics(plans_on, dialogue_overlap_enabled=True)
    assert diag_off["dialogue_overlap_enabled"] is False
    assert diag_on["dialogue_overlap_enabled"] is True
    assert diag_off["fallback_reasons"] == {FALLBACK_OVERLAP_DISABLED: 1}
    assert diag_on["fallback_reasons"] == {FALLBACK_RENDERER_EXTENSION_REQUIRED: 1}


# --- Additional contract/diagnostics coverage -------------------------------

def test_diagnostics_aggregate_fields_present_and_bounded():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0), _clip("c", 4.0, 6.0)]
    plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=False)
    diag = dialogue_pacing_transition_diagnostics(plans, dialogue_overlap_enabled=False)
    for key in (
        "dialogue_pacing_evaluated", "dialogue_overlap_enabled", "transition_count",
        "mode_counts", "tight_cut_count", "hard_cut_count", "j_cut_count", "l_cut_count",
        "micro_audio_overlap_count", "total_gap_removed_sec", "total_audio_overlap_sec",
        "fallback_count", "fallback_reasons",
    ):
        assert key in diag
    assert diag["transition_count"] == 2
    assert len(diag["transitions"]) == 2


def test_renderer_support_matrix_classifies_all_five_modes_honestly():
    assert RENDERER_SUPPORT_MATRIX[HARD_CUT] == SUPPORTED_NOW
    assert RENDERER_SUPPORT_MATRIX[TIGHT_CUT] == SUPPORTED_NOW
    assert RENDERER_SUPPORT_MATRIX[J_CUT] == REQUIRES_RENDERER_EXTENSION
    assert RENDERER_SUPPORT_MATRIX[L_CUT] == REQUIRES_RENDERER_EXTENSION
    assert RENDERER_SUPPORT_MATRIX[MICRO_AUDIO_OVERLAP] == REQUIRES_RENDERER_EXTENSION


def test_empty_selected_returns_no_transitions():
    plans = plan_dialogue_pacing_transitions([_clip("a", 0.0, 2.0)], {}, dialogue_overlap_enabled=False)
    assert plans == ()


def test_apply_pass_noop_on_missing_or_empty_selected():
    draft = DraftTimeline(
        schema_version="cutsell.v1", project_id="p1", strategy=EditStrategy.MIXED,
        selected=(), alternates=(), discarded=(),
    )
    result = ProcessingResult(
        schema_version="cutsell.v1", project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={},
    )
    updated = apply_dialogue_pacing_transition_pass(result, dialogue_overlap_enabled=False)
    assert updated is result


def test_safety_status_always_safe_in_phase1():
    clips = [_clip("a", 0.0, 2.0), _clip("b", 2.0, 4.0)]
    for overlap in (False, True):
        plans = plan_dialogue_pacing_transitions(clips, {}, dialogue_overlap_enabled=overlap)
        assert all(plan.safety_status == SAFETY_SAFE for plan in plans)
