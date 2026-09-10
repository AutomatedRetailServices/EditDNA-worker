"""D-215 -- PACING V2 TRANSITION DECISION FOUNDATION. OFFLINE ONLY.

Proves `pacing_transition_decision.decide_transition` safely distinguishes
HARD_CUT/TIGHT_CUT/J_CUT/L_CUT/MICRO_AUDIO_OVERLAP using only existing
evidence (word timings, Boundary's own diagnostics via the live D-142
`plan_dialogue_pacing_transitions`, D-038's `classify_claim`, D-187's
optional `ProsodicDeliveryEvidence`) -- no ASR rerun, no provider, no new
duration threshold, no live wiring. Generic fixtures only -- no Video00
wording, timestamps, or clip ids.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import (
    FALLBACK_OVERLAP_DISABLED,
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
)
from cutsell_worker.pacing_transition_decision import (
    CONFLICT_CORRECTION_RELATIONSHIP,
    CONFLICT_DOUBLE_SPEECH,
    CONFLICT_MEANING_CRITICAL,
    CONFLICT_RETRY_RELATIONSHIP,
    CONFLICT_WORD_TIMING_MISSING,
    DECISION_CONFLICTED,
    DECISION_SAFE_FALLBACK,
    DECISION_SUPPORTED,
    DECISION_UNKNOWN,
    DOUBLE_SPEECH_NO_OVERLAP_REQUIRED,
    DOUBLE_SPEECH_SAFE_J_CUT,
    DOUBLE_SPEECH_SAFE_L_CUT,
    DOUBLE_SPEECH_SAFE_MICRO_OVERLAP,
    DOUBLE_SPEECH_SAFE_NO_OVERLAP,
    DOUBLE_SPEECH_UNKNOWN,
    GAP_KEEP_PAUSE,
    GAP_OVERLAP,
    GAP_TIGHTEN,
    GAP_UNKNOWN,
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_UNKNOWN,
    decide_transition,
    dialogue_pacing_transition_decision_run_summary,
    sequence_consistency_diagnostics,
)
from cutsell_worker.prosodic_audio_v2 import ProsodicDeliveryEvidence


SOURCE_A = "synthetic_source_a"
SOURCE_B = "synthetic_source_b"


def _words(text: str, start: float, end: float) -> tuple[Word, ...]:
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _clip(clip_id: str, start: float, end: float, text: str, *, words=None, source: str = SOURCE_A, order: int = 0) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=order, start=start, end=end,
        text=text, caption_text=text, words=(words if words is not None else _words(text, start, end)),
        semantic_role=SemanticRole.STORY, selected=True,
    )


def _prosody(*, vocal_continuity_state: str = "UNKNOWN", restart_or_interruption_state: str = "UNKNOWN") -> ProsodicDeliveryEvidence:
    return ProsodicDeliveryEvidence(
        candidate_id="c", source_asset_id=SOURCE_A, source_start=0.0, source_end=1.0,
        analysis_status="EVALUATED",
        speech_duration_sec=1.0, voiced_or_active_speech_duration_sec=1.0,
        speech_rate=2.0, speech_rate_state="MODERATE",
        pause_count=0, pause_total_sec=0.0, pause_structure_state="UNKNOWN",
        hesitation_state="UNKNOWN", restart_or_interruption_state=restart_or_interruption_state,
        vocal_continuity_state=vocal_continuity_state,
        energy_mean=None, energy_variation=None, energy_dynamics_state="UNKNOWN",
        emphasis_dynamics_state="UNKNOWN",
        pitch_analysis_status="PITCH_ANALYSIS_NOT_IMPLEMENTED", pitch_variation_state="UNKNOWN",
        delivery_variation_state="UNKNOWN",
        evidence_confidence="UNKNOWN", missing_evidence=(),
        provenance="prosodic_audio_v2_phase_a",
    )


# --- 1-4: baseline HARD_CUT / TIGHT_CUT / pause geometry --------------------

def test_01_clean_hard_cut_baseline():
    left = _clip("a", 0.0, 5.0, "one two three")
    right = _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False)
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.fallback_reason == FALLBACK_OVERLAP_DISABLED


def test_02_clean_tight_cut_when_boundary_already_trimmed():
    left = _clip("a", 0.0, 5.0, "one two three")
    right = _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    boundary_diag = {"boundary_engine_pass": {"audio_edge_rows": [
        {"clip_id": "a", "actions": [{"action": "tighten_audio_exit", "trim_sec": 0.3}]},
    ], "visual_edge_rows": []}}
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, boundary_diagnostics=boundary_diag)
    assert plan.mode == TIGHT_CUT
    assert plan.pacing_gap_decision == GAP_TIGHTEN


def test_03_meaningful_pause_retained_as_keep_pause():
    # Left's own trailing text is a negation -- meaning-critical -- so even
    # though a real inter-segment pause exists, it must be KEPT, not tightened.
    left = _clip("a", 0.0, 5.0, "no it is not true", words=_words("no it is not true", 0.0, 4.5))
    right = _clip("b", 5.5, 10.0, "four five six", source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False)
    assert plan.pacing_gap_decision == GAP_KEEP_PAUSE


def test_04_non_meaningful_gap_with_no_boundary_trim_still_tightens_baseline():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.5, 10.0, "four five six", source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False)
    assert plan.mode == HARD_CUT
    assert plan.pacing_gap_decision == GAP_TIGHTEN


# --- 5-6: J-cut safe / blocked by outgoing speech ---------------------------

def _safe_jcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def test_05_jcut_safe_when_lead_falls_in_rights_own_leading_silence():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.mode == J_CUT
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.double_speech_status == DOUBLE_SPEECH_SAFE_J_CUT


def test_06_jcut_blocked_by_outgoing_speech_double_speech():
    # A lead large enough to reach back into LEFT's own trailing words.
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.9))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=2.0)
    assert plan.mode == HARD_CUT  # falls back
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert plan.double_speech_status == DOUBLE_SPEECH_NO_OVERLAP_REQUIRED
    assert CONFLICT_DOUBLE_SPEECH in plan.conflict_flags


# --- 7: J-cut blocked by meaning-critical outgoing word ---------------------

def test_07_jcut_blocked_by_meaning_critical_word_at_edge():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "no it is not correct", words=_words("no it is not correct", 5.0, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3)
    assert plan.mode == HARD_CUT
    assert plan.meaning_safety_status == SAFETY_BLOCKED
    assert CONFLICT_MEANING_CRITICAL in plan.conflict_flags


# --- 8-9: L-cut safe / blocked by incoming speech ---------------------------

def _safe_lcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def test_08_lcut_safe_when_tail_falls_in_lefts_own_trailing_silence():
    left, right = _safe_lcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_tail_sec=0.4)
    assert plan.mode == L_CUT
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.double_speech_status == DOUBLE_SPEECH_SAFE_L_CUT


def test_09_lcut_blocked_by_incoming_speech_double_speech():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.0, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_tail_sec=2.0)
    assert plan.mode == HARD_CUT
    assert plan.double_speech_status == DOUBLE_SPEECH_NO_OVERLAP_REQUIRED


# --- 10: L-cut blocked by meaning conflict ----------------------------------

def test_10_lcut_blocked_by_meaning_conflict():
    # The negation clause is packed into the LAST 0.4s of left's own span --
    # exactly the candidate tail window -- so it genuinely falls inside the
    # double-speech window being evaluated.
    left = _clip("a", 0.0, 5.0, "the number is not five", words=_words("the number is not five", 4.6, 5.0))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_tail_sec=0.4)
    assert plan.mode == HARD_CUT
    assert plan.meaning_safety_status == SAFETY_BLOCKED


# --- 11-12: micro-overlap safe / unsafe -------------------------------------

def test_11_micro_overlap_safe_acoustic_edge():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1, candidate_audio_tail_sec=0.1)
    assert plan.mode == MICRO_AUDIO_OVERLAP
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.double_speech_status == DOUBLE_SPEECH_SAFE_MICRO_OVERLAP


def test_12_micro_overlap_unsafe_double_speech():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.9))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.1, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3, candidate_audio_tail_sec=0.3)
    assert plan.mode == HARD_CUT
    assert plan.double_speech_status == DOUBLE_SPEECH_NO_OVERLAP_REQUIRED


# --- 13-15: negation / number / factual-qualifier protection ---------------

def test_13_negation_protected():
    # The negation clause is packed into the FIRST 0.3s of right's own
    # span -- exactly the candidate lead window -- so it genuinely falls
    # inside the double-speech window being evaluated.
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "it did not work", words=_words("it did not work", 5.0, 5.3), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3)
    assert plan.meaning_safety_status == SAFETY_BLOCKED


def test_14_number_protected():
    # classify_claim's MEASUREMENT_QUANTITY marker requires an actual
    # digit token (not a spelled-out number) alongside a unit marker --
    # confirmed directly against cutsell_worker/semantic_claims.py.
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "25 percent complete", words=_words("25 percent complete", 5.0, 5.3), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3)
    assert plan.meaning_safety_status == SAFETY_BLOCKED


def test_15_factual_qualifier_protected():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "the result was negative", words=_words("the result was negative", 5.0, 5.3), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3)
    assert plan.meaning_safety_status == SAFETY_BLOCKED


# --- 16: correction protected -----------------------------------------------

def test_16_correction_relationship_always_falls_back():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1, relationship_hint=RELATIONSHIP_CORRECTION)
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert CONFLICT_CORRECTION_RELATIONSHIP in plan.conflict_flags


# --- 17: continuation can tighten -------------------------------------------

def test_17_continuation_relationship_does_not_block_a_safe_jcut():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2, relationship_hint=RELATIONSHIP_CONTINUATION)
    assert plan.mode == J_CUT
    assert plan.decision_status == DECISION_SUPPORTED


# --- 18: retry ambiguity falls back -----------------------------------------

def test_18_retry_relationship_forces_conflicted_hard_cut():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1, relationship_hint=RELATIONSHIP_RETRY)
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_CONFLICTED
    assert CONFLICT_RETRY_RELATIONSHIP in plan.conflict_flags


# --- 19-20: word timing / prosody missing -----------------------------------

def test_19_word_timing_missing_falls_back_to_unknown():
    left = _clip("a", 0.0, 5.0, "one two three", words=())
    right = _clip("b", 5.0, 10.0, "four five six", words=(), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3)
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_UNKNOWN
    assert plan.word_safety_status == SAFETY_UNKNOWN
    assert CONFLICT_WORD_TIMING_MISSING in plan.conflict_flags


def test_20_prosody_missing_still_produces_a_safe_decision():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2,
                              left_prosody=None, right_prosody=None)
    assert plan.mode == J_CUT
    assert plan.decision_status == DECISION_SUPPORTED
    assert "prosodic_audio_v2" not in plan.provenance


# --- 21-22: prosody supports / blocks ---------------------------------------

def test_21_prosody_continuity_recorded_as_supporting_provenance():
    left, right = _safe_jcut_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2,
        left_prosody=_prosody(vocal_continuity_state="CONTINUOUS"),
    )
    assert plan.mode == J_CUT
    assert "prosodic_audio_v2" in plan.provenance


def test_22_prosody_restart_blocks_an_otherwise_safe_overlap():
    left, right = _safe_jcut_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2,
        right_prosody=_prosody(restart_or_interruption_state="RESTART_DETECTED"),
    )
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert "prosodic_restart_or_discontinuity" in plan.conflict_flags


# --- 23: Boundary edges unchanged -------------------------------------------

def test_23_boundary_edges_never_mutated():
    left, right = _safe_jcut_pair()
    left_start, left_end, right_start, right_end = left.start, left.end, right.start, right.end
    decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert (left.start, left.end, right.start, right.end) == (left_start, left_end, right_start, right_end)


# --- 24: source identity -----------------------------------------------------

def test_24_source_identity_preserved_in_plan():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.left_clip_id == "a" and plan.right_clip_id == "b"


# --- 25-26: same-source / multi-source --------------------------------------

def test_25_same_source_pair_still_evaluated_correctly():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5), source=SOURCE_A)
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_A, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.mode == J_CUT


def test_26_multi_source_pair_evaluated_correctly():
    left, right = _safe_jcut_pair()
    assert left.source_asset_id != right.source_asset_id
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.mode == J_CUT


# --- 27-29: renderer contract compatibility (D-214 reuse in tests) ----------

def test_27_jcut_decision_is_executable_by_the_d214_renderer_contract():
    from cutsell_worker.render_plan import RenderSegment
    from cutsell_worker import render as render_module
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.mode == J_CUT
    # Translate the decision into the D-214 RenderSegment contract -- no
    # renderer redesign, just proving the two contracts compose.
    right_segment = RenderSegment(
        clip_id=right.clip_id, source_asset_id=right.source_asset_id, source_path="unused",
        start=right.start, end=right.end, audio_start=right.start - plan.overlap_duration,
    )
    assert right_segment.has_independent_audio_window is True
    render_module.validate_audio_window  # the exact D-214 validator this decision's output would go through


def test_28_lcut_decision_is_executable_by_the_d214_renderer_contract():
    from cutsell_worker.render_plan import RenderSegment
    left, right = _safe_lcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_tail_sec=0.4)
    assert plan.mode == L_CUT
    left_segment = RenderSegment(
        clip_id=left.clip_id, source_asset_id=left.source_asset_id, source_path="unused",
        start=left.start, end=left.end, audio_end=left.end + plan.overlap_duration,
    )
    assert left_segment.has_independent_audio_window is True


def test_29_micro_overlap_decision_is_executable_by_the_d214_renderer_contract():
    from cutsell_worker.render_plan import RenderSegment
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1, candidate_audio_tail_sec=0.1)
    assert plan.mode == MICRO_AUDIO_OVERLAP
    left_segment = RenderSegment(clip_id="a", source_asset_id=SOURCE_A, source_path="unused", start=0.0, end=5.0, audio_end=5.1)
    right_segment = RenderSegment(clip_id="b", source_asset_id=SOURCE_B, source_path="unused", start=5.0, end=10.0, audio_start=4.9)
    assert left_segment.has_independent_audio_window and right_segment.has_independent_audio_window


def test_30_renderer_unavailable_shape_falls_back_when_overlap_disabled():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, candidate_audio_lead_sec=0.2)
    assert plan.mode == HARD_CUT
    assert plan.fallback_reason == FALLBACK_OVERLAP_DISABLED


# --- 31: no exact-duration invention ----------------------------------------

def test_31_module_invents_no_duration_constant():
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text()
    for forbidden in ("_MS =", "_SEC = 0.", "THRESHOLD", "_ms =", "tight_cut_ms", "j_cut_ms", "l_cut_ms", "overlap_ms"):
        assert forbidden not in source, f"D-215 invented a duration constant: {forbidden!r}"


# --- 32-33: determinism / input-order stability -----------------------------

def test_32_deterministic_repeat():
    left, right = _safe_jcut_pair()
    plan1 = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    left2, right2 = _safe_jcut_pair()
    plan2 = decide_transition(left2, right2, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan1 == plan2


def test_33_input_order_stable_left_right_not_silently_swapped():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    assert plan.left_clip_id == left.clip_id
    assert plan.right_clip_id == right.clip_id


# --- 34-46: discipline / scope-isolation checks -----------------------------

def test_34_no_asr_or_provider_import():
    forbidden_modules = {
        "requests", "httpx", "urllib", "openai", "google", "genai", "modal", "runpod",
        "audio_silence", "whisper", "speech_recognition",
    }
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    hit = imported & forbidden_modules
    assert not hit, f"D-215 module imports an out-of-scope module: {hit}"


def test_35_no_boundary_ordering_besttake_family_module_imported():
    forbidden_modules = {
        "boundary_engine_pass", "post_selection_edge_only_boundary", "post_selection_interior_gap_trim",
        "ordering_realization_plan", "ordering_composer_adapter", "ordering_live_diagnostics_integration",
        "take_judge", "deterministic_best_take_authority", "multimodal_besttake_arbiter",
        "realization_resolver", "bounded_finalist_arbiter",
    }
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    hit = imported & forbidden_modules
    assert not hit, f"D-215 module imports an out-of-scope authority: {hit}"


def test_36_no_renderer_semantic_decision_in_module():
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text()
    for forbidden in ("_concat_render_command", "render_timeline_with_audio_windows", "ffmpeg"):
        assert forbidden not in source


def test_37_no_qa_commercial_or_sales_funnel_fields():
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text().lower()
    for forbidden in ("commercial_moment", "sales_funnel", "cut.ai", "human gold", "video00"):
        assert forbidden not in source


def test_38_no_live_wiring_into_universal_clean_cut_or_pipeline():
    # AST-based, not a raw substring check: dialogue_pacing_transition.py's
    # OWN docstring/comments legitimately name this module (documenting
    # which new fields it populates) without importing or calling it --
    # that is not live wiring and must not fail this check.
    for relative in ("universal_clean_cut.py", "pipeline.py", "dialogue_pacing_transition.py"):
        path = pathlib.Path("cutsell_worker") / relative
        tree = ast.parse(path.read_text())
        imported = set()
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[-1])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        assert "pacing_transition_decision" not in imported, f"{relative} imports the D-215 module"
        assert "decide_transition" not in called, f"{relative} calls decide_transition"


def test_39_live_pacing_modes_unchanged():
    from cutsell_worker.dialogue_pacing_transition import PHASE_1_EXECUTABLE_MODES
    assert PHASE_1_EXECUTABLE_MODES == (HARD_CUT, TIGHT_CUT)


def test_40_dialogue_transition_plan_extension_is_backward_compatible():
    from cutsell_worker.dialogue_pacing_transition import DialogueTransitionPlan
    plan = DialogueTransitionPlan(
        transition_index=0, left_clip_id="a", right_clip_id="b", mode=HARD_CUT,
        dialogue_overlap_enabled=False, visual_cut_time=1.0, left_audio_end=1.0, right_audio_start=1.0,
    )
    assert plan.pacing_gap_decision is None
    assert plan.conflict_flags == ()


def test_41_no_master_score_field():
    source = pathlib.Path("cutsell_worker/pacing_transition_decision.py").read_text().lower()
    for forbidden in ("master_score", "confidence_score", "overall_score"):
        assert forbidden not in source


def test_42_no_global_optimizer_only_pairwise():
    import inspect
    sig = inspect.signature(decide_transition)
    assert "left" in sig.parameters and "right" in sig.parameters
    assert "sequence" not in sig.parameters and "all_segments" not in sig.parameters


def test_43_run_summary_counts_correctly():
    left, right = _safe_jcut_pair()
    plan_j = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    left2 = _clip("c", 20.0, 25.0, "one two three", words=_words("one two three", 20.0, 24.5))
    right2 = _clip("d", 25.0, 30.0, "four five six", words=_words("four five six", 25.0, 29.5), source=SOURCE_B, order=3)
    plan_hard = decide_transition(left2, right2, dialogue_overlap_enabled=False, transition_index=1)
    summary = dialogue_pacing_transition_decision_run_summary((plan_j, plan_hard))
    assert summary["transition_count"] == 2
    assert summary["j_cut_count"] == 1
    assert summary["hard_cut_count"] == 1


def test_44_sequence_consistency_diagnostics_flags_nothing_for_normal_sequence():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    diag = sequence_consistency_diagnostics((plan,))
    assert diag["inconsistent_pair_count"] == 0


def test_45_compileall_clean():
    import py_compile
    py_compile.compile("cutsell_worker/pacing_transition_decision.py", doraise=True)
    py_compile.compile("cutsell_worker/dialogue_pacing_transition.py", doraise=True)


def test_46_no_double_speech_conflicted_used_outside_relationship_gate():
    # DOUBLE_SPEECH_CONFLICTED is reserved for the retry-relationship gate;
    # the main geometric path uses NO_OVERLAP_REQUIRED for real double
    # speech, matching this task's own literal instruction.
    from cutsell_worker.pacing_transition_decision import DOUBLE_SPEECH_CONFLICTED
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.9))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.1, 9.5), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=2.0)
    assert plan.double_speech_status != DOUBLE_SPEECH_CONFLICTED
    assert plan.double_speech_status == DOUBLE_SPEECH_NO_OVERLAP_REQUIRED
