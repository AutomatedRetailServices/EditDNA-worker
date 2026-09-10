"""D-218F -- Pacing V2 fallback transition-index identity integrity fix.

D-218R's real-media analysis root-caused a genuine `cutsell_worker` bug:
`_fallback_plan()` in `pacing_transition_decision.py` never reapplied the
caller's `transition_index` onto the plan it returned. Because `baseline`
comes from `plan_dialogue_pacing_transitions((left, right), ...)` -- a
fresh 2-clip mini-sequence whose own internal enumeration always assigns
`transition_index=0` -- every plan built via `_fallback_plan` (the RETRY
relationship gate, the CORRECTION relationship gate, and the overlap-
disabled/no-candidate-offered gate) silently reverted to index 0
regardless of its real position in the full timeline. This corrupted
`sequence_consistency_diagnostics`'s own per-pair identity labels (D-218R
observed all 9 flagged real pairs mislabeled `(0, 0)`).

This file proves the single, targeted fix: `_fallback_plan` now takes and
reapplies `transition_index`, and all three of its call sites pass the
real one through -- with ZERO change to any editorial decision (mode,
gap decision, safety status, fallback reason, candidate timing, conflict
flags) for identical inputs. No Video00 literals, no new thresholds, no
provider, no Boundary/Ordering/Renderer/Family/BestTake change.
"""
from __future__ import annotations

import ast
import pathlib

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import (
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
)
from cutsell_worker.pacing_transition_decision import (
    CONFLICT_CORRECTION_RELATIONSHIP,
    CONFLICT_OVERLAP_DISABLED,
    CONFLICT_RETRY_RELATIONSHIP,
    CONFLICT_WORD_TIMING_MISSING,
    DECISION_CONFLICTED,
    DECISION_SAFE_FALLBACK,
    DECISION_SUPPORTED,
    DECISION_UNKNOWN,
    DOUBLE_SPEECH_SAFE_J_CUT,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_UNKNOWN,
    decide_transition,
    sequence_consistency_diagnostics,
)
from cutsell_worker.prosodic_audio_v2 import ProsodicDeliveryEvidence

MODULE_PATH = pathlib.Path("cutsell_worker/pacing_transition_decision.py")
MODULE_SOURCE = MODULE_PATH.read_text()

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


def _no_candidate_pair(index_label: int):
    left = _clip(f"a{index_label}", 0.0, 5.0, "one two three")
    right = _clip(f"b{index_label}", 5.0, 10.0, "four five six", source=SOURCE_B, order=index_label + 1)
    return left, right


def _safe_jcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def _safe_lcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def _safe_micro_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    return left, right


def _meaning_block_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "it did not work", words=_words("it did not work", 5.0, 5.3), source=SOURCE_B, order=1)
    return left, right


# ---------------------------------------------------------------------------
# 1-4: fallback index preserved at several distinct real positions.
# ---------------------------------------------------------------------------

def test_01_fallback_index_0():
    # dialogue_overlap_enabled=False -> the CONFLICT_OVERLAP_DISABLED
    # shape of this same fallback gate (D-218R's own real RAW never set
    # any overlap-enabling flag, so this is the branch every one of the
    # 26 real transitions actually took -- CONFLICT_NO_CANDIDATE_OFFERED,
    # covered separately below, is the sibling shape of the SAME gate).
    left, right = _no_candidate_pair(0)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=0)
    assert plan.transition_index == 0
    assert plan.fallback_reason == CONFLICT_OVERLAP_DISABLED


def test_02_fallback_index_1():
    left, right = _no_candidate_pair(1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=1)
    assert plan.transition_index == 1


def test_03_fallback_index_2():
    left, right = _no_candidate_pair(2)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=2)
    assert plan.transition_index == 2


def test_04_fallback_index_10():
    left, right = _no_candidate_pair(10)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=10)
    assert plan.transition_index == 10


# ---------------------------------------------------------------------------
# 5-10: every _fallback_plan call site (and, for completeness, the
# main-path gates that were already correct) preserves the real index.
# ---------------------------------------------------------------------------

def test_05_conflict_no_candidate_offered_index_preserved():
    # dialogue_overlap_enabled=True but no candidate offered -- the SAME
    # `_fallback_plan` call site as test_01, just the sibling branch. The
    # locally-computed `reason` here is CONFLICT_NO_CANDIDATE_OFFERED, but
    # it is only ever surfaced into `fallback_reason`/`conflict_flags`
    # when overlap is disabled (a separate, pre-existing behavior this
    # task does not change) -- decision_status stays SUPPORTED either
    # way. The identity fix applies identically to both branches of this
    # one gate since they share the same `_fallback_plan` call.
    left, right = _no_candidate_pair(7)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=7)
    assert plan.fallback_reason is None
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.transition_index == 7


def test_05b_overlap_disabled_index_preserved():
    left, right = _no_candidate_pair(8)
    plan = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=8)
    assert plan.fallback_reason == CONFLICT_OVERLAP_DISABLED
    assert plan.transition_index == 8


def test_06_missing_word_fallback_index_preserved():
    # This path returns via the module's own MAIN return (already correct
    # pre-fix), included here as a completeness/no-regression control.
    left = _clip("a", 0.0, 5.0, "one two three", words=())
    right = _clip("b", 5.0, 10.0, "four five six", words=(), source=SOURCE_B, order=1)
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3, transition_index=9)
    assert plan.decision_status == DECISION_UNKNOWN
    assert plan.word_safety_status == SAFETY_UNKNOWN
    assert CONFLICT_WORD_TIMING_MISSING in plan.conflict_flags
    assert plan.transition_index == 9


def test_07_correction_fallback_index_preserved():
    left, right = _safe_jcut_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1,
        relationship_hint=RELATIONSHIP_CORRECTION, transition_index=5,
    )
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert CONFLICT_CORRECTION_RELATIONSHIP in plan.conflict_flags
    assert plan.transition_index == 5


def test_08_retry_fallback_index_preserved():
    left, right = _safe_jcut_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.1,
        relationship_hint=RELATIONSHIP_RETRY, transition_index=6,
    )
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_CONFLICTED
    assert CONFLICT_RETRY_RELATIONSHIP in plan.conflict_flags
    assert plan.transition_index == 6


def test_09_meaning_fallback_index_preserved():
    # Main-path (not _fallback_plan) -- completeness/no-regression control.
    left, right = _meaning_block_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.3, transition_index=11)
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert plan.transition_index == 11


def test_10_prosodic_restart_fallback_index_preserved():
    # Main-path (not _fallback_plan) -- completeness/no-regression control.
    left, right = _safe_jcut_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2,
        right_prosody=_prosody(restart_or_interruption_state="RESTART_DETECTED"),
        transition_index=12,
    )
    assert plan.mode == HARD_CUT
    assert plan.decision_status == DECISION_SAFE_FALLBACK
    assert plan.transition_index == 12


# ---------------------------------------------------------------------------
# 11-13: supported (non-fallback) advanced-mode paths keep their index too.
# ---------------------------------------------------------------------------

def test_11_supported_jcut_index_preserved():
    left, right = _safe_jcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2, transition_index=3)
    assert plan.mode == J_CUT
    assert plan.decision_status == DECISION_SUPPORTED
    assert plan.double_speech_status == DOUBLE_SPEECH_SAFE_J_CUT
    assert plan.transition_index == 3


def test_12_supported_lcut_index_preserved():
    left, right = _safe_lcut_pair()
    plan = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_tail_sec=0.4, transition_index=4)
    assert plan.mode == L_CUT
    assert plan.transition_index == 4


def test_13_supported_micro_overlap_index_preserved():
    left, right = _safe_micro_pair()
    plan = decide_transition(
        left, right, dialogue_overlap_enabled=True,
        candidate_audio_lead_sec=0.1, candidate_audio_tail_sec=0.1, transition_index=13,
    )
    assert plan.mode == MICRO_AUDIO_OVERLAP
    assert plan.transition_index == 13


# ---------------------------------------------------------------------------
# 14-19: NO DECISION DRIFT -- every other field is byte-identical to the
# pre-fix value for identical inputs, only the identity field changed.
# ---------------------------------------------------------------------------

def test_14_hard_fallback_decision_status_unchanged():
    left, right = _no_candidate_pair(20)
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=True)  # default transition_index=0
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=20)
    assert plan_no_index.decision_status == plan_real_index.decision_status == DECISION_SUPPORTED


def test_15_selected_mode_unchanged():
    left, right = _no_candidate_pair(21)
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=True)
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=21)
    assert plan_no_index.mode == plan_real_index.mode


def test_16_fallback_reason_unchanged():
    left, right = _no_candidate_pair(22)
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=False)
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=False, transition_index=22)
    assert plan_no_index.fallback_reason == plan_real_index.fallback_reason == CONFLICT_OVERLAP_DISABLED


def test_17_safety_statuses_unchanged():
    left, right = _no_candidate_pair(23)
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=True)
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=23)
    assert (plan_no_index.meaning_safety_status, plan_no_index.word_safety_status, plan_no_index.double_speech_status) == (
        plan_real_index.meaning_safety_status, plan_real_index.word_safety_status, plan_real_index.double_speech_status,
    )


def test_18_candidate_timing_fields_unchanged():
    left, right = _safe_jcut_pair()
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2)
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=True, candidate_audio_lead_sec=0.2, transition_index=17)
    assert plan_no_index.overlap_duration == plan_real_index.overlap_duration


def test_19_conflict_flags_unchanged():
    left, right = _no_candidate_pair(24)
    plan_no_index = decide_transition(left, right, dialogue_overlap_enabled=True)
    plan_real_index = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=24)
    assert plan_no_index.conflict_flags == plan_real_index.conflict_flags


# ---------------------------------------------------------------------------
# 20: multi-transition sequence -- unique, correctly-ordered indices, and
# sequence-consistency diagnostics reference real neighboring positions.
# ---------------------------------------------------------------------------

def test_20_multi_transition_sequence_unique_indices_no_fallback_collision():
    # 4 adjacent pairs (indices 0-3), all forced through the same
    # `_fallback_plan` gate D-218R's real RAW actually exercised on every
    # one of its 26 real transitions (dialogue_overlap_enabled=False ->
    # CONFLICT_OVERLAP_DISABLED, since that flag was never set for that
    # run).
    plans = [
        decide_transition(*_no_candidate_pair(i), dialogue_overlap_enabled=False, transition_index=i)
        for i in range(4)
    ]
    assert [p.transition_index for p in plans] == [0, 1, 2, 3]
    # Before the fix this would have been [0, 0, 0, 0] -- a genuine
    # identity collision purely from fallback construction.
    assert len({p.transition_index for p in plans}) == 4


def _keep_pause_fallback_pair(idx: int):
    left = _clip(f"a{idx}", 0.0, 5.0, "no it is not true", words=_words("no it is not true", 0.0, 4.5))
    right = _clip(f"b{idx}", 5.5, 10.0, "four five six", source=SOURCE_B, order=idx + 1)
    return left, right


def _tighten_fallback_pair(idx: int):
    left = _clip(f"a{idx}", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip(f"b{idx}", 5.5, 10.0, "four five six", source=SOURCE_B, order=idx + 1)
    return left, right


def test_20b_sequence_consistency_references_real_neighboring_positions():
    # This is the EXACT shape D-218R found on real data: two fallback
    # plans (both HARD_CUT, both gap_removed_duration=0.0 -- identical
    # "evidence") that genuinely reach a DIFFERENT pacing_gap_decision
    # (KEEP_PAUSE vs TIGHTEN, driven by D-038's own claim-criticality
    # classifier on each one's own trailing words) -- a real, correctly-
    # flagged inconsistency. Before this fix, both fallback plans shared
    # the baseline's bogus transition_index=0 regardless of position;
    # after the fix, the flag must name the REAL, distinct positions.
    plan_5 = decide_transition(*_keep_pause_fallback_pair(5), dialogue_overlap_enabled=False, transition_index=5)
    plan_6 = decide_transition(*_tighten_fallback_pair(6), dialogue_overlap_enabled=False, transition_index=6)
    assert plan_5.mode == plan_6.mode
    assert plan_5.gap_removed_duration == plan_6.gap_removed_duration
    assert plan_5.pacing_gap_decision != plan_6.pacing_gap_decision

    diag = sequence_consistency_diagnostics((plan_5, plan_6))
    assert diag["inconsistent_pair_count"] == 1
    flag = diag["inconsistent_pairs"][0]
    assert flag["left_index"] == 5
    assert flag["right_index"] == 6
    # The pre-fix bug would have produced (0, 0) here regardless of the
    # plans' real positions -- structurally impossible to reproduce now.
    assert (flag["left_index"], flag["right_index"]) != (0, 0)


# ---------------------------------------------------------------------------
# 21: deterministic repeat.
# ---------------------------------------------------------------------------

def test_21_deterministic_repeat():
    left, right = _no_candidate_pair(9)
    plan1 = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=9)
    plan2 = decide_transition(left, right, dialogue_overlap_enabled=True, transition_index=9)
    assert plan1 == plan2


# ---------------------------------------------------------------------------
# 22-28: structural/scope audits.
# ---------------------------------------------------------------------------

def test_22_no_video00_hardcode():
    forbidden = ("VIDEO-2026-07-30", "D40F1D43", "5E01F214")
    for needle in forbidden:
        assert needle not in MODULE_SOURCE


def test_23_no_new_duration_threshold_constant():
    tree = ast.parse(MODULE_SOURCE)
    docstring_end = 0
    module_docstring = ast.get_docstring(tree, clean=False)
    if module_docstring:
        docstring_end = MODULE_SOURCE.index(module_docstring) + len(module_docstring)
    code_only = MODULE_SOURCE[docstring_end:]
    for needle in ("J_CUT_MS", "L_CUT_MS", "OVERLAP_MS", "0.15", "0.25", "150", "250"):
        assert needle not in code_only, f"possible invented duration constant: {needle!r}"


def test_24_no_provider_or_network_import():
    for banned in ("requests", "urllib", "openai", "google.generativeai", "genai"):
        assert banned not in MODULE_SOURCE


def test_25_no_boundary_or_ordering_module_imported():
    for banned in ("boundary_engine", "human_boundary_polish", "ordering_", "editorial_moment_sequence"):
        assert banned not in MODULE_SOURCE


def test_26_no_family_or_besttake_module_imported():
    for banned in ("retry_family", "best_take_resolver", "deterministic_best_take_authority"):
        assert banned not in MODULE_SOURCE


def test_27_no_renderer_module_imported():
    # The module's own docstring legitimately NAMES `render_plan.py`/
    # `render.py` as downstream canonical-order stages (prose, not code)
    # -- the real property is no actual import statement pulling either
    # module in.
    tree = ast.parse(MODULE_SOURCE)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    for banned in ("render_plan", "render", ".render_plan", ".render"):
        assert banned not in imported_names, f"unexpected renderer import: {banned!r}"


def test_28_no_live_advanced_authority_introduced():
    # Same structural fact D-215/D-216/D-217 already proved: this module
    # is never imported by the live wiring seam directly (only via the
    # D-216/D-217 diagnostics-only integration, itself flag-gated).
    live_files = ("universal_clean_cut.py", "pipeline.py")
    for fname in live_files:
        path = pathlib.Path("cutsell_worker") / fname
        text = path.read_text()
        assert "pacing_transition_decision" not in text


def test_29_compileall_clean():
    import py_compile
    py_compile.compile(str(MODULE_PATH), doraise=True)


def test_30_fallback_plan_signature_requires_transition_index():
    # Structural proof the fix is the signature itself, not an incidental
    # default -- a caller MUST supply transition_index (no silent zero).
    tree = ast.parse(MODULE_SOURCE)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_fallback_plan")
    kwonly_names = [a.arg for a in func.args.kwonlyargs]
    assert "transition_index" in kwonly_names
    idx = kwonly_names.index("transition_index")
    assert func.args.kw_defaults[idx] is None  # no default -- caller-required
