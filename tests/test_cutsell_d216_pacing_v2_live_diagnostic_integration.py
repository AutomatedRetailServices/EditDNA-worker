"""D-216 -- PACING V2 LIVE DIAGNOSTIC INTEGRATION. OFFLINE-FIRST.

Proves `pacing_v2_live_diagnostics_integration.build_pacing_v2_live_diagnostics`
correctly adapts D-215's `decide_transition` over real, already-Boundary-
finalized `DraftClip` sequences at the live Pacing seam
(`universal_clean_cut.py`, strictly after D-142's own live
`apply_dialogue_pacing_transition_pass`) -- default-OFF feature flag, zero
live J_CUT/L_CUT/MICRO_AUDIO_OVERLAP authority, no re-computation of ASR/
P1/P2/Ordering/BestTake/Prosodic evidence, no mutation of `draft.selected`,
Boundary edges, D-142's own live diagnostics, or any `RenderSegment`/
renderer state. Generic fixtures only -- no Video00 wording, timestamps, or
clip ids.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import (
    HARD_CUT,
    J_CUT,
    L_CUT,
    MICRO_AUDIO_OVERLAP,
    TIGHT_CUT,
)
from cutsell_worker.pacing_transition_decision import (
    DECISION_CONFLICTED,
    DECISION_SAFE_FALLBACK,
    DECISION_SUPPORTED,
    DOUBLE_SPEECH_SAFE_J_CUT,
    DOUBLE_SPEECH_SAFE_L_CUT,
    DOUBLE_SPEECH_SAFE_MICRO_OVERLAP,
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
)
from cutsell_worker.prosodic_audio_v2 import ProsodicDeliveryEvidence
from cutsell_worker import pacing_v2_live_diagnostics_integration as pv2

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "pacing_v2_live_diagnostics_integration.py").read_text()


def _code_only(source: str) -> str:
    marker = '"""\nfrom __future__'
    idx = source.find(marker)
    return source[idx:] if idx != -1 else source


CODE_ONLY = _code_only(MODULE_SOURCE)

SOURCE_A = "synthetic_source_a"
SOURCE_B = "synthetic_source_b"


def _words(text: str, start: float, end: float) -> tuple[Word, ...]:
    tokens = text.split()
    if not tokens:
        return ()
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


def _safe_jcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def _safe_lcut_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    return left, right


def _safe_micro_overlap_pair():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.4, 9.5), source=SOURCE_B, order=1)
    return left, right


# ---------------------------------------------------------------------------
# 1-2: flag default OFF / ON.
# ---------------------------------------------------------------------------

def test_01_flag_default_off_when_env_unset():
    assert pv2.pacing_v2_diagnostics_enabled(env={}) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_02_flag_on_with_truthy_values(value):
    assert pv2.pacing_v2_diagnostics_enabled(env={"CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED": value}) is True


def test_02b_flag_off_with_falsy_values():
    for value in ("0", "false", "no", "off", ""):
        assert pv2.pacing_v2_diagnostics_enabled(env={"CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED": value}) is False


# ---------------------------------------------------------------------------
# 3: zero compute when OFF -- proven structurally at the live call site
# (see TestStructuralAudits.test_wiring_call_is_flag_gated) and here by the
# module's own public surface never being invoked by a caller that first
# checks the flag.
# ---------------------------------------------------------------------------

def test_03_not_evaluable_with_fewer_than_two_selected():
    result = pv2.build_pacing_v2_live_diagnostics((_clip("a", 0.0, 5.0, "one"),), dialogue_overlap_enabled=False)
    assert result["capability_status"] == pv2.CAPABILITY_NOT_EVALUABLE
    assert result["missing_evidence"] == (pv2.MISSING_FEWER_THAN_TWO_SELECTED,)
    assert result["transition_count"] == 0
    assert result["transitions"] == ()


def test_03b_not_evaluable_with_zero_selected():
    result = pv2.build_pacing_v2_live_diagnostics((), dialogue_overlap_enabled=False)
    assert result["capability_status"] == pv2.CAPABILITY_NOT_EVALUABLE


# ---------------------------------------------------------------------------
# 4-6: one pair / multiple adjacent pairs / ordered adjacency preserved.
# ---------------------------------------------------------------------------

def test_04_one_pair():
    left = _clip("a", 0.0, 5.0, "one two three")
    right = _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    assert result["transition_count"] == 1
    assert result["transitions"][0]["left_clip_id"] == "a"
    assert result["transitions"][0]["right_clip_id"] == "b"


def test_05_multiple_adjacent_pairs():
    a = _clip("a", 0.0, 5.0, "one two three")
    b = _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    c = _clip("c", 10.0, 15.0, "seven eight nine", source=SOURCE_B, order=2)
    result = pv2.build_pacing_v2_live_diagnostics((a, b, c), dialogue_overlap_enabled=False)
    assert result["transition_count"] == 2
    pairs = [(row["left_clip_id"], row["right_clip_id"]) for row in result["transitions"]]
    assert pairs == [("a", "b"), ("b", "c")]


def test_06_ordered_adjacency_preserved_not_all_pairs():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    c = _clip("c", 10.0, 15.0, "three", source=SOURCE_B, order=2)
    result = pv2.build_pacing_v2_live_diagnostics((a, b, c), dialogue_overlap_enabled=False)
    pairs = [(row["left_clip_id"], row["right_clip_id"]) for row in result["transitions"]]
    assert ("a", "c") not in pairs  # never a non-adjacent pair


# ---------------------------------------------------------------------------
# 7-8: words reused (no ASR rerun) / missing words.
# ---------------------------------------------------------------------------

def test_07_words_reused_by_reference_not_recomputed():
    captured = {}
    import cutsell_worker.pacing_v2_live_diagnostics_integration as mod

    def fake_decide_transition(left, right, **kwargs):
        captured["left"] = left
        captured["right"] = right
        from cutsell_worker.dialogue_pacing_transition import plan_dialogue_pacing_transitions
        return plan_dialogue_pacing_transitions((left, right), {}, dialogue_overlap_enabled=False)[0]

    left, right = _clip("a", 0.0, 5.0, "one two three"), _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    orig = mod.decide_transition
    mod.decide_transition = fake_decide_transition
    try:
        mod.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    finally:
        mod.decide_transition = orig
    assert captured["left"] is left  # same object -- .words never rebuilt
    assert captured["right"] is right


def test_08_missing_words_does_not_crash():
    left = _clip("a", 0.0, 5.0, "", words=())
    right = _clip("b", 5.0, 10.0, "", words=(), source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    assert result["transition_count"] == 1
    assert result["transitions"][0]["selected_mode"] in (HARD_CUT, TIGHT_CUT)


# ---------------------------------------------------------------------------
# 9-10: Prosodic reused if available / absent.
# ---------------------------------------------------------------------------

def test_09_prosody_reused_if_available():
    left, right = _clip("a", 0.0, 5.0, "one"), _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    prosody = {"a": _prosody(vocal_continuity_state="CONTINUOUS")}
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False, prosody_by_clip_id=prosody)
    assert result["transitions"][0]["prosodic_status"] == pv2.PROSODIC_STATUS_AVAILABLE
    assert result["prosodic_available_count"] == 1
    assert result["prosodic_unavailable_count"] == 0


def test_10_prosody_absent_by_default():
    left, right = _clip("a", 0.0, 5.0, "one"), _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    assert result["transitions"][0]["prosodic_status"] == pv2.PROSODIC_STATUS_UNAVAILABLE
    assert result["prosodic_available_count"] == 0
    assert result["prosodic_unavailable_count"] == 1


# ---------------------------------------------------------------------------
# 11: relationship hint reused.
# ---------------------------------------------------------------------------

def test_11_relationship_hint_reused():
    left, right = _clip("a", 0.0, 5.0, "one"), _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    hints = {("a", "b"): RELATIONSHIP_CORRECTION}
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True, relationship_hint_by_pair=hints,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.2}},
    )
    row = result["transitions"][0]
    assert row["relationship_hint"] == RELATIONSHIP_CORRECTION
    assert row["decision_status"] == DECISION_SAFE_FALLBACK
    assert result["relationship_hint_count"] == 1


# ---------------------------------------------------------------------------
# 12-15: no P1/P2 rerun, no ASR, no provider -- structural.
# ---------------------------------------------------------------------------

class TestStructuralNoRecompute:
    def test_12_13_no_p1_p2_rerun(self):
        banned = (
            "build_editorial_moment_understanding_for_sources(",
            "build_whole_video_editorial_understanding(",
            "from .editorial_moment_sequence_integration",
            "from .whole_video_editorial_reasoning",
        )
        for name in banned:
            assert name not in MODULE_SOURCE

    def test_14_no_asr(self):
        for name in ("from .asr", "from .language_spine import"):
            assert name not in MODULE_SOURCE

    def test_15_no_provider_or_network(self):
        for name in ("requests.", "urllib", "http.client", "OPENAI_API_KEY", "genai.", "GenerativeModel"):
            assert name not in MODULE_SOURCE
        assert "from .prosodic_audio_v2 import" not in MODULE_SOURCE  # type-consumer only, never calls it
        assert "analyze_prosodic_delivery" not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# 16-21: HARD / TIGHT / KEEP_PAUSE / J / L / MICRO diagnostics.
# ---------------------------------------------------------------------------

def test_16_hard_cut_diagnostic():
    left, right = _clip("a", 0.0, 5.0, "one two three"), _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    assert result["transitions"][0]["selected_mode"] == HARD_CUT


def test_17_tight_cut_diagnostic():
    left, right = _clip("a", 0.0, 5.0, "one two three"), _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    boundary_diag = {"boundary_engine_pass": {"audio_edge_rows": [
        {"clip_id": "a", "actions": [{"action": "tighten_audio_exit", "trim_sec": 0.3}]},
    ], "visual_edge_rows": []}}
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False, boundary_diagnostics=boundary_diag)
    assert result["transitions"][0]["selected_mode"] == TIGHT_CUT
    assert result["transitions"][0]["pacing_gap_decision"] == "TIGHTEN"


def test_18_keep_pause_diagnostic():
    left = _clip("a", 0.0, 5.0, "it is not five", words=_words("it is not five", 3.0, 5.0))
    right = _clip("b", 6.5, 10.0, "four five six", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    assert result["transitions"][0]["pacing_gap_decision"] == "KEEP_PAUSE"


def test_19_j_cut_diagnostic():
    left, right = _safe_jcut_pair()
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.2}},
    )
    row = result["transitions"][0]
    assert row["selected_mode"] == J_CUT
    assert row["double_speech_status"] == DOUBLE_SPEECH_SAFE_J_CUT
    assert row["decision_status"] == DECISION_SUPPORTED


def test_20_l_cut_diagnostic():
    left, right = _safe_lcut_pair()
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"tail": 0.4}},
    )
    row = result["transitions"][0]
    assert row["selected_mode"] == L_CUT
    assert row["double_speech_status"] == DOUBLE_SPEECH_SAFE_L_CUT


def test_21_micro_overlap_diagnostic():
    left, right = _safe_micro_overlap_pair()
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.1, "tail": 0.1}},
    )
    row = result["transitions"][0]
    assert row["selected_mode"] == MICRO_AUDIO_OVERLAP
    assert row["double_speech_status"] == DOUBLE_SPEECH_SAFE_MICRO_OVERLAP
    assert row["candidate_overlap"] is not None


# ---------------------------------------------------------------------------
# 22: advanced diagnostic never executed live.
# ---------------------------------------------------------------------------

def test_22_advanced_diagnostic_never_executed_live():
    # The wiring block in universal_clean_cut.py never reads plan.mode back
    # into anything the renderer/D-142 consume -- proven structurally. Code
    # only -- the docstring's own explanatory prose mentions these names.
    assert "draft.selected =" not in CODE_ONLY
    assert "RenderSegment" not in CODE_ONLY
    assert "render_plan" not in CODE_ONLY
    assert "import render" not in CODE_ONLY


# ---------------------------------------------------------------------------
# 23-27: D-142 unchanged / Boundary edges unchanged / selected ids unchanged /
# RenderSegment live audio windows unchanged / renderer command unchanged.
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    def _fixture(self):
        from cutsell_worker.contracts import (
            DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION,
        )
        words_a = _words("uno dos tres cuatro", 0.2, 4.8)
        words_b = _words("cinco seis siete", 5.2, 9.5)
        left = DraftClip(
            clip_id="kept_a", source_asset_id="src", source_order=0, start=0.0, end=5.0,
            text="uno dos tres cuatro", caption_text="uno dos tres cuatro", words=words_a,
            semantic_role=SemanticRole.STORY, selected=True,
        )
        right = DraftClip(
            clip_id="kept_b", source_asset_id="src", source_order=1, start=5.0, end=10.0,
            text="cinco seis siete", caption_text="cinco seis siete", words=words_b,
            semantic_role=SemanticRole.STORY, selected=True,
        )
        draft = DraftTimeline(
            schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
            selected=(left, right), alternates=(), discarded=(), diagnostics={},
        )
        result = ProcessingResult(schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY, draft=draft, stage_status={})
        return result, (left, right)

    def _run_pipeline_stub(self, monkeypatch, request_obj):
        """Mirrors the D-097.C precedent: fake `process_local_sources` so
        `process_universal_clean_cut_sources` runs its own real post-Freeze
        Boundary + D-142 + D-216 pipeline stages over a real, controlled
        draft, without needing full ASR/take-judge/whole-video fixtures."""
        import cutsell_worker.universal_clean_cut as universal
        result, clips = self._fixture()

        def fake_process(request, local_paths, **kwargs):
            return result

        monkeypatch.setattr(universal, "process_local_sources", fake_process)
        monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda res, paths: res)
        monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda res, paths, **kw: res)
        out = universal.process_universal_clean_cut_sources(
            request_obj, {}, asr_provider=object(), selection_reasoner=None,
        )
        return out, clips

    def test_23_default_off_d142_diagnostics_present_pacing_v2_absent(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", raising=False)
        out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert "dialogue_pacing_transition" in out.draft.diagnostics
        assert "pacing_v2" not in out.draft.diagnostics

    def test_24_boundary_edges_unchanged_flag_on(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        left, right = clips
        selected_by_id = {c.clip_id: c for c in out.draft.selected}
        assert selected_by_id["kept_a"].start == left.start
        assert selected_by_id["kept_a"].end == left.end
        assert selected_by_id["kept_b"].start == right.start
        assert selected_by_id["kept_b"].end == right.end

    def test_25_selected_ids_unchanged_flag_on(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        assert [c.clip_id for c in out.draft.selected] == [c.clip_id for c in clips]

    def test_26_renderable_segment_fields_never_touched(self, monkeypatch):
        # No RenderSegment is even constructed at this pipeline seam; the
        # structural audit in TestStructuralNoRecompute/TestStructuralAudits
        # proves the module never imports render_plan/render at all.
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert "render_plan" not in CODE_ONLY and "RenderSegment" not in CODE_ONLY
        assert out.draft.diagnostics.get("pacing_v2") is not None

    def test_27_d142_own_diagnostics_identical_flag_on_vs_off(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "0")
        off_out, _ = self._run_pipeline_stub(monkeypatch, object())
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        on_out, _ = self._run_pipeline_stub(monkeypatch, object())
        assert off_out.draft.diagnostics["dialogue_pacing_transition"] == on_out.draft.diagnostics["dialogue_pacing_transition"]

    def test_40_default_off_parity_all_other_keys_identical(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "0")
        off_out, _ = self._run_pipeline_stub(monkeypatch, object())
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        on_out, _ = self._run_pipeline_stub(monkeypatch, object())
        # D-224 adds one more additive-only key (`pacing_v2_handle_aware`)
        # under the SAME flag -- excluded here for the same reason
        # `pacing_v2` already was: this test proves every OTHER diagnostics
        # key is byte-identical regardless of the flag.
        # D-235G adds `selection_freeze_diagnostics`, present UNCONDITIONALLY
        # (not gated by this flag at all) -- but its own `pacing_v2_
        # serialized`/`pacing_v2_handle_aware_serialized` fields are a
        # deliberate, honest key-presence OBSERVATION of this exact flag's
        # effect (see `cutsell_worker/selection_freeze_diagnostics.py`), so
        # its content legitimately differs between the off/on runs -- that
        # is the fields doing their job, not a leak. Excluded for the same
        # reason as the two keys above: this test proves every OTHER key
        # (i.e. everything NOT already known to observe this flag) is
        # byte-identical regardless of it.
        excluded = ("pacing_v2", "pacing_v2_handle_aware", "selection_freeze_diagnostics")
        off_keys = {k: v for k, v in off_out.draft.diagnostics.items() if k not in excluded}
        on_keys = {k: v for k, v in on_out.draft.diagnostics.items() if k not in excluded}
        assert off_keys == on_keys
        assert "pacing_v2" not in off_out.draft.diagnostics
        assert "pacing_v2" in on_out.draft.diagnostics

    def test_41_flag_on_immutability_only_new_diagnostics_key(self, monkeypatch):
        monkeypatch.setenv("CUTSELL_PACING_V2_DIAGNOSTICS_ENABLED", "1")
        out, clips = self._run_pipeline_stub(monkeypatch, object())
        left, right = clips
        assert [c.clip_id for c in out.draft.selected] == ["kept_a", "kept_b"]
        selected_by_id = {c.clip_id: c for c in out.draft.selected}
        assert (selected_by_id["kept_a"].start, selected_by_id["kept_a"].end) == (left.start, left.end)
        assert (selected_by_id["kept_b"].start, selected_by_id["kept_b"].end) == (right.start, right.end)
        assert out.draft.diagnostics["pacing_v2"]["schema_version"] == pv2.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 28-30: meaning / word / double-speech firewalls -- expected 0 violations.
# ---------------------------------------------------------------------------

def test_28_meaning_firewall_blocks_and_never_yields_advanced_mode():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.5))
    right = _clip("b", 5.0, 10.0, "no it is not correct", words=_words("no it is not correct", 5.0, 9.5), source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.3}},
    )
    row = result["transitions"][0]
    assert row["meaning_safety_status"] == SAFETY_BLOCKED
    assert row["selected_mode"] not in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
    assert row["firewall_violation"] is False
    assert result["firewall_violation_count"] == 0


def test_29_word_safety_firewall_blocks_and_never_yields_advanced_mode():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.9))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.3, 9.5), source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 2.0}},
    )
    row = result["transitions"][0]
    assert row["selected_mode"] not in (J_CUT, L_CUT, MICRO_AUDIO_OVERLAP)
    assert row["firewall_violation"] is False


def test_30_double_speech_firewall_blocks_and_never_yields_advanced_mode():
    left = _clip("a", 0.0, 5.0, "one two three", words=_words("one two three", 0.0, 4.9))
    right = _clip("b", 5.0, 10.0, "four five six", words=_words("four five six", 5.1, 9.5), source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.3, "tail": 0.3}},
    )
    row = result["transitions"][0]
    assert row["selected_mode"] == HARD_CUT
    assert row["firewall_violation"] is False
    assert result["firewall_violation_count"] == 0


# ---------------------------------------------------------------------------
# 31-33: correction / retry / continuation relationship handling.
# ---------------------------------------------------------------------------

def test_31_correction_relationship_always_safe_fallback():
    left, right = _clip("a", 0.0, 5.0, "one"), _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        relationship_hint_by_pair={("a", "b"): RELATIONSHIP_CORRECTION},
        candidate_timing_by_pair={("a", "b"): {"lead": 0.2}},
    )
    assert result["transitions"][0]["decision_status"] == DECISION_SAFE_FALLBACK


def test_32_retry_relationship_always_conflicted_hard_cut():
    left, right = _clip("a", 0.0, 5.0, "one"), _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        relationship_hint_by_pair={("a", "b"): RELATIONSHIP_RETRY},
    )
    row = result["transitions"][0]
    assert row["decision_status"] == DECISION_CONFLICTED
    assert row["selected_mode"] == HARD_CUT


def test_33_continuation_relationship_does_not_restrict_a_safe_jcut():
    left, right = _safe_jcut_pair()
    result = pv2.build_pacing_v2_live_diagnostics(
        (left, right), dialogue_overlap_enabled=True,
        candidate_timing_by_pair={("a", "b"): {"lead": 0.2}},
        relationship_hint_by_pair={("a", "b"): RELATIONSHIP_CONTINUATION},
    )
    assert result["transitions"][0]["selected_mode"] == J_CUT


# ---------------------------------------------------------------------------
# 34: multi-source.
# ---------------------------------------------------------------------------

def test_34_multi_source_pair_evaluated_correctly():
    left = _clip("a", 0.0, 5.0, "one two three", source=SOURCE_A)
    right = _clip("b", 5.0, 10.0, "four five six", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    row = result["transitions"][0]
    assert row["left_source_asset_id"] == SOURCE_A
    assert row["right_source_asset_id"] == SOURCE_B


# ---------------------------------------------------------------------------
# 35-36: diagnostics bounded, no transcript dump.
# ---------------------------------------------------------------------------

def test_35_36_diagnostics_bounded_no_transcript():
    left = _clip("a", 0.0, 5.0, "a fairly long sentence with many words in it here")
    right = _clip("b", 5.0, 10.0, "another long sentence with plenty more words too", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((left, right), dialogue_overlap_enabled=False)
    row = result["transitions"][0]
    assert "text" not in row
    assert "words" not in row
    assert "transcript" not in result
    assert "transcript" not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# 37: run summary correctness.
# ---------------------------------------------------------------------------

def test_37_run_summary_correctness():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    c = _clip("c", 10.0, 15.0, "three", source=SOURCE_B, order=2)
    result = pv2.build_pacing_v2_live_diagnostics((a, b, c), dialogue_overlap_enabled=False)
    summary = result["run_summary"]
    assert summary["transition_count"] == 2
    assert summary["hard_cut_count"] == 2


# ---------------------------------------------------------------------------
# 38-39: deterministic repeat / input-order stability.
# ---------------------------------------------------------------------------

def test_38_deterministic_repeat():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    r1 = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False)
    r2 = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False)
    assert r1 == r2


def test_39_input_order_stability_not_silently_swapped():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False)
    row = result["transitions"][0]
    assert row["left_clip_id"] == "a" and row["right_clip_id"] == "b"


# ---------------------------------------------------------------------------
# 42-44: no QA / commercial / master-score references.
# ---------------------------------------------------------------------------

class TestStructuralAudits:
    def test_42_no_qa_references(self):
        for banned in ("cut_ai", "cutai", "human_gold", "quality_ladder", "benchmark_label"):
            assert banned not in CODE_ONLY.lower()

    def test_43_no_commercial_or_funnel_fields(self):
        for banned in ("commercial", "sales_funnel", "funnel", "cta_score", "hook_strength", "narrative_quality"):
            assert banned not in CODE_ONLY.lower()

    def test_44_no_master_score(self):
        assert '"score"' not in CODE_ONLY
        assert "master_score" not in CODE_ONLY

    def test_no_family_besttake_boundary_ordering_authority_mutation(self):
        forbidden = (
            "take_group_id =", "_semantic_best_take", "bounded_finalist_authority",
            "boundary_engine_pass.apply", "BoundaryEngine(", "ordering_realization_plan",
            "composite_resolver", "realization_resolver",
        )
        for name in forbidden:
            assert name not in CODE_ONLY

    def test_no_selected_clip_mutation(self):
        assert "selected_clip_id =" not in CODE_ONLY
        assert ".selected = " not in CODE_ONLY

    def test_d142_reused_not_reimplemented(self):
        # This module never redefines D-142's own live planner or `_plan_row`.
        assert "def plan_dialogue_pacing_transitions(" not in MODULE_SOURCE
        assert "def apply_dialogue_pacing_transition_pass(" not in MODULE_SOURCE

    def test_d215_decide_transition_reused_not_reimplemented(self):
        assert "def decide_transition(" not in MODULE_SOURCE
        assert "decide_transition(" in MODULE_SOURCE  # it IS called

    def test_wiring_call_is_flag_gated(self):
        # D-217 replaced the direct D-216 call with `build_pacing_v2_live_
        # diagnostics_with_real_evidence` (pacing_v2_evidence_adapter.py),
        # which itself calls this module's own unmodified function
        # underneath -- both names share this prefix, so the check accepts
        # either, still requiring the call inside the flag-gated block.
        source = (REPO_ROOT / "cutsell_worker" / "universal_clean_cut.py").read_text()
        idx = source.index("if pacing_v2_diagnostics_enabled():")
        call_idx = source.index("build_pacing_v2_live_diagnostics", idx)
        assert call_idx > idx  # the real call site is textually inside the flag-gated block
        # And the flag check precedes the call with nothing but the block's
        # own body between them (no other top-level `if` reopens scope).
        between = source[idx:call_idx]
        assert between.count("\n    else:") == 0 and "def " not in between

    def test_wiring_import_present_and_additive_only(self):
        source = (REPO_ROOT / "cutsell_worker" / "universal_clean_cut.py").read_text()
        assert "from .pacing_v2_live_diagnostics_integration import" in source
        # D-224 reformatted this call site's own dict literal across
        # multiple lines (adding its own additive "pacing_v2_handle_aware"
        # key alongside this one) -- an exact-substring match on the
        # original single-line shape is no longer meaningful; a plain
        # "key: value" substring survives any such reformatting.
        assert '"pacing_v2": pacing_v2_diag' in source

    def test_ast_no_module_level_side_effects(self):
        tree = ast.parse(MODULE_SOURCE)
        for node in tree.body:
            if isinstance(node, (ast.Expr,)) and not isinstance(node, ast.Constant):
                # allow only docstring-shaped expressions
                if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                    pytest.fail(f"unexpected module-level side effect: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# 45: compileall.
# ---------------------------------------------------------------------------

def test_45_module_compiles_and_imports():
    import cutsell_worker.pacing_v2_live_diagnostics_integration  # noqa: F401


def test_capability_status_available_with_no_optional_evidence():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False)
    assert result["capability_status"] == pv2.CAPABILITY_AVAILABLE


def test_capability_status_partial_on_mixed_prosody_coverage():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    c = _clip("c", 10.0, 15.0, "three", source=SOURCE_B, order=2)
    result = pv2.build_pacing_v2_live_diagnostics(
        (a, b, c), dialogue_overlap_enabled=False, prosody_by_clip_id={"a": _prosody()},
    )
    assert result["capability_status"] == pv2.CAPABILITY_PARTIAL


def test_comparison_classification_agreement_vs_incomparable():
    a = _clip("a", 0.0, 5.0, "one")
    b = _clip("b", 5.0, 10.0, "two", source=SOURCE_B, order=1)
    result_agree = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False, live_transition_modes=(HARD_CUT,))
    assert result_agree["transitions"][0]["live_vs_v2_comparison"] == pv2.COMPARISON_AGREEMENT
    assert result_agree["comparison_agreement_count"] == 1

    result_none = pv2.build_pacing_v2_live_diagnostics((a, b), dialogue_overlap_enabled=False)
    assert result_none["transitions"][0]["live_vs_v2_comparison"] == pv2.COMPARISON_INCOMPARABLE
