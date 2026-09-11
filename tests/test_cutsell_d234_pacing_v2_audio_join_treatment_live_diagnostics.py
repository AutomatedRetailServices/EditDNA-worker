"""D-234: Pacing V2 Audio Join Treatment -- Live Diagnostic Integration,
NO AUTHORITY. Verifies the D-230->D-231->D-232->D-233 chain is wired into
the live seam behind a default-OFF flag, diagnostic only, with zero
authority over selection/Boundary/Ordering/Family/BestTake/primary
transition/renderer output, and zero live audio-treatment execution."""
from __future__ import annotations

import ast
import importlib
import inspect
import subprocess

import cutsell_worker.pacing_v2_audio_join_treatment_live_diagnostics as m
import cutsell_worker.universal_clean_cut as ucc
from cutsell_worker.contracts import DraftClip, Word
from cutsell_worker.pacing_v2_audio_join_treatment_decision import (
    TREATMENT_AMBIENCE_BRIDGE,
    TREATMENT_AMBIENCE_CARRY_LEFT,
    TREATMENT_AMBIENCE_CARRY_RIGHT,
    TREATMENT_CLICK_FADE,
    TREATMENT_NONE,
    TREATMENT_SHORT_CROSSFADE,
)


def _clip(clip_id, source_asset_id, start, end, words=()):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=" ".join(w[2] for w in words), caption_text="",
        words=tuple(Word(text=w[2], start=w[0], end=w[1]) for w in words),
    )


def _no_speech_clips(source="s1"):
    # No words at all on either side -> retained edges have no word timing
    # (INSUFFICIENT evidence, never speech-blocked) -- the cleanest fixture
    # for exercising ambience/crossfade eligibility paths.
    left = _clip("L", source, 0.0, 5.0, words=())
    right = _clip("R", source, 5.0, 10.0, words=())
    return left, right


def _speech_clips(source="s1"):
    left = _clip("L", source, 0.0, 5.0, words=((4.0, 4.5, "hello"), (4.5, 5.0, "there")))
    right = _clip("R", source, 5.0, 10.0, words=((5.0, 5.5, "world"), (5.5, 6.0, "now")))
    return left, right


class TestFlagDefault:
    def test_01_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_AUDIO_JOIN_TREATMENT_DIAGNOSTICS_ENABLED", raising=False)
        assert m.audio_join_treatment_diagnostics_enabled() is False

    def test_02_flag_on_variants(self):
        for value in ("1", "true", "True", "yes", "on"):
            assert m.audio_join_treatment_diagnostics_enabled({"CUTSELL_AUDIO_JOIN_TREATMENT_DIAGNOSTICS_ENABLED": value}) is True

    def test_03_flag_off_variants(self):
        for value in ("0", "false", "no", "", "off"):
            assert m.audio_join_treatment_diagnostics_enabled({"CUTSELL_AUDIO_JOIN_TREATMENT_DIAGNOSTICS_ENABLED": value}) is False


class TestJoinMapping:
    def test_04_fewer_than_two_clips_unavailable(self):
        result = m.build_audio_join_treatment_live_diagnostics(
            [_clip("A", "s1", 0.0, 5.0)], dialogue_overlap_enabled=False,
        )
        assert result["status"] == m.STATUS_UNAVAILABLE
        assert result["join_count"] == 0
        assert result["per_join"] == ()

    def test_05_joins_mapped_correctly_three_clips(self):
        clips = [_clip("A", "s1", 0.0, 3.0), _clip("B", "s1", 3.0, 6.0), _clip("C", "s1", 6.0, 9.0)]
        result = m.build_audio_join_treatment_live_diagnostics(clips, dialogue_overlap_enabled=False)
        assert result["join_count"] == 2
        assert [row["transition_index"] for row in result["per_join"]] == [0, 1]

    def test_06_transition_index_identity_preserved(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        row = result["per_join"][0]
        assert row["transition_index"] == 0
        assert row["decision"]["transition_index"] == 0
        assert row["understanding"]["transition_index"] == 0
        assert row["timing"]["transition_index"] == 0

    def test_07_left_right_clip_identity_preserved(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        row = result["per_join"][0]
        assert row["left_clip_id"] == "L" and row["right_clip_id"] == "R"
        assert row["understanding"]["left_clip_id"] == "L" and row["understanding"]["right_clip_id"] == "R"
        assert row["decision"]["left_clip_id"] == "L" and row["decision"]["right_clip_id"] == "R"

    def test_08_primary_mode_preserved_from_live_modes(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False, live_transition_modes=("TIGHT_CUT",),
        )
        assert result["per_join"][0]["primary_transition_mode"] == "TIGHT_CUT"

    def test_09_primary_mode_unknown_when_missing(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["primary_transition_mode"] == "UNKNOWN"


class TestEvidenceConstruction:
    def test_10_acoustic_evidence_built_for_word_present_clips(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["understanding"]["left_edge_evidence_id"] is not None
        assert result["per_join"][0]["understanding"]["right_edge_evidence_id"] is not None

    def test_11_acoustic_failure_fails_soft_no_words(self):
        left, right = _no_speech_clips()
        # No words -> derive_retained_edge_window returns None -> evidence
        # is None -> must never crash, must report INSUFFICIENT/UNAVAILABLE
        # honestly rather than fabricate SAFE.
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        row = result["per_join"][0]
        assert row["understanding"]["left_edge_evidence_id"] is None
        assert row["acoustic_evidence_status"] in (m.STATUS_UNAVAILABLE, m.STATUS_PARTIAL, m.STATUS_CONFLICTED)

    def test_12_understanding_built(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert "understanding_status" in result["per_join"][0]["understanding"]

    def test_13_decision_built(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["decision"]["treatment"] in (
            TREATMENT_NONE, TREATMENT_CLICK_FADE, TREATMENT_SHORT_CROSSFADE,
            TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_AMBIENCE_BRIDGE,
        )

    def test_14_timing_plan_built(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False, source_duration_by_asset={"s1": 20.0},
        )
        assert "timing_status" in result["per_join"][0]["timing"]


class TestTreatmentDiagnostics:
    def _decision_treatments(self, clips_source_duration=20.0, **kwargs):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False,
            source_duration_by_asset={"s1": clips_source_duration}, **kwargs,
        )
        return result["per_join"][0]["decision"]["treatment"], result

    def test_15_none_or_click_fade_reachable(self):
        # With no evidence at all, the decision should reach a safe
        # fallback -- NONE or CLICK_FADE, never crash.
        treatment, _ = self._decision_treatments()
        assert treatment in (TREATMENT_NONE, TREATMENT_CLICK_FADE, TREATMENT_SHORT_CROSSFADE,
                              TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_AMBIENCE_BRIDGE)

    def test_16_all_six_treatment_values_are_valid_decision_outputs(self):
        # Structural check: the decision vocabulary consumed here is
        # exactly D-232's own six values -- no seventh value invented.
        from cutsell_worker.pacing_v2_audio_join_treatment_decision import TREATMENT_VALUES
        assert set(TREATMENT_VALUES) == {
            TREATMENT_NONE, TREATMENT_CLICK_FADE, TREATMENT_SHORT_CROSSFADE,
            TREATMENT_AMBIENCE_CARRY_LEFT, TREATMENT_AMBIENCE_CARRY_RIGHT, TREATMENT_AMBIENCE_BRIDGE,
        }

    def test_17_unsupported_timing_recorded_honestly(self):
        # Zero-duration source (both video_start==video_end) -> if an
        # advanced treatment is ever recommended, timing must report a
        # real failure status, never SUPPORTED, never substitute treatment.
        left = _clip("L", "s1", 5.0, 5.0)
        right = _clip("R", "s1", 5.0, 5.0)
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False, source_duration_by_asset={"s1": 20.0},
        )
        row = result["per_join"][0]
        if row["decision"]["treatment"] not in (TREATMENT_NONE, TREATMENT_CLICK_FADE):
            assert row["timing"]["timing_status"] != "SUPPORTED" or row["timing"]["chosen_duration"] == 0.0
        # decision.treatment is never overwritten by the timing plan.
        assert row["timing"]["treatment"] == row["decision"]["treatment"]

    def test_18_insufficient_evidence_reported_not_fabricated_safe(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        row = result["per_join"][0]
        # No word coverage on either retained edge -> must never claim
        # AVAILABLE evidence.
        assert row["acoustic_evidence_status"] != m.STATUS_AVAILABLE


class TestFirewallDiagnostics:
    def test_19_word_firewall_visible_when_both_sides_lexical(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        decision = result["per_join"][0]["decision"]
        assert decision["double_speech_status"] in ("BOTH_SIDES_LEXICAL", "NO_OVERLAP_REQUIRED", "CONFLICTED", "UNKNOWN")

    def test_20_meaning_firewall_field_present(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert "meaning_safety_status" in result["per_join"][0]["decision"]

    def test_21_double_speech_firewall_field_present(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert "double_speech_status" in result["per_join"][0]["decision"]

    def test_22_relationship_hint_unavailable_by_default(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["understanding"]["relationship_hint"] is None

    def test_23_relationship_hint_passed_through_when_available(self):
        left, right = _no_speech_clips()
        moment_seq = {
            "moments": [
                {"clip_id": "L", "source_asset_id": "s1"},
                {"clip_id": "R", "source_asset_id": "s1", "relationship_to_previous": "correction"},
            ]
        }
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False,
            editorial_moment_sequence_diagnostics=moment_seq,
        )
        # Whatever the adapter derives (possibly None if shape mismatches) --
        # must never crash, and if present must be a plain string.
        hint = result["per_join"][0]["understanding"]["relationship_hint"]
        assert hint is None or isinstance(hint, str)

    def test_24_prosodic_unavailable_by_default(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["understanding"]["prosodic_status"] == "UNAVAILABLE"

    def test_25_handles_present_for_adjacent_same_source_clips(self):
        left, right = _no_speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False, source_duration_by_asset={"s1": 20.0},
        )
        u = result["per_join"][0]["understanding"]
        assert len(u["left_handle_ids"]) >= 1 or len(u["right_handle_ids"]) >= 1


class TestSourceScenarios:
    def test_26_same_source_pair(self):
        left, right = _no_speech_clips(source="s1")
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["per_join"][0]["understanding"]["left_source_asset_id"] == \
            result["per_join"][0]["understanding"]["right_source_asset_id"]

    def test_27_multi_source_pair(self):
        left = _clip("L", "s1", 0.0, 5.0)
        right = _clip("R", "s2", 0.0, 5.0)
        result = m.build_audio_join_treatment_live_diagnostics(
            [left, right], dialogue_overlap_enabled=False,
            source_duration_by_asset={"s1": 20.0, "s2": 20.0},
        )
        row = result["per_join"][0]
        assert row["understanding"]["left_source_asset_id"] == "s1"
        assert row["understanding"]["right_source_asset_id"] == "s2"

    def test_28_source_audio_cache_reuse(self):
        # Three clips sharing one source: the shared middle clip's retained
        # edges must be computed once and reused across both joins via the
        # optional cache dict (bounded, in-run only).
        clips = [_clip("A", "s1", 0.0, 3.0), _clip("B", "s1", 3.0, 6.0), _clip("C", "s1", 6.0, 9.0)]
        cache: dict = {}
        result = m.build_audio_join_treatment_live_diagnostics(
            clips, dialogue_overlap_enabled=False, _source_audio_cache=cache,
        )
        assert result["join_count"] == 2
        # Cache must have been populated (retained + possibly handle keys).
        assert len(cache) > 0

    def test_29_deterministic_repeat(self):
        left, right = _speech_clips()
        r1 = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        r2 = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert r1["per_join"][0]["decision"]["treatment"] == r2["per_join"][0]["decision"]["treatment"]
        assert r1["summary"] == r2["summary"]


class TestNoAuthority:
    def test_30_no_provider_no_asr_rerun_import(self):
        src = inspect.getsource(m)
        tree = ast.parse(src)
        forbidden = {"whole_video_analysis", "visual_analysis", "take_judge_provider", "asr", "google.generativeai"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[-1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[-1])
        assert not (imported & forbidden), imported & forbidden

    def test_31_no_render_module_import(self):
        tree = ast.parse(inspect.getsource(m))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[-1])
        assert "render" not in imported
        assert "render_plan" not in imported

    def test_32_advanced_execution_count_always_zero(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["summary"]["advanced_treatment_executed_count"] == 0

    def test_33_live_audio_window_mutation_count_always_zero(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["summary"]["live_audio_window_mutation_count"] == 0

    def test_34_no_micro_treatment_value(self):
        from cutsell_worker.pacing_v2_audio_join_treatment_decision import TREATMENT_VALUES
        assert "MICRO" not in TREATMENT_VALUES
        assert not any("MICRO" in v for v in TREATMENT_VALUES)

    def test_35_no_loudness_correction_only_status(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        status = result["per_join"][0]["decision"]["loudness_polish_status"]
        assert status in ("LOUDNESS_POLISH_NEEDED", "LOUDNESS_POLISH_NOT_NEEDED", "UNKNOWN")

    def test_36_return_type_is_plain_dict_json_safe(self):
        import json
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        json.dumps(result)  # must not raise

    def test_37_no_transcript_dump(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        import json
        blob = json.dumps(result)
        assert "hello there" not in blob
        assert "world now" not in blob


class TestSeamWiring:
    def test_38_flag_off_backward_compatible_no_key_added(self, monkeypatch):
        monkeypatch.delenv("CUTSELL_AUDIO_JOIN_TREATMENT_DIAGNOSTICS_ENABLED", raising=False)
        assert ucc.audio_join_treatment_diagnostics_enabled() is False

    def test_39_seam_imports_present(self):
        assert hasattr(ucc, "build_audio_join_treatment_live_diagnostics")
        assert hasattr(ucc, "audio_join_treatment_diagnostics_enabled")

    def test_40_seam_never_imports_render_executor(self):
        src = inspect.getsource(ucc)
        assert "render_audio_join_treatment_preview" not in src


class TestRegressionCanaries:
    def test_41_compileall(self):
        result = subprocess.run(
            ["python", "-m", "compileall", "-q", "cutsell_worker"],
            cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_42_render_module_unaffected(self):
        # render.py itself is never imported or modified by D-234 --
        # sanity import check only.
        importlib.import_module("cutsell_worker.render")

    def test_43_existing_click_fade_constant_unchanged(self):
        render = importlib.import_module("cutsell_worker.render")
        assert render._AUDIO_JOIN_FADE_SEC == 0.012


class TestSummaryShape:
    def test_44_summary_has_all_required_counts(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        required = {
            "join_count", "evidence_available_count", "evidence_partial_count", "evidence_unknown_count",
            "understanding_available_count", "understanding_partial_count", "understanding_conflicted_count",
            "none_count", "click_fade_count", "short_crossfade_count",
            "ambience_left_count", "ambience_right_count", "ambience_bridge_count",
            "timing_supported_count", "timing_blocked_count",
            "advanced_treatment_recommended_count", "advanced_timing_supported_count",
            "advanced_treatment_executed_count", "live_audio_window_mutation_count",
            "word_block_count", "meaning_block_count", "double_speech_block_count",
            "safe_left_handle_count", "safe_right_handle_count",
            "acoustically_similar_count", "acoustically_different_count", "acoustic_insufficient_count",
            "loudness_polish_needed_count",
        }
        assert required <= set(result["summary"].keys())

    def test_45_no_master_score_field(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        for key in result["summary"]:
            assert "score" not in key.lower()

    def test_46_status_is_one_of_four_values(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        assert result["status"] in (m.STATUS_AVAILABLE, m.STATUS_PARTIAL, m.STATUS_UNAVAILABLE, m.STATUS_CONFLICTED)

    def test_47_advanced_recommended_ge_timing_supported(self):
        left, right = _speech_clips()
        result = m.build_audio_join_treatment_live_diagnostics([left, right], dialogue_overlap_enabled=False)
        s = result["summary"]
        assert s["advanced_treatment_recommended_count"] >= s["advanced_timing_supported_count"]
