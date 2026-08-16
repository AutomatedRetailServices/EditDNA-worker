from __future__ import annotations

import pytest

from cutsell_worker.brain_runtime import RUNPOD_LOCAL_BACKEND, build_brain_runtime
from cutsell_worker.config import load_runtime_config
from cutsell_worker.contracts import SourceAsset, TranscriptSegment
from cutsell_worker.whole_video_local import RunPodLocalWholeVideoProvider


def test_runpod_local_brain_never_activates_external_models_from_key_presence():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "OPENAI_API_KEY": "present-but-must-not-be-used",
        "GEMINI_API_KEY": "also-present-but-not-enabled",
        "CUTSELL_CLEAN_CUT_JUDGE": "1",
    }
    config = load_runtime_config(env)

    assert config.openai_api_key_present is True
    assert config.semantic_ready is False
    assert config.visual_ready is False
    assert config.clean_cut_judge_ready is False

    brain = build_brain_runtime(config, env)
    assert brain.backend == RUNPOD_LOCAL_BACKEND
    assert brain.external_calls_enabled is False
    assert brain.editorial_judge is None
    assert brain.whole_video_provider.__class__.__name__ == "RunPodLocalWholeVideoProvider"
    assert brain.visual_provider is None
    assert brain.take_grouping_provider is None
    assert brain.take_judge_provider is not None
    assert brain.take_judge_provider.__class__.__name__ == "HybridTakeJudgeProvider"
    assert brain.take_judge_provider.editorial_judge is None
    assert brain.clean_cut_provider is None
    assert brain.composer_provider is None
    assert brain.draft_review_provider is None


def test_explicit_hybrid_enable_without_gemini_key_fails_open_to_local():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)
    assert brain.hybrid_settings.enabled is True
    assert brain.editorial_judge is None
    assert brain.external_calls_enabled is False


def test_explicit_hybrid_enable_plus_gemini_key_constructs_flash_lite_without_network():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "GEMINI_API_KEY": "fake-construction-only",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)
    assert brain.external_calls_enabled is True
    assert brain.editorial_judge is not None
    assert brain.editorial_judge.provider_name == "google"
    assert brain.editorial_judge.model_name == "gemini-3.5-flash-lite"
    assert brain.hybrid_settings.max_cost_per_edit_usd == 0.0075
    # Best Take wrapper stays local so semantic cleanup cannot charge twice per group.
    assert brain.take_judge_provider.editorial_judge is None


def test_runpod_local_whole_video_context_keeps_source_shell_for_local_events():
    source = SourceAsset(
        source_asset_id="src-1",
        project_id="project-1",
        user_id="user-1",
        original_name="raw.mp4",
        source_order=0,
        duration_sec=12.0,
        uri="s3://bucket/raw.mp4",
    )
    transcript = TranscriptSegment(
        source_asset_id="src-1",
        start=0.0,
        end=2.0,
        text="This is a local transcript for the creator recording.",
    )

    result = RunPodLocalWholeVideoProvider().analyze((source,), (transcript,), ())

    assert result.status.status == "applied"
    assert result.status.provider == "runpod_local_whole_video"
    assert len(result.sources) == 1
    context = result.sources[0]
    assert context.source_asset_id == "src-1"
    assert context.edit_mode == "natural"
    assert "local transcript" in context.summary
    assert context.events == ()


def test_non_runpod_brain_backend_is_rejected():
    config = load_runtime_config({"CUTSELL_BRAIN_BACKEND": "external_openai"})
    with pytest.raises(RuntimeError, match="only 'runpod_local' is permitted"):
        build_brain_runtime(config)
