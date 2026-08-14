from __future__ import annotations

import pytest

from cutsell_worker.brain_runtime import RUNPOD_LOCAL_BACKEND, build_brain_runtime
from cutsell_worker.config import load_runtime_config
from cutsell_worker.contracts import SourceAsset, TranscriptSegment
from cutsell_worker.whole_video_local import RunPodLocalWholeVideoProvider


def test_runpod_local_brain_never_activates_openai_from_key_presence():
    config = load_runtime_config({
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "OPENAI_API_KEY": "present-but-must-not-be-used",
        "CUTSELL_CLEAN_CUT_JUDGE": "1",
    })

    assert config.openai_api_key_present is True
    assert config.semantic_ready is False
    assert config.visual_ready is False
    assert config.clean_cut_judge_ready is False

    brain = build_brain_runtime(config)
    assert brain.backend == RUNPOD_LOCAL_BACKEND
    assert brain.external_calls_enabled is False
    assert brain.whole_video_provider.__class__.__name__ == "RunPodLocalWholeVideoProvider"
    assert brain.visual_provider is None
    assert brain.take_grouping_provider is None
    assert brain.take_judge_provider is None
    assert brain.clean_cut_provider is None
    assert brain.composer_provider is None
    assert brain.draft_review_provider is None


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
