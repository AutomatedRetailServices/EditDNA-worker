import copy
import importlib.util
import os
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    if name in sys.modules:
        return
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module


_stub_missing_module("requests")
_stub_missing_module("boto3")
_stub_missing_module("clip")
_stub_missing_module("faster_whisper", WhisperModel=object)

from worker import pipeline


def _clip(clip_id="chosen", source_index=0):
    return {
        "id": clip_id,
        "start": 0.0,
        "end": 1.0,
        "text": clip_id,
        "slot": "STORY",
        "semantic_score": 1.0,
        "source_index": source_index,
        "meta": {"keep": True},
    }


def test_all_valid_selected_ids_render_and_duplicates_are_deduplicated(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, "has_audio_stream", lambda _path: False)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    pipeline.render_funnel_video(
        "source.mp4", str(tmp_path), [_clip("a"), _clip("b")], ["a", "b", "a"]
    )

    assert len(calls) == 1
    assert "concat=n=2:v=1:a=0" in calls[0][calls[0].index("-filter_complex") + 1]


@pytest.mark.parametrize(
    "selected, missing",
    [(["known", "missing"], ["missing"]), (["one", "known", "two"], ["one", "two"]),
    ],
)
def test_missing_selected_ids_fail_before_render(monkeypatch, tmp_path, selected, missing):
    render_started = []
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: render_started.append(True),
    )

    with pytest.raises(pipeline.MissingSelectedClipsError) as caught:
        pipeline.render_funnel_video("source.mp4", str(tmp_path), [_clip("known")], selected)

    assert all(clip_id in str(caught.value) for clip_id in missing)
    assert render_started == []


@pytest.fixture
def configured_pipeline(monkeypatch, tmp_path):
    state = {"render": 0, "upload": 0, "job_dirs": []}
    clips = [_clip()]

    def session_dir(_session_id):
        path = tmp_path / f"job-{len(state['job_dirs'])}"
        path.mkdir()
        state["job_dirs"].append(path)
        return str(path)

    def download(_url, path):
        with open(path, "wb") as output:
            output.write(b"source")

    def render(_sources, directory, _clips, _ids):
        state["render"] += 1
        output = os.path.join(directory, "final.mp4")
        with open(output, "wb") as rendered:
            rendered.write(b"render")
        return output

    monkeypatch.setattr(pipeline, "ensure_session_dir", session_dir)
    monkeypatch.setattr(pipeline, "download_to_local", download)
    monkeypatch.setattr(pipeline, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(pipeline, "run_asr", lambda _path: [{}])
    monkeypatch.setattr(pipeline, "sentence_boundary_micro_cuts", lambda _asr: copy.deepcopy(clips))
    monkeypatch.setattr(pipeline, "merge_incomplete_phrases", lambda value: value)
    monkeypatch.setattr(pipeline, "enrich_clips_semantic", lambda _clips: False)
    monkeypatch.setattr(pipeline, "dedupe_clips", lambda value: value)
    monkeypatch.setattr(pipeline, "run_visual_pass", lambda *_args: False)
    monkeypatch.setattr(pipeline, "build_slots_dict", lambda _clips: {})
    monkeypatch.setattr(pipeline, "build_composer", lambda _clips, mode: {"mode": mode, "used_clip_ids": ["chosen"]})
    monkeypatch.setattr(pipeline, "pretty_print_composer", lambda *_args: "diagnostic")
    monkeypatch.setattr(pipeline, "render_funnel_video", render)
    monkeypatch.setattr(pipeline, "save_result_json_to_s3", lambda _result: None)
    monkeypatch.setattr(pipeline, "S3_BUCKET", "output-bucket")

    def upload(*_args):
        state["upload"] += 1
        return "https://output.example/final.mp4"

    monkeypatch.setattr(pipeline, "upload_to_s3", upload)
    return state


def test_successful_upload_returns_url_and_cleans_job_files(configured_pipeline):
    result = pipeline.run_pipeline("session", files=["signed input"], mode="human")

    assert result["ok"] is True
    assert result["output_video_url"] == "https://output.example/final.mp4"
    assert configured_pipeline["upload"] == 1
    assert not configured_pipeline["job_dirs"][0].exists()
    assert not {
        "input_local",
        "input_files_local",
        "output_video_local",
        "clean_cut_output_video_local",
    } & result.keys()
    assert all("source_local" not in clip for clip in result["clips"])


def test_local_only_success_transfers_directory_ownership_to_caller(
    configured_pipeline, monkeypatch
):
    monkeypatch.setattr(pipeline, "S3_BUCKET", None)

    result = pipeline.run_pipeline("session", files=["input"], mode="human")

    assert result["ok"] is True
    assert result["output_video_url"] is None
    assert os.path.exists(result["output_video_local"])
    assert all(os.path.exists(path) for path in result["input_files_local"])
    assert os.path.exists(result["input_local"])
    assert os.path.exists(result["clips"][0]["source_local"])
    assert configured_pipeline["job_dirs"][0].exists()


@pytest.mark.parametrize("source_count", [1, 2])
def test_empty_human_selection_fails_without_source_zero_fallback(
    configured_pipeline, monkeypatch, source_count
):
    monkeypatch.setattr(pipeline, "build_composer", lambda _clips, mode: {"mode": mode, "used_clip_ids": []})

    with pytest.raises(pipeline.SelectionError, match="No clips.*human"):
        pipeline.run_pipeline("session", files=["input"] * source_count, mode="human")

    assert configured_pipeline["render"] == configured_pipeline["upload"] == 0
    assert not configured_pipeline["job_dirs"][0].exists()


def test_empty_blooper_selection_fails(configured_pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "build_composer", lambda _clips, mode: {"mode": mode, "used_clip_ids": []})
    with pytest.raises(pipeline.SelectionError, match="No clips.*blooper"):
        pipeline.run_pipeline("session", files=["input"], mode="blooper")


def test_clean_empty_selection_keeps_legacy_behavior(configured_pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "select_clean_cut_clip_ids", lambda _clips: [])
    result = pipeline.run_pipeline("session", files=["input"], mode="clean")
    assert result["ok"] is True
    assert "clean_cut_output_video_local" not in result
    assert configured_pipeline["render"] == 0


def test_upload_exception_fails_and_cleans(configured_pipeline, monkeypatch):
    provider_error = OSError("provider unavailable")
    def failed_upload(*_args):
        raise pipeline.UploadError("upload failed") from provider_error

    monkeypatch.setattr(pipeline, "upload_to_s3", failed_upload)

    with pytest.raises(pipeline.UploadError) as caught:
        pipeline.run_pipeline("session", files=["input"], mode="human")

    assert caught.value.__cause__ is provider_error
    assert not configured_pipeline["job_dirs"][0].exists()


def test_empty_upload_url_is_not_success(configured_pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "upload_to_s3", lambda *_args: "")
    with pytest.raises(pipeline.UploadError, match="no output URL"):
        pipeline.run_pipeline("session", files=["input"], mode="human")


def test_s3_provider_exception_is_wrapped_with_original_cause(monkeypatch):
    provider_error = OSError("provider unavailable")

    class Client:
        def upload_file(self, *_args):
            raise provider_error

    monkeypatch.setattr(pipeline.boto3, "client", lambda _service: Client(), raising=False)
    with pytest.raises(pipeline.UploadError) as caught:
        pipeline.upload_to_s3("final.mp4", "bucket", "key")
    assert caught.value.__cause__ is provider_error


def test_render_failure_cleans(configured_pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline, "render_funnel_video", lambda *_args: (_ for _ in ()).throw(RuntimeError("ffmpeg failed"))
    )
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        pipeline.run_pipeline("session", files=["input"], mode="human")
    assert not configured_pipeline["job_dirs"][0].exists()


def test_cleanup_failure_does_not_replace_original_error(configured_pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline, "render_funnel_video", lambda *_args: (_ for _ in ()).throw(RuntimeError("original render error"))
    )
    monkeypatch.setattr(pipeline.shutil, "rmtree", lambda _path: (_ for _ in ()).throw(OSError("cleanup error")))
    with pytest.raises(RuntimeError, match="original render error"):
        pipeline.run_pipeline("session", files=["input"], mode="human")
