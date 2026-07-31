import copy
import importlib.util
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module


_stub_missing_module("requests")
_stub_missing_module("boto3")
_stub_missing_module("clip")
_stub_missing_module("faster_whisper", WhisperModel=object)

from worker import pipeline


def clip(clip_id, start, *, keep=True, slot="STORY", semantic_score=0.0):
    return {
        "id": clip_id,
        "start": start,
        "end": start + 1.0,
        "text": clip_id,
        "slot": slot,
        "semantic_score": semantic_score,
        "meta": {"keep": keep},
    }


def test_selector_includes_story_and_non_sales_clips():
    clips = [
        clip("story", 0, slot="STORY"),
        clip("other", 1, slot="OTHER"),
    ]

    assert pipeline.select_clean_cut_clip_ids(clips) == ["story", "other"]


def test_selector_includes_low_semantic_clips():
    clips = [clip("low", 0, semantic_score=0.01)]

    assert pipeline.select_clean_cut_clip_ids(clips) == ["low"]


def test_selector_returns_source_timestamp_order():
    clips = [clip("late", 9), clip("early", 1), clip("middle", 4)]

    assert pipeline.select_clean_cut_clip_ids(clips) == ["early", "middle", "late"]


def test_selector_excludes_clips_canonically_marked_not_keep():
    clips = [clip("discard", 0, keep=False), clip("keep", 1)]

    assert pipeline.select_clean_cut_clip_ids(clips) == ["keep"]


def test_selector_does_not_mutate_analyzed_clips():
    clips = [clip("late", 2), clip("early", 1, semantic_score=0.2)]
    before = copy.deepcopy(clips)

    pipeline.select_clean_cut_clip_ids(clips)

    assert clips == before


@pytest.fixture
def pipeline_run(monkeypatch, tmp_path):
    analyzed_clips = [
        clip("story-low", 2, slot="STORY", semantic_score=0.01),
        clip("sales", 1, slot="HOOK", semantic_score=0.99),
        clip("discard", 0, keep=False, slot="CTA", semantic_score=1.0),
    ]
    calls = {"asr": 0, "semantic": 0, "vision": 0, "composer": 0}
    rendered = []
    composer = {"used_clip_ids": ["composer-only"], "diagnostic": True}

    monkeypatch.setattr(pipeline, "S3_BUCKET", None)
    monkeypatch.setattr(pipeline, "ensure_session_dir", lambda _session_id: str(tmp_path))
    monkeypatch.setattr(pipeline, "download_to_local", lambda _url, _path: None)
    monkeypatch.setattr(pipeline, "probe_duration", lambda _path: 3.0)

    def run_asr(_path):
        calls["asr"] += 1
        return [{"text": "transcript"}]

    monkeypatch.setattr(pipeline, "run_asr", run_asr)
    monkeypatch.setattr(
        pipeline, "sentence_boundary_micro_cuts", lambda _segments: analyzed_clips
    )
    monkeypatch.setattr(pipeline, "merge_incomplete_phrases", lambda clips: clips)

    def enrich(clips):
        calls["semantic"] += 1
        return True

    monkeypatch.setattr(pipeline, "enrich_clips_semantic", enrich)
    monkeypatch.setattr(pipeline, "dedupe_clips", lambda clips: clips)

    def vision(_input, _session_dir, _clips):
        calls["vision"] += 1
        return True

    monkeypatch.setattr(pipeline, "run_visual_pass", vision)
    monkeypatch.setattr(pipeline, "build_slots_dict", lambda _clips: {})

    def build_composer(_clips, mode):
        calls["composer"] += 1
        return {**composer, "mode": mode}

    monkeypatch.setattr(pipeline, "build_composer", build_composer)
    monkeypatch.setattr(pipeline, "pretty_print_composer", lambda *_args: "diagnostics")

    def render(_input, _session_dir, _clips, used_ids):
        rendered.append(list(used_ids))
        return str(tmp_path / "final.mp4")

    monkeypatch.setattr(pipeline, "render_funnel_video", render)
    monkeypatch.setattr(pipeline, "save_result_json_to_s3", lambda _result: None)

    def run(mode):
        return pipeline.run_pipeline("session", files=["https://example.test/video"], mode=mode)

    return run, calls, rendered, composer


def test_clean_mode_renders_clean_cut_ids_instead_of_composer_ids(pipeline_run):
    run, _calls, rendered, _composer = pipeline_run

    result = run("clean")

    assert rendered == [["sales", "story-low"]]
    assert result["clean_cut_used_clip_ids"] == ["sales", "story-low"]


def test_clean_mode_still_computes_and_retains_composer_diagnostics(pipeline_run):
    run, calls, _rendered, composer = pipeline_run

    result = run("clean")

    assert calls["composer"] == 1
    assert result["composer"] == {**composer, "mode": "clean"}
    assert result["composer_human"] == "diagnostics"


def test_human_mode_continues_rendering_composer_ids(pipeline_run):
    run, _calls, rendered, _composer = pipeline_run

    result = run("human")

    assert rendered == [["composer-only"]]
    assert result["clean_cut_used_clip_ids"] == []


def test_clean_mode_uses_single_shared_analysis_pass(pipeline_run):
    run, calls, _rendered, _composer = pipeline_run

    run("clean")

    assert calls["asr"] == 1
    assert calls["semantic"] == 1
    assert calls["vision"] == 1


@pytest.mark.parametrize("mode", ["clean", "human", "blooper"])
def test_clean_cut_result_fields_are_backward_compatible(pipeline_run, mode):
    run, _calls, _rendered, _composer = pipeline_run

    result = run(mode)

    assert "clean_cut_used_clip_ids" in result
    assert "clean_cut_output_video_local" in result
    assert "clean_cut_output_video_url" in result
    if mode == "clean":
        assert result["clean_cut_output_video_local"] == result["output_video_local"]
        assert result["clean_cut_output_video_url"] == result["output_video_url"]
    else:
        assert result["clean_cut_used_clip_ids"] == []
        assert result["clean_cut_output_video_local"] is None
        assert result["clean_cut_output_video_url"] is None
