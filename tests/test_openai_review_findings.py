import logging
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
from worker.models.openai_provider import BoundaryResult, TakeJudgeResult, Verdict


def clip(clip_id="a", start=0.0, end=6.0):
    return {"id": clip_id, "start": start, "end": end, "text": "sample", "slot": "HOOK", "meta": {"keep": True}, "source_index": 0}


@pytest.mark.parametrize("operation", ["bad_take", "boundary", "take_judge"])
def test_missing_key_prevents_visual_work_and_provider_calls(monkeypatch, caplog, tmp_path, operation):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extracted = []
    called = []
    monkeypatch.setattr(pipeline, "grab_frame_at_timestamp", lambda *args: extracted.append(args) or True)
    monkeypatch.setattr(pipeline, "detect_bad_take", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(pipeline, "refine_boundaries", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(pipeline, "judge_takes", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(pipeline, "BOUNDARY_REFINER_ENABLED", True)
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", True)
    items = [clip("a"), clip("b", 0.2, 6.2)]

    with caplog.at_level(logging.WARNING, logger="editdna.pipeline"):
        if operation == "bad_take":
            result = pipeline.reject_visual_bad_takes(items, str(tmp_path), "input.mp4")
        elif operation == "boundary":
            result = pipeline.refine_clip_boundaries_with_vision("input.mp4", str(tmp_path), items)
        else:
            monkeypatch.setattr(pipeline, "find_sibling_groups", lambda _clips: [items])
            result = pipeline.run_take_judge(items, str(tmp_path), "input.mp4")

    assert result in (None, False)
    assert extracted == []
    assert called == []
    assert len([record for record in caplog.records if record.levelno >= logging.WARNING]) == 1
    assert all(item["meta"]["keep"] for item in items)


@pytest.mark.parametrize("operation", ["bad_take", "boundary", "take_judge"])
def test_configured_visual_operations_reach_extraction_and_provider(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("OPENAI_API_KEY", "configured-test-key")
    extracted = []
    called = []

    def extract(_input, _timestamp, output):
        extracted.append(output)
        with open(output, "wb") as frame:
            frame.write(b"image")
        return True

    monkeypatch.setattr(pipeline, "grab_frame_at_timestamp", extract)
    monkeypatch.setattr(pipeline, "BOUNDARY_REFINER_ENABLED", True)
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", True)
    items = [clip("a"), clip("b", 0.2, 6.2)]

    if operation == "bad_take":
        monkeypatch.setattr(pipeline, "detect_bad_take", lambda *args, **kwargs: called.append(args) or Verdict.GOOD)
        pipeline.reject_visual_bad_takes(items, str(tmp_path), "input.mp4")
    elif operation == "boundary":
        monkeypatch.setattr(pipeline, "refine_boundaries", lambda *args, **kwargs: called.append(args) or BoundaryResult(Verdict.GOOD, Verdict.GOOD, Verdict.GOOD))
        pipeline.refine_clip_boundaries_with_vision("input.mp4", str(tmp_path), items)
    else:
        monkeypatch.setattr(pipeline, "find_sibling_groups", lambda _clips: [items])
        monkeypatch.setattr(pipeline, "judge_takes", lambda *args, **kwargs: called.append(args) or TakeJudgeResult("a", {"a": 1.0, "b": 0.5}))
        pipeline.run_take_judge(items, str(tmp_path), "input.mp4")

    assert extracted
    assert called
