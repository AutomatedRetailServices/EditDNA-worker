"""Contract tests for the RunPod serverless `focused` compact return value.

RAW #114 (Unified Selection benchmark) exposed a return-contract mismatch:
`run_single_universal_clean_cut_validation()` already computes Unified Selection
reasoner diagnostics, but `serverless_handler._focused()` discarded them before
returning the compact RunPod output that GitHub Actions inspects. These tests
lock the compact contract so that regression is caught before a paid RAW run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    if name in sys.modules:
        return
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module


# serverless_handler.py imports `runpod` unconditionally at module scope for
# the RunPod SDK, which is only available inside the GPU worker container
# image, not in the API/CI dependency set. Stub it so `_focused()` can be
# exercised without a real RunPod environment.
_stub_missing_module("runpod", serverless=types.SimpleNamespace(start=lambda *_a, **_k: None))

from cutsell_worker import serverless_handler  # noqa: E402


FULL_VALIDATION_RESULT = {
    "schema_version": "test-schema",
    "brain_backend": "runpod_local",
    "external_brain_calls_enabled": True,
    "selection_reasoner_enabled": True,
    "selection_reasoner_status": "applied",
    "selection_reasoner_provider": "google",
    "selection_reasoner_model": "gemini-3.5-flash-lite",
    "hybrid_requested_group_count": 4,
    "selected_count": 7,
    "alternate_count": 3,
    "discarded_count": 2,
    "selected_duration_sec": 141.667,
    "elapsed_sec": 12.5,
    # A provider key must never be echoed back to the compact RunPod output,
    # even if it were ever accidentally present on the full result dict.
    "gemini_api_key": "sk-should-not-leak",
}


@pytest.fixture
def patched_focused(monkeypatch, tmp_path):
    """Patch out real ASR/render/S3 work so `_focused()` runs in-process."""

    def fake_validation(source_key, *, project_id, preview_output):
        assert source_key == "some/source.mp4"
        return dict(FULL_VALIDATION_RESULT)

    uploaded = {}

    def fake_upload_artifact(local_path, *, key, content_type):
        uploaded[key] = local_path
        return f"s3://fake-bucket/{key}"

    monkeypatch.setattr(serverless_handler, "run_single_universal_clean_cut_validation", fake_validation)
    monkeypatch.setattr(serverless_handler, "_upload_artifact", fake_upload_artifact)

    work_dir = tmp_path / "cutsell-serverless"

    def fake_path(value):
        if str(value) == "/tmp/cutsell-serverless":
            return work_dir
        return serverless_handler.Path(value)

    monkeypatch.setattr(serverless_handler, "Path", fake_path)
    return uploaded


def test_focused_exposes_unified_selection_reasoner_state(patched_focused):
    out = serverless_handler._focused({
        "source_key": "some/source.mp4",
        "benchmark_id": "video00-unified-selection-test",
    })

    assert out["selection_reasoner_enabled"] is True
    assert out["selection_reasoner_status"] == "applied"
    assert out["selection_reasoner_provider"] == "google"
    assert out["selection_reasoner_model"] == "gemini-3.5-flash-lite"
    assert out["hybrid_requested_group_count"] == 4


def test_focused_compact_state_matches_full_validation_result(patched_focused):
    out = serverless_handler._focused({"source_key": "some/source.mp4"})

    for key in (
        "selection_reasoner_enabled",
        "selection_reasoner_status",
        "selection_reasoner_provider",
        "selection_reasoner_model",
        "hybrid_requested_group_count",
        "selected_count",
        "alternate_count",
        "discarded_count",
        "selected_duration_sec",
        "elapsed_sec",
    ):
        assert out[key] == FULL_VALIDATION_RESULT[key], key


def test_focused_disabled_reasoner_state_remains_observable(patched_focused, monkeypatch):
    disabled_result = dict(FULL_VALIDATION_RESULT)
    disabled_result.update(
        selection_reasoner_enabled=False,
        selection_reasoner_status=None,
        selection_reasoner_provider=None,
        selection_reasoner_model=None,
        hybrid_requested_group_count=0,
    )
    monkeypatch.setattr(
        serverless_handler,
        "run_single_universal_clean_cut_validation",
        lambda *a, **k: dict(disabled_result),
    )

    out = serverless_handler._focused({"source_key": "some/source.mp4"})

    # A falsy/None reasoner state must still come through explicitly, not be
    # silently dropped or defaulted to a value that hides the real state.
    assert out["selection_reasoner_enabled"] is False
    assert out["selection_reasoner_status"] is None
    assert out["selection_reasoner_provider"] is None
    assert out["selection_reasoner_model"] is None
    assert out["hybrid_requested_group_count"] == 0


def test_focused_never_leaks_secret_looking_fields(patched_focused):
    out = serverless_handler._focused({"source_key": "some/source.mp4"})

    serialized = json.dumps(out)
    assert "sk-should-not-leak" not in serialized
    assert "gemini_api_key" not in out
