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
from pathlib import Path

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


# ---------------------------------------------------------------------------
# D-036 items 6/7: the authoritative delivery gate -- a QC-invalidated
# candidate must never be surfaced/uploaded as if it were deliverable.
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_focused_with_real_preview_file(monkeypatch, tmp_path):
    """Like `patched_focused`, but the fake validation actually writes a
    (tiny, fake) preview file to disk, matching the real order: the render
    physically happens before `_focused` ever inspects `live_render_qc`."""
    uploaded = {}

    def fake_upload_artifact(local_path, *, key, content_type):
        uploaded[key] = local_path
        return f"s3://fake-bucket/{key}"

    monkeypatch.setattr(serverless_handler, "_upload_artifact", fake_upload_artifact)
    # The fake preview below is not a real, decodable MP4 -- avoid a real
    # ffprobe dependency in this contract test by stubbing duration probing.
    monkeypatch.setattr(serverless_handler, "_probe_duration", lambda path: 1.5)

    work_dir = tmp_path / "cutsell-serverless"

    def fake_path(value):
        if str(value) == "/tmp/cutsell-serverless":
            return work_dir
        return serverless_handler.Path(value)

    monkeypatch.setattr(serverless_handler, "Path", fake_path)

    def make_fake_validation(live_render_qc: dict):
        def fake_validation(source_key, *, project_id, preview_output):
            work_dir.mkdir(parents=True, exist_ok=True)
            Path(preview_output).write_bytes(b"fake-mp4-bytes")
            return {
                **FULL_VALIDATION_RESULT,
                "preview_path": preview_output if live_render_qc.get("deliverable") else None,
                "preview_skipped_reason": None if live_render_qc.get("deliverable") else "post_render_qc_semantic_mismatch_invalidated",
                "live_render_qc": live_render_qc,
            }
        return fake_validation

    return uploaded, make_fake_validation


def test_qc_invalidated_candidate_is_not_uploaded_as_final_deliverable(monkeypatch, patched_focused_with_real_preview_file):
    uploaded, make_fake_validation = patched_focused_with_real_preview_file
    monkeypatch.setattr(
        serverless_handler, "run_single_universal_clean_cut_validation",
        make_fake_validation({
            "status": "SEMANTIC_MISMATCH_INVALIDATED", "deliverable": False,
            "delivery_status": "NOT_DELIVERABLE_SEMANTIC_MISMATCH_INVALIDATED",
            "output_path": None, "plan_id": "plan_x", "plan_version": 1,
            "semantic_hash": "hash_x", "render_attempt_count": 1, "attempts": [],
        }),
    )

    out = serverless_handler._focused({"source_key": "some/source.mp4", "benchmark_id": "invalid-run"})

    assert out["deliverable"] is False
    assert out["delivery_status"] == "NOT_DELIVERABLE_SEMANTIC_MISMATCH_INVALIDATED"
    assert out["preview_uri"] is None
    assert not any(key.endswith("/preview.mp4") for key in uploaded)
    assert any(key.endswith("diagnostic-invalidated-preview.mp4") for key in uploaded)


def test_diagnostic_invalidated_render_is_preserved_with_explicit_invalid_status(monkeypatch, patched_focused_with_real_preview_file):
    uploaded, make_fake_validation = patched_focused_with_real_preview_file
    monkeypatch.setattr(
        serverless_handler, "run_single_universal_clean_cut_validation",
        make_fake_validation({
            "status": "NEEDS_HUMAN_REVIEW", "deliverable": False,
            "delivery_status": "NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW",
            "output_path": None, "plan_id": "plan_x", "plan_version": 1,
            "semantic_hash": "hash_x", "render_attempt_count": 3, "attempts": [],
        }),
    )

    out = serverless_handler._focused({"source_key": "some/source.mp4", "benchmark_id": "needs-review-run"})

    assert out["diagnostic_preview_uri"] is not None
    assert "diagnostic-invalidated-preview.mp4" in out["diagnostic_preview_uri"]
    assert out["preview_uri"] is None
    assert out["deliverable"] is False


def test_qc_pass_candidate_is_uploaded_normally(monkeypatch, patched_focused_with_real_preview_file):
    uploaded, make_fake_validation = patched_focused_with_real_preview_file
    monkeypatch.setattr(
        serverless_handler, "run_single_universal_clean_cut_validation",
        make_fake_validation({
            "status": "PASS", "deliverable": True, "delivery_status": "DELIVERABLE",
            "output_path": "/tmp/out.mp4", "plan_id": "plan_ok", "plan_version": 2,
            "semantic_hash": "hash_ok", "render_attempt_count": 1, "attempts": [],
        }),
    )

    out = serverless_handler._focused({"source_key": "some/source.mp4", "benchmark_id": "pass-run"})

    assert out["deliverable"] is True
    assert out["delivery_status"] == "DELIVERABLE"
    assert out["preview_uri"] is not None
    assert out["preview_uri"].endswith("preview.mp4")
    assert out["diagnostic_preview_uri"] is None


def test_final_delivery_metadata_references_exact_plan_id_version_hash(monkeypatch, patched_focused_with_real_preview_file):
    uploaded, make_fake_validation = patched_focused_with_real_preview_file
    monkeypatch.setattr(
        serverless_handler, "run_single_universal_clean_cut_validation",
        make_fake_validation({
            "status": "PASS", "deliverable": True, "delivery_status": "DELIVERABLE",
            "output_path": "/tmp/out.mp4", "plan_id": "plan_exact_123", "plan_version": 5,
            "semantic_hash": "hash_exact_456", "render_attempt_count": 2, "attempts": [],
        }),
    )

    out = serverless_handler._focused({"source_key": "some/source.mp4", "benchmark_id": "metadata-run"})

    assert out["live_render_qc_plan_id"] == "plan_exact_123"
    assert out["live_render_qc_plan_version"] == 5
    assert out["live_render_qc_semantic_hash"] == "hash_exact_456"
    assert out["live_render_qc_render_attempt_count"] == 2
