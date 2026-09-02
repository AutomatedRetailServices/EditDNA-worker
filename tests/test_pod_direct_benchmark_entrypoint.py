"""D-042 follow-up: direct Pod-execution entrypoint (bypasses HTTP
pod_job_server/port-8080 entirely). Fully mocked -- no subprocess, no
S3, no GPU, no network."""
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


# serverless_handler.py imports the real `runpod` SDK at module level (only
# used under its own `if __name__ == "__main__"` guard) and `boto3` for S3
# uploads -- neither is installed in this dev sandbox. Same stub pattern
# tests/test_cutsell_serverless_run_op_dispatch.py already established.
_stub_missing_module("runpod", serverless=types.SimpleNamespace(start=lambda *_a, **_k: None))

import cutsell_worker.pod_direct_benchmark_entrypoint as entry  # noqa: E402


def _fake_run_capture_factory(results: dict[tuple, tuple[bool, str]]):
    def _fake(cmd, timeout=entry._SUBPROCESS_TIMEOUT_S):
        key = tuple(cmd)
        for pattern, value in results.items():
            if pattern == "torch" and "torch" in cmd[-1]:
                return value
            if pattern == "ffmpeg" and cmd[0] == "ffmpeg":
                return value
            if pattern == "import" and "cutsell_worker" in cmd[-1]:
                return value
        raise AssertionError(f"unscripted command: {cmd}")

    return _fake


# ---------------------------------------------------------------------------
# run_sanity_checks
# ---------------------------------------------------------------------------
def test_run_sanity_checks_all_pass(monkeypatch):
    monkeypatch.setattr(
        entry,
        "_run_capture",
        _fake_run_capture_factory(
            {
                "torch": (True, "True\nNVIDIA GeForce RTX 4090"),
                "ffmpeg": (True, "ffmpeg version 6.0 Copyright (c) 2000-2023"),
                "import": (True, "cutsell_worker import ok"),
            }
        ),
    )
    result = entry.run_sanity_checks()
    assert result["ok"] is True
    assert result["cuda_available"] is True
    assert result["device_name"] == "NVIDIA GeForce RTX 4090"
    assert result["ffmpeg_check_ok"] is True
    assert result["cutsell_import_ok"] is True


def test_run_sanity_checks_cuda_unavailable_fails_overall(monkeypatch):
    monkeypatch.setattr(
        entry,
        "_run_capture",
        _fake_run_capture_factory(
            {
                "torch": (True, "False\nNO_CUDA"),
                "ffmpeg": (True, "ffmpeg version 6.0"),
                "import": (True, "cutsell_worker import ok"),
            }
        ),
    )
    result = entry.run_sanity_checks()
    assert result["ok"] is False
    assert result["cuda_available"] is False
    assert result["torch_check_ok"] is True  # the python call itself succeeded


def test_run_sanity_checks_ffmpeg_missing_fails_overall(monkeypatch):
    monkeypatch.setattr(
        entry,
        "_run_capture",
        _fake_run_capture_factory(
            {
                "torch": (True, "True\nNVIDIA GeForce RTX 4090"),
                "ffmpeg": (False, "exception running ['ffmpeg', '-version']: [Errno 2] No such file or directory"),
                "import": (True, "cutsell_worker import ok"),
            }
        ),
    )
    result = entry.run_sanity_checks()
    assert result["ok"] is False
    assert result["ffmpeg_check_ok"] is False


def test_run_sanity_checks_import_failure_fails_overall(monkeypatch):
    monkeypatch.setattr(
        entry,
        "_run_capture",
        _fake_run_capture_factory(
            {
                "torch": (True, "True\nNVIDIA GeForce RTX 4090"),
                "ffmpeg": (True, "ffmpeg version 6.0"),
                "import": (False, "Traceback ... ModuleNotFoundError"),
            }
        ),
    )
    result = entry.run_sanity_checks()
    assert result["ok"] is False
    assert result["cutsell_import_ok"] is False


def test_run_capture_never_raises_on_missing_binary():
    ok, output = entry._run_capture(["definitely-not-a-real-binary-xyz"])
    assert ok is False
    assert "exception running" in output


# ---------------------------------------------------------------------------
# main() -- payload parsing + dispatch, fully mocked
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_real_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "_WORK_DIR", tmp_path)
    uploaded = []

    def _fake_upload(local_path, *, key, content_type):
        uploaded.append({"local_path": local_path, "key": key, "content_type": content_type})
        return f"s3://fake-bucket/{key}"

    monkeypatch.setattr(entry, "_upload_artifact", _fake_upload)
    return uploaded


def test_main_missing_payload_env_var_fails_fast(monkeypatch, capsys):
    monkeypatch.delenv("CUTSELL_BENCHMARK_PAYLOAD_JSON", raising=False)
    assert entry.main() == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_main_missing_benchmark_id_fails_fast(monkeypatch):
    monkeypatch.setenv("CUTSELL_BENCHMARK_PAYLOAD_JSON", json.dumps({"op": "focused", "source_key": "x"}))
    assert entry.main() == 1


def test_main_sanity_failure_never_calls_run_op(monkeypatch, _no_real_uploads):
    monkeypatch.setenv(
        "CUTSELL_BENCHMARK_PAYLOAD_JSON",
        json.dumps({"op": "focused", "source_key": "x", "benchmark_id": "bench-1"}),
    )
    monkeypatch.setattr(entry, "run_sanity_checks", lambda: {"ok": False, "cuda_available": False})
    called = {"run_op": False}
    monkeypatch.setattr(entry, "run_op", lambda op, payload: called.update(run_op=True) or {"ok": True})

    exit_code = entry.main()

    assert exit_code == 1
    assert called["run_op"] is False
    keys = [u["key"] for u in _no_real_uploads]
    assert "cutsell/serverless/bench-1/sanity_check.json" in keys
    assert not any("run_output.json" in k for k in keys)


def test_main_sanity_pass_calls_run_op_and_uploads_output(monkeypatch, _no_real_uploads):
    payload = {"op": "focused", "source_key": "x", "benchmark_id": "bench-2", "auto_speech_visual_microtrim": True}
    monkeypatch.setenv("CUTSELL_BENCHMARK_PAYLOAD_JSON", json.dumps(payload))
    monkeypatch.setattr(entry, "run_sanity_checks", lambda: {"ok": True, "cuda_available": True})
    captured_calls = []

    def _fake_run_op(op, p):
        captured_calls.append((op, p))
        return {"ok": True, "result_uri": "s3://bucket/result.json", "preview_uri": "s3://bucket/preview.mp4"}

    monkeypatch.setattr(entry, "run_op", _fake_run_op)

    exit_code = entry.main()

    assert exit_code == 0
    assert captured_calls == [("focused", payload)]
    keys = [u["key"] for u in _no_real_uploads]
    assert "cutsell/serverless/bench-2/sanity_check.json" in keys
    assert "cutsell/serverless/bench-2/run_output.json" in keys


def test_main_run_op_exception_uploads_error_report_not_a_crash(monkeypatch, _no_real_uploads):
    monkeypatch.setenv(
        "CUTSELL_BENCHMARK_PAYLOAD_JSON",
        json.dumps({"op": "focused", "source_key": "x", "benchmark_id": "bench-3"}),
    )
    monkeypatch.setattr(entry, "run_sanity_checks", lambda: {"ok": True, "cuda_available": True})

    def _raise(op, payload):
        raise RuntimeError("simulated pipeline crash")

    monkeypatch.setattr(entry, "run_op", _raise)

    exit_code = entry.main()

    assert exit_code == 1
    keys = [u["key"] for u in _no_real_uploads]
    assert "cutsell/serverless/bench-3/pod-execution-error.json" in keys


def test_main_run_op_returns_ok_false_is_a_failure_exit_code(monkeypatch, _no_real_uploads):
    monkeypatch.setenv(
        "CUTSELL_BENCHMARK_PAYLOAD_JSON",
        json.dumps({"op": "focused", "source_key": "x", "benchmark_id": "bench-4"}),
    )
    monkeypatch.setattr(entry, "run_sanity_checks", lambda: {"ok": True, "cuda_available": True})
    monkeypatch.setattr(entry, "run_op", lambda op, payload: {"ok": False, "error": "deliverable check failed"})

    assert entry.main() == 1


def test_main_defaults_op_to_focused_when_absent(monkeypatch, _no_real_uploads):
    monkeypatch.setenv(
        "CUTSELL_BENCHMARK_PAYLOAD_JSON",
        json.dumps({"source_key": "x", "benchmark_id": "bench-5"}),
    )
    monkeypatch.setattr(entry, "run_sanity_checks", lambda: {"ok": True, "cuda_available": True})
    captured = []
    monkeypatch.setattr(entry, "run_op", lambda op, payload: captured.append(op) or {"ok": True})

    entry.main()

    assert captured == ["focused"]
