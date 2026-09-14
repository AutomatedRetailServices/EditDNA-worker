"""D-043 (CutSell Modal GPU execution -- first live validation): pure
diagnostic-logic tests. No `modal` package dependency, no real subprocess
calls to a real GPU -- ffmpeg/ffprobe/torch are all monkeypatched."""
from __future__ import annotations

import json
import types

import pytest

import modal_gpu_diagnostics as diag


def test_run_capture_returns_true_on_success(monkeypatch):
    class _FakeCompletedProcess:
        returncode = 0
        stdout = "ffmpeg version 6.0\n"
        stderr = ""

    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
    ok, out = diag._run_capture(["ffmpeg", "-version"])
    assert ok is True
    assert "ffmpeg version 6.0" in out


def test_run_capture_returns_false_on_nonzero_exit(monkeypatch):
    class _FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "command not found"

    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: _FakeCompletedProcess())
    ok, out = diag._run_capture(["nonexistent-binary"])
    assert ok is False
    assert "command not found" in out


def test_run_capture_never_raises_on_missing_binary(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(diag.subprocess, "run", _raise)
    ok, out = diag._run_capture(["totally-missing-binary"])
    assert ok is False
    assert "no such file" in out


def test_first_line_skips_blank_lines():
    assert diag._first_line("\n\n  ffmpeg version 6.0\nmore stuff\n") == "ffmpeg version 6.0"


def test_first_line_empty_string_for_all_blank():
    assert diag._first_line("\n\n   \n") == ""


def _fake_torch(*, cuda_available, device_name="NVIDIA L4", capability=(8, 9), version="2.6.0", cuda_version="12.4"):
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda idx=0: device_name,
        get_device_capability=lambda idx=0: capability,
    )
    return types.SimpleNamespace(__version__=version, cuda=cuda_ns, version=types.SimpleNamespace(cuda=cuda_version))


def test_collect_gpu_diagnostics_all_pass(monkeypatch):
    fake_torch = _fake_torch(cuda_available=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "ffmpeg version 6.0\n") if cmd[0] == "ffmpeg" else (True, "ffprobe version 6.0\n"))

    result = diag.collect_gpu_diagnostics()

    assert result["ok"] is True
    assert result["completion_status"] == "COMPLETED"
    assert result["cuda_available"] is True
    assert result["torch_version"] == "2.6.0"
    assert result["cuda_version"] == "12.4"
    assert result["gpu_model"] == "NVIDIA L4"
    assert result["compute_capability"] == "8.9"
    assert result["ffmpeg_available"] is True
    assert result["ffmpeg_version"] == "ffmpeg version 6.0"
    assert result["ffprobe_available"] is True
    assert result["ffprobe_version"] == "ffprobe version 6.0"
    assert result["torch_error"] is None
    assert result["elapsed_s"] >= 0
    assert isinstance(result["python_version"], str) and result["python_version"]


def test_collect_gpu_diagnostics_never_leaks_a_non_plain_object(monkeypatch):
    # D-043 live failure (run 33612105029): modal run crashed client-side
    # with DeserializationError("... 'torch' module is not available in
    # the local environment") -- some torch-typed value leaked into the
    # returned dict despite bool()/f-string casts elsewhere. Simulates a
    # "leaky" torch whose version/capability fields are NOT plain str/int
    # (a custom class only __str__-able, not JSON-native) and asserts the
    # final result is nonetheless fully plain-JSON-native.
    class _LeakyStr:
        def __init__(self, value):
            self._value = value

        def __str__(self):
            return self._value

    class _LeakyInt:
        def __init__(self, value):
            self._value = value

        def __int__(self):
            return self._value

        def __str__(self):
            return str(self._value)

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx=0: _LeakyStr("NVIDIA L4"),
        get_device_capability=lambda idx=0: (_LeakyInt(8), _LeakyInt(9)),
    )
    leaky_torch = types.SimpleNamespace(
        __version__=_LeakyStr("2.6.0"), cuda=cuda_ns, version=types.SimpleNamespace(cuda=_LeakyStr("12.4"))
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", leaky_torch)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "v\n"))

    result = diag.collect_gpu_diagnostics()

    # Every value must be a plain JSON-native type -- no leaked object
    # whose class would need importing to reconstruct.
    reserialized = json.loads(json.dumps(result))
    assert reserialized == result
    for value in result.values():
        assert isinstance(value, (str, int, float, bool, type(None)))
    assert result["torch_version"] == "2.6.0"
    assert result["cuda_version"] == "12.4"
    assert result["gpu_model"] == "NVIDIA L4"
    assert result["compute_capability"] == "8.9"


def test_collect_gpu_diagnostics_cuda_unavailable_is_not_ok(monkeypatch):
    fake_torch = _fake_torch(cuda_available=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "v\n"))

    result = diag.collect_gpu_diagnostics()

    assert result["cuda_available"] is False
    assert result["gpu_model"] is None
    assert result["compute_capability"] is None
    assert result["ok"] is False
    assert result["completion_status"] == "COMPLETED_WITH_ISSUES"


def test_collect_gpu_diagnostics_torch_import_failure_is_reported_not_raised(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no module named torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "v\n"))

    result = diag.collect_gpu_diagnostics()

    assert result["torch_error"] == "no module named torch"
    assert result["cuda_available"] is False
    assert result["ok"] is False


def test_collect_gpu_diagnostics_missing_ffmpeg_is_not_ok(monkeypatch):
    fake_torch = _fake_torch(cuda_available=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    def _fake_run_capture(cmd, timeout=30.0):
        if cmd[0] == "ffmpeg":
            return False, "not found"
        return True, "ffprobe version 6.0\n"

    monkeypatch.setattr(diag, "_run_capture", _fake_run_capture)

    result = diag.collect_gpu_diagnostics()

    assert result["ffmpeg_available"] is False
    assert result["ffmpeg_version"] is None
    assert result["ok"] is False


def test_collect_gpu_diagnostics_result_is_json_serializable(monkeypatch):
    fake_torch = _fake_torch(cuda_available=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "v\n"))

    result = diag.collect_gpu_diagnostics()

    serialized = json.dumps(result)
    assert json.loads(serialized) == result


@pytest.mark.parametrize(
    "field",
    [
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_model",
        "compute_capability",
        "ffmpeg_available",
        "ffmpeg_version",
        "ffprobe_available",
        "ffprobe_version",
        "ok",
        "completion_status",
        "elapsed_s",
    ],
)
def test_collect_gpu_diagnostics_always_includes_required_field(monkeypatch, field):
    fake_torch = _fake_torch(cuda_available=True)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(diag, "_run_capture", lambda cmd, timeout=30.0: (True, "v\n"))

    result = diag.collect_gpu_diagnostics()

    assert field in result
