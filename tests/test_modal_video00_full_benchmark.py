"""D-043 full Video00 execution phase: tests for
modal_video00_full_benchmark.py's App/Image/Function/Secret wiring and its
run_op() delegation. The real `modal` package is not installed in this dev
sandbox; stubbed the same way tests/test_modal_gpu_minimal_test.py stubs
it, extended with fakes for the new Image methods (pip_install,
pip_install_from_requirements) and modal.Secret."""
from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    """Same stubbing precedent as tests/test_modal_gpu_minimal_test.py,
    extended to MERGE attributes onto an already-stubbed fake module rather
    than a one-shot "only if absent" -- two separate test files (this one
    and test_modal_gpu_minimal_test.py) both need a fake `modal` module
    with different attributes (this one additionally needs `Secret`), and
    whichever test file pytest collects first would otherwise "win" the
    stub, silently leaving the second file's needed attributes (e.g.
    `modal.Secret`) missing on the shared sys.modules entry. Never shadows
    a real installed `modal` package."""
    existing = sys.modules.get(name)
    if existing is not None and not getattr(existing, "__cutsell_test_stub__", False):
        return  # a real module (or a non-stub module) is already imported
    if existing is None:
        if importlib.util.find_spec(name) is not None:
            return  # a real installable package exists; never shadow it
        existing = types.ModuleType(name)
        existing.__cutsell_test_stub__ = True
        sys.modules[name] = existing
    for attribute, value in attributes.items():
        setattr(existing, attribute, value)


class _FakeImage:
    def __init__(self):
        self.from_registry_calls: list[tuple] = []
        self.apt_install_calls: list[tuple] = []
        self.pip_install_calls: list[tuple] = []
        self.pip_install_from_requirements_calls: list[tuple] = []
        self.add_local_python_source_calls: list[tuple] = []

    def from_registry(self, *args, **kwargs):
        self.from_registry_calls.append((args, kwargs))
        return self

    def apt_install(self, *args, **kwargs):
        self.apt_install_calls.append((args, kwargs))
        return self

    def pip_install(self, *args, **kwargs):
        self.pip_install_calls.append((args, kwargs))
        return self

    def pip_install_from_requirements(self, *args, **kwargs):
        self.pip_install_from_requirements_calls.append((args, kwargs))
        return self

    def add_local_python_source(self, *args, **kwargs):
        self.add_local_python_source_calls.append((args, kwargs))
        return self


class _FakeSecret:
    calls: list[dict] = []

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_dict(cls, data):
        cls.calls.append(dict(data))
        return cls(data)


class _FakeFunctionHandle:
    def __init__(self, fn, kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _FakeApp:
    def __init__(self, name):
        self.name = name
        self.function_calls: list[dict] = []

    def function(self, **kwargs):
        self.function_calls.append(kwargs)

        def deco(fn):
            return _FakeFunctionHandle(fn, kwargs)

        return deco

    def local_entrypoint(self):
        def deco(fn):
            return fn

        return deco


_fake_image_instance = _FakeImage()
_stub_missing_module("modal", App=_FakeApp, Image=_fake_image_instance, Secret=_FakeSecret)

import modal_gpu_config as cfg  # noqa: E402
import modal_video00_full_benchmark as mvb  # noqa: E402


def test_gpu_type_is_l4():
    assert mvb.MODAL_GPU_TYPE == "L4"


def test_app_has_expected_name():
    assert mvb.app.name == cfg.MODAL_VIDEO00_APP_NAME


def test_function_registered_with_l4_gpu_and_video00_timeout():
    assert len(mvb.app.function_calls) == 1
    call_kwargs = mvb.app.function_calls[0]
    assert call_kwargs["gpu"] == "L4"
    assert call_kwargs["timeout"] == cfg.DEFAULT_MODAL_VIDEO00_TIMEOUT_S
    assert 0 < call_kwargs["timeout"] <= cfg.MAX_MODAL_VIDEO00_TIMEOUT_S


def test_function_has_no_retry_loop():
    # Same crash-loop protection the minimal smoke test's own live failure
    # (run 33602989294) required -- never optional.
    call_kwargs = mvb.app.function_calls[0]
    assert call_kwargs["retries"] == 0


def test_function_never_requests_an_excluded_gpu():
    call_kwargs = mvb.app.function_calls[0]
    assert call_kwargs["gpu"] not in cfg.EXCLUDED_MODAL_GPU_TYPES


def test_function_has_a_secret_attached():
    call_kwargs = mvb.app.function_calls[0]
    assert call_kwargs["secrets"] == [mvb.cutsell_env_secret]


def test_image_built_from_the_cutsell_base_image_verbatim():
    assert _fake_image_instance.from_registry_calls
    args, _kwargs = _fake_image_instance.from_registry_calls[0]
    assert args[0] == cfg.CUTSELL_BASE_IMAGE


def test_image_installs_the_same_apt_packages_as_the_dockerfile():
    assert _fake_image_instance.apt_install_calls
    args, _kwargs = _fake_image_instance.apt_install_calls[0]
    assert args == cfg.CUTSELL_APT_PACKAGES


def test_image_installs_from_the_canonical_requirements_file():
    assert _fake_image_instance.pip_install_from_requirements_calls
    args, _kwargs = _fake_image_instance.pip_install_from_requirements_calls[0]
    assert args[0] == cfg.CUTSELL_REQUIREMENTS_FILE


def test_image_installs_the_same_runpod_pip_spec_as_the_dockerfile():
    assert _fake_image_instance.pip_install_calls
    args, _kwargs = _fake_image_instance.pip_install_calls[0]
    assert args[0] == cfg.CUTSELL_RUNPOD_PIP_SPEC


def test_image_mounts_the_whole_cutsell_worker_package():
    assert _fake_image_instance.add_local_python_source_calls
    args, _kwargs = _fake_image_instance.add_local_python_source_calls[0]
    assert "cutsell_worker" in args


def test_image_mounts_modal_gpu_config_too():
    # D-043 live evidence: the first live dispatch (App ap-WX9iPZfMQnhPQJsM1rZwLm,
    # Function fu-UtPpmyf89ZpgT6JFPc8mDr) crash-looped for ~90 minutes --
    # three container attempts, each dying within ~2s with
    # `ModuleNotFoundError: No module named 'modal_gpu_config'` -- because
    # this script's own top-level `from modal_gpu_config import (...)`
    # was never mounted, only cutsell_worker was. Locks the fix so this
    # specific omission can never silently regress.
    assert _fake_image_instance.add_local_python_source_calls
    args, _kwargs = _fake_image_instance.add_local_python_source_calls[0]
    assert "modal_gpu_config" in args


def test_resolve_env_secret_returns_empty_secret_when_path_unset(monkeypatch):
    monkeypatch.delenv(cfg.CUTSELL_ENV_JSON_PATH_ENV, raising=False)
    secret = mvb._resolve_env_secret()
    assert secret.data == {}


def test_resolve_env_secret_reads_the_given_json_file(tmp_path, monkeypatch):
    env_file = tmp_path / "cutsell-env.json"
    env_file.write_text(json.dumps({"S3_BUCKET": "my-bucket", "GEMINI_API_KEY": "fake-key"}))
    monkeypatch.setenv(cfg.CUTSELL_ENV_JSON_PATH_ENV, str(env_file))
    secret = mvb._resolve_env_secret()
    assert secret.data == {"S3_BUCKET": "my-bucket", "GEMINI_API_KEY": "fake-key"}


def test_resolve_env_secret_rejects_an_empty_json_object(tmp_path, monkeypatch):
    env_file = tmp_path / "cutsell-env.json"
    env_file.write_text("{}")
    monkeypatch.setenv(cfg.CUTSELL_ENV_JSON_PATH_ENV, str(env_file))
    with pytest.raises(RuntimeError):
        mvb._resolve_env_secret()


def test_run_video00_benchmark_delegates_to_run_op_not_reimplemented(monkeypatch):
    # No forked/duplicated editorial logic -- the same run_op() dispatcher
    # RunPod Serverless and RunPod Pod both already use.
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    sentinel = {"ok": True, "sentinel_marker": "no-duplicated-editor", "selected_count": 23}
    calls = []

    def _fake_run_op(op, payload):
        calls.append((op, payload))
        return sentinel

    fake_module.run_op = _fake_run_op
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    payload = {"op": "focused", "source_key": "some/key.mp4", "benchmark_id": "b1", "auto_speech_visual_microtrim": True}
    result = mvb.run_video00_benchmark.remote(payload)

    assert calls == [("focused", payload)]
    assert result == sentinel


def test_run_video00_benchmark_reports_exception_without_crashing(monkeypatch):
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")

    def _raise(op, payload):
        raise ValueError("source_key is required")

    fake_module.run_op = _raise
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    result = mvb.run_video00_benchmark.remote({"op": "focused"})

    assert result["ok"] is False
    assert "source_key is required" in result["error"]
    assert result["error_type"] == "ValueError"


def test_run_video00_benchmark_result_is_json_serializable(monkeypatch):
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    fake_module.run_op = lambda op, payload: {"ok": True, "selected_count": 23, "deliverable": True}
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    result = mvb.run_video00_benchmark.remote({"op": "focused"})

    serialized = json.dumps(result)
    assert json.loads(serialized) == result


def test_main_requires_payload_env_var(monkeypatch):
    monkeypatch.delenv(cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV, raising=False)
    with pytest.raises(RuntimeError, match=cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV):
        mvb.main()
