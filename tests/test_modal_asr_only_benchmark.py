"""D-053 Section 5: tests for modal_asr_only_benchmark.py's App/Image/
Function/Secret wiring and its cutsell_worker.asr_only_benchmark
delegation. Mirrors tests/test_modal_video00_full_benchmark.py's stubbing
approach exactly -- the real `modal` package is not installed in this dev
sandbox."""
from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    """Same MERGE-onto-shared-stub precedent as
    test_modal_video00_full_benchmark.py -- multiple test files need a
    fake `modal` module with different attributes, and whichever file
    pytest collects first must not "win" the stub and leave later files'
    needed attributes missing. Never shadows a real installed `modal`
    package."""
    existing = sys.modules.get(name)
    if existing is not None and not getattr(existing, "__cutsell_test_stub__", False):
        return
    if existing is None:
        if importlib.util.find_spec(name) is not None:
            return
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
import modal_asr_only_benchmark as mab  # noqa: E402


def test_gpu_type_is_l4():
    assert mab.MODAL_GPU_TYPE == "L4"


def test_app_has_expected_name_distinct_from_full_video00_app():
    assert mab.app.name == cfg.MODAL_ASR_ONLY_APP_NAME
    assert mab.app.name != cfg.MODAL_VIDEO00_APP_NAME


def test_function_registered_with_l4_gpu_and_asr_only_timeout():
    assert len(mab.app.function_calls) == 1
    call_kwargs = mab.app.function_calls[0]
    assert call_kwargs["gpu"] == "L4"
    assert call_kwargs["timeout"] == cfg.DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S
    assert 0 < call_kwargs["timeout"] <= cfg.MAX_MODAL_ASR_ONLY_TIMEOUT_S


def test_asr_only_timeout_is_materially_shorter_than_full_video00_timeout():
    # D-053 Section 5: "This should make one ASR test materially cheaper
    # and faster than a full RAW."
    assert cfg.DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S < cfg.DEFAULT_MODAL_VIDEO00_TIMEOUT_S


def test_function_has_no_retry_loop():
    call_kwargs = mab.app.function_calls[0]
    assert call_kwargs["retries"] == 0


def test_function_never_requests_an_excluded_gpu():
    call_kwargs = mab.app.function_calls[0]
    assert call_kwargs["gpu"] not in cfg.EXCLUDED_MODAL_GPU_TYPES


def test_function_has_a_secret_attached():
    call_kwargs = mab.app.function_calls[0]
    assert call_kwargs["secrets"] == [mab.cutsell_env_secret]


def test_image_built_from_the_cutsell_base_image_verbatim():
    assert _fake_image_instance.from_registry_calls
    args, _kwargs = _fake_image_instance.from_registry_calls[0]
    assert args[0] == cfg.CUTSELL_BASE_IMAGE


def test_image_mounts_the_whole_cutsell_worker_package_and_modal_gpu_config():
    assert _fake_image_instance.add_local_python_source_calls
    args, _kwargs = _fake_image_instance.add_local_python_source_calls[0]
    assert "cutsell_worker" in args
    assert "modal_gpu_config" in args


def test_resolve_env_secret_returns_empty_secret_when_path_unset(monkeypatch):
    monkeypatch.delenv(cfg.CUTSELL_ENV_JSON_PATH_ENV, raising=False)
    secret = mab._resolve_env_secret()
    assert secret.data == {}


def test_resolve_env_secret_reads_the_given_json_file(tmp_path, monkeypatch):
    env_file = tmp_path / "cutsell-env.json"
    env_file.write_text(json.dumps({"S3_BUCKET": "my-bucket"}))
    monkeypatch.setenv(cfg.CUTSELL_ENV_JSON_PATH_ENV, str(env_file))
    secret = mab._resolve_env_secret()
    assert secret.data == {"S3_BUCKET": "my-bucket"}


def test_run_asr_only_benchmark_delegates_not_reimplemented(monkeypatch):
    # No forked/duplicated ASR logic -- delegates straight to
    # cutsell_worker.asr_only_benchmark.run_asr_only_benchmark.
    fake_module = types.ModuleType("cutsell_worker.asr_only_benchmark")
    sentinel = {"ok": True, "evidence_hash": "deadbeef", "sentinel_marker": "no-duplicated-asr"}
    calls = []

    def _fake_run(payload):
        calls.append(payload)
        return sentinel

    fake_module.run_asr_only_benchmark = _fake_run
    monkeypatch.setitem(sys.modules, "cutsell_worker.asr_only_benchmark", fake_module)

    payload = {"source_key": "some/key.mp4", "benchmark_id": "b1"}
    result = mab.run_asr_only_benchmark.remote(payload)

    assert calls == [payload]
    assert result == sentinel


def test_run_asr_only_benchmark_reports_exception_without_crashing(monkeypatch):
    fake_module = types.ModuleType("cutsell_worker.asr_only_benchmark")

    def _raise(payload):
        raise ValueError("source_key is required")

    fake_module.run_asr_only_benchmark = _raise
    monkeypatch.setitem(sys.modules, "cutsell_worker.asr_only_benchmark", fake_module)

    result = mab.run_asr_only_benchmark.remote({})

    assert result["ok"] is False
    assert "source_key is required" in result["error"]
    assert result["error_type"] == "ValueError"


def test_run_asr_only_benchmark_result_is_json_serializable(monkeypatch):
    fake_module = types.ModuleType("cutsell_worker.asr_only_benchmark")
    fake_module.run_asr_only_benchmark = lambda payload: {"ok": True, "evidence_hash": "abc"}
    monkeypatch.setitem(sys.modules, "cutsell_worker.asr_only_benchmark", fake_module)

    result = mab.run_asr_only_benchmark.remote({"source_key": "x"})

    serialized = json.dumps(result)
    assert json.loads(serialized) == result


def test_main_requires_payload_env_var(monkeypatch):
    monkeypatch.delenv(cfg.CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV, raising=False)
    with pytest.raises(RuntimeError, match=cfg.CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV):
        mab.main()


def test_main_never_reuses_the_full_video00_payload_env_var():
    # D-053: an ASR-only dispatch must never be confused with (or
    # accidentally reuse) a full Video00 payload shape.
    assert cfg.CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV != cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV
