"""D-043 (CutSell Modal GPU execution -- first live validation): tests
for modal_gpu_minimal_test.py's App/Image/Function wiring -- GPU type,
base image, bounded timeout, and that diagnostics are delegated to
modal_gpu_diagnostics (never a second, duplicated implementation). The
real `modal` package is not installed in this dev sandbox; stubbed the
same way tests/test_pod_direct_benchmark_entrypoint.py stubs `runpod`."""
from __future__ import annotations

import importlib.util
import sys
import types


def _stub_missing_module(name, **attributes):
    if name in sys.modules:
        return
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module


class _FakeImage:
    def __init__(self):
        self.from_registry_calls: list[tuple] = []
        self.apt_install_calls: list[tuple] = []

    def from_registry(self, *args, **kwargs):
        self.from_registry_calls.append((args, kwargs))
        return self

    def apt_install(self, *args, **kwargs):
        self.apt_install_calls.append((args, kwargs))
        return self


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
_stub_missing_module("modal", App=_FakeApp, Image=_fake_image_instance)

import modal_gpu_config as cfg  # noqa: E402
import modal_gpu_minimal_test as mgt  # noqa: E402


def test_gpu_type_is_l4():
    assert mgt.MODAL_GPU_TYPE == "L4"


def test_app_has_expected_name():
    assert mgt.app.name == "cutsell-gpu-minimal-isolation"


def test_function_registered_with_l4_gpu_and_bounded_timeout():
    assert len(mgt.app.function_calls) == 1
    call_kwargs = mgt.app.function_calls[0]
    assert call_kwargs["gpu"] == "L4"
    assert call_kwargs["timeout"] == cfg.DEFAULT_MODAL_TIMEOUT_S
    assert 0 < call_kwargs["timeout"] <= cfg.MAX_MODAL_SMOKE_TEST_TIMEOUT_S


def test_function_never_requests_an_excluded_gpu():
    call_kwargs = mgt.app.function_calls[0]
    assert call_kwargs["gpu"] not in cfg.EXCLUDED_MODAL_GPU_TYPES


def test_image_built_from_the_cutsell_base_image_verbatim():
    assert _fake_image_instance.from_registry_calls
    args, _kwargs = _fake_image_instance.from_registry_calls[0]
    assert args[0] == cfg.CUTSELL_BASE_IMAGE


def test_image_installs_ffmpeg_matching_the_dockerfile():
    assert _fake_image_instance.apt_install_calls
    args, _kwargs = _fake_image_instance.apt_install_calls[0]
    assert "ffmpeg" in args


def test_run_minimal_gpu_check_delegates_to_diagnostics_module_not_reimplemented(monkeypatch):
    sentinel = {"ok": True, "sentinel_marker": "no-duplicated-editor"}
    monkeypatch.setattr(mgt, "collect_gpu_diagnostics", lambda: sentinel)
    result = mgt.run_minimal_gpu_check.remote()
    assert result is sentinel
