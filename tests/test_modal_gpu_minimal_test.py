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
    """Merges attributes onto an already-stubbed fake module rather than a
    one-shot "only if absent" -- test_modal_video00_full_benchmark.py also
    stubs `modal` (with additional attributes, e.g. `Secret`), and
    whichever test file pytest collects first must not "win" the stub and
    leave the other file's needed attributes missing on the shared
    sys.modules entry. Never shadows a real installed `modal` package."""
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
        self.add_local_python_source_calls: list[tuple] = []

    def from_registry(self, *args, **kwargs):
        self.from_registry_calls.append((args, kwargs))
        return self

    def apt_install(self, *args, **kwargs):
        self.apt_install_calls.append((args, kwargs))
        return self

    def add_local_python_source(self, *args, **kwargs):
        self.add_local_python_source_calls.append((args, kwargs))
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


def test_function_has_no_retry_loop():
    # D-043 live evidence (run 33602989294): Modal's own default
    # container-crash retry behavior kept relaunching L4 GPU containers
    # for ~18 minutes on a crash-looping function -- directly violating
    # the explicit "no retry loop" requirement. retries=0 is required,
    # not optional.
    call_kwargs = mgt.app.function_calls[0]
    assert call_kwargs["retries"] == 0


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


def test_image_mounts_local_source_so_the_container_can_import_it():
    # D-043 live evidence (run 33602989294): without this, every
    # container crashed immediately with ModuleNotFoundError for
    # modal_gpu_config -- Image.from_registry + apt_install alone never
    # made this repo's own local modules importable remotely.
    assert _fake_image_instance.add_local_python_source_calls
    args, _kwargs = _fake_image_instance.add_local_python_source_calls[0]
    assert "modal_gpu_config" in args
    assert "modal_gpu_diagnostics" in args


def test_run_minimal_gpu_check_delegates_to_diagnostics_module_not_reimplemented(monkeypatch):
    sentinel = {"ok": True, "sentinel_marker": "no-duplicated-editor"}
    monkeypatch.setattr(mgt, "collect_gpu_diagnostics", lambda: sentinel)
    result = mgt.run_minimal_gpu_check.remote()
    assert result is sentinel
