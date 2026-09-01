"""D-041 GPU-fallback-audit follow-up: `serverless_handler._health()` must
report as much hardware/runtime evidence as safely available even when CUDA
initialization fails -- specifically so an incompatible GPU such as RTX PRO
6000 Blackwell (sm_120) reports explicitly why, instead of collapsing to a
bare `cuda_available=false`. Diagnostic-only: no CutSell editorial/Clean Cut
logic is exercised or changed by this file.

Every hardware probe inside `_health()` is independently guarded, so these
tests build a fully-controllable fake `torch` module (the real dependency is
only installed inside the GPU worker container image, not in this CI/dev
environment) and drive each guarded branch directly.
"""
from __future__ import annotations

import importlib.util
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


# serverless_handler.py imports `runpod` unconditionally at module scope --
# only available inside the GPU worker container image. Same stubbing
# pattern as tests/test_cutsell_serverless_focused_contract.py.
_stub_missing_module("runpod", serverless=types.SimpleNamespace(start=lambda *_a, **_k: None))

from cutsell_worker import serverless_handler  # noqa: E402


_DEFAULT_ARCH_LIST = ("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")


class _FakeCuda:
    def __init__(
        self,
        *,
        is_available=True,
        device_count=1,
        device_name="NVIDIA RTX A6000",
        capability=(8, 6),
        arch_list=_DEFAULT_ARCH_LIST,
        is_available_raises=None,
        device_count_raises=None,
        device_name_raises=None,
        capability_raises=None,
    ):
        self._is_available = is_available
        self._device_count = device_count
        self._device_name = device_name
        self._capability = capability
        self._arch_list = arch_list
        self._is_available_raises = is_available_raises
        self._device_count_raises = device_count_raises
        self._device_name_raises = device_name_raises
        self._capability_raises = capability_raises

    def is_available(self):
        if self._is_available_raises:
            raise self._is_available_raises
        return self._is_available

    def device_count(self):
        if self._device_count_raises:
            raise self._device_count_raises
        return self._device_count

    def get_device_name(self, _index):
        if self._device_name_raises:
            raise self._device_name_raises
        return self._device_name

    def get_device_capability(self, _index):
        if self._capability_raises:
            raise self._capability_raises
        return self._capability

    def get_arch_list(self):
        return list(self._arch_list)


def _fake_torch(**cuda_kwargs) -> types.ModuleType:
    fake = types.ModuleType("torch")
    fake.__version__ = "2.6.0"
    fake.version = types.SimpleNamespace(cuda="12.4")
    fake.cuda = _FakeCuda(**cuda_kwargs)
    return fake


@pytest.fixture(autouse=True)
def _no_real_nvidia_smi(monkeypatch):
    # nvidia-smi is a best-effort, always-optional probe -- force it absent
    # so these tests don't depend on whatever host they happen to run on.
    import subprocess as _subprocess

    def _raise(*_a, **_k):
        raise FileNotFoundError("nvidia-smi not present in this environment")

    monkeypatch.setattr(_subprocess, "run", _raise)


# ---------------------------------------------------------------------------
# 1. supported GPU
# ---------------------------------------------------------------------------
def test_health_reports_supported_gpu_with_no_incompatibility_claim(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        is_available=True, device_count=1, device_name="NVIDIA RTX A6000", capability=(8, 6),
    ))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["cuda_available"] is True
    assert result["device_name"] == "NVIDIA RTX A6000"
    assert result["compute_capability"] == "sm_86"
    assert result["torch_version"] == "2.6.0"
    assert result["torch_compiled_cuda_version"] == "12.4"
    assert result["incompatibility_reason"] is None
    assert "hostname" in result
    assert "worker_id" in result
    assert "cuda_runtime_version" in result


# ---------------------------------------------------------------------------
# 2. unsupported compute capability (the RTX PRO 6000 Blackwell / sm_120 case)
# ---------------------------------------------------------------------------
def test_health_reports_unsupported_compute_capability_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        is_available=False,
        device_count=1,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        capability=(12, 0),
    ))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["cuda_available"] is False
    assert result["device_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert result["compute_capability"] == "sm_120"
    assert result["incompatibility_reason"] is not None
    assert "sm_120" in result["incompatibility_reason"]
    assert "does not support this GPU" in result["incompatibility_reason"]


# ---------------------------------------------------------------------------
# 3. CUDA unavailable (no device at all -- must not fabricate an
#    incompatibility claim when there is no GPU to diagnose)
# ---------------------------------------------------------------------------
def test_health_cuda_unavailable_no_device_reports_no_incompatibility_claim(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(is_available=False, device_count=0))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["cuda_available"] is False
    assert result["cuda_device_count"] == 0
    assert result["device_name"] is None
    assert result["compute_capability"] is None
    # No compute capability was ever detected -- there is nothing to
    # diagnose an incompatibility against, so no claim is fabricated.
    assert result["incompatibility_reason"] is None


# ---------------------------------------------------------------------------
# 4. device-name lookup failure
# ---------------------------------------------------------------------------
def test_health_device_name_lookup_failure_is_captured_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        is_available=True, device_count=1,
        device_name_raises=RuntimeError("device name probe boom"),
        capability=(8, 6),
    ))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["device_name"] is None
    assert "device_name_error" in result
    assert "boom" in result["device_name_error"]
    # The failure of one probe must not hide the others.
    assert result["compute_capability"] == "sm_86"
    assert result["cuda_available"] is True


# ---------------------------------------------------------------------------
# 5. capability lookup failure
# ---------------------------------------------------------------------------
def test_health_capability_lookup_failure_is_captured_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        is_available=False, device_count=1,
        device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        capability_raises=RuntimeError("capability probe boom"),
    ))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["compute_capability"] is None
    assert "compute_capability_error" in result
    assert "boom" in result["compute_capability_error"]
    # The failure of one probe must not hide the others.
    assert result["device_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    # No compute capability could be read -- fail safely, don't guess at an
    # incompatibility diagnosis without the data to support it.
    assert result["incompatibility_reason"] is None


# ---------------------------------------------------------------------------
# Extra coverage beyond the 5 required cases: is_available() itself raising
# (the "fail safely if GPU details cannot be read" requirement's most
# fundamental case) must not crash the health op at all.
# ---------------------------------------------------------------------------
def test_health_is_available_raising_is_captured_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        is_available_raises=RuntimeError("driver init boom"),
        device_count=0,
    ))
    result = serverless_handler._health()
    assert result["ok"] is True
    assert result["cuda_available"] is False
    assert "cuda_init_error" in result
    assert "boom" in result["cuda_init_error"]
