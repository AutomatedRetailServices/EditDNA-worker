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
    monkeypatch.delenv("S3_BUCKET", raising=False)
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
    # D-056.1 item 1: benchmark_result_uri is additive-only -- None here
    # because no S3_BUCKET is configured in this offline test, exactly the
    # "can't persist, don't fail the benchmark" path.
    assert result == {**sentinel, "benchmark_result_uri": None}


def test_run_video00_benchmark_reports_exception_without_crashing(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
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
    monkeypatch.delenv("S3_BUCKET", raising=False)
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


# --- D-056.1 "BENCHMARK EXECUTION RELIABILITY ONLY" ----------------------
# D-056's Run C dispatch saga (docs/CUTSELL_DECISIONS.md D-056 Section 8)
# root-caused why the mandated 3rd run could never be recovered: the ONLY
# place a compact result was ever persisted was a local file, written by
# main() ONLY after its blocking `.remote()` call returned cleanly -- a
# SIGTERM at any point before that write permanently discarded an
# otherwise-successful remote result. The fixtures/tests below exercise
# the real S3 put/get code paths in _persist_benchmark_result and
# _load_persisted_benchmark_result against an in-memory fake S3 client
# (never a real bucket), matching this file's existing precedent of
# mocking at the external-system boundary (`cutsell_worker.serverless_
# handler.run_op` above) rather than the internals behind it.


class _FakeS3Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _FakeS3Client:
    """In-memory stand-in for boto3's S3 client, scoped to exactly the two
    calls the persistence helpers make (put_object / get_object)."""

    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "ContentType": ContentType})
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        self.get_calls.append({"Bucket": Bucket, "Key": Key})
        if (Bucket, Key) not in self.store:
            raise KeyError(f"no such key: {Bucket}/{Key}")
        return {"Body": _FakeS3Body(self.store[(Bucket, Key)])}


def _install_fake_s3(monkeypatch, *, bucket="test-bucket") -> _FakeS3Client:
    import boto3

    fake_client = _FakeS3Client()
    monkeypatch.setenv("S3_BUCKET", bucket)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: fake_client)
    return fake_client


def test_benchmark_result_s3_key_is_deterministic_and_namespaced():
    key1 = cfg.benchmark_result_s3_key("video00-modal-123-1")
    key2 = cfg.benchmark_result_s3_key("video00-modal-123-1")
    assert key1 == key2
    assert key1 == "cutsell/benchmark-results/video00-modal-123-1/compact-result.json"
    # Never the same namespace as serverless_handler._focused()'s own
    # full-diagnostics upload (`cutsell/serverless/...`).
    assert not key1.startswith("cutsell/serverless/")


def test_benchmark_result_s3_key_rejects_empty_benchmark_id():
    with pytest.raises(ValueError):
        cfg.benchmark_result_s3_key("")


def test_persist_benchmark_result_writes_to_the_deterministic_key(monkeypatch):
    fake_s3 = _install_fake_s3(monkeypatch)
    result = {"ok": True, "selected_count": 5}

    uri = mvb._persist_benchmark_result("bench-1", result)

    assert uri == "s3://test-bucket/cutsell/benchmark-results/bench-1/compact-result.json"
    assert len(fake_s3.put_calls) == 1
    assert fake_s3.put_calls[0]["Key"] == cfg.benchmark_result_s3_key("bench-1")
    stored = json.loads(fake_s3.store[("test-bucket", cfg.benchmark_result_s3_key("bench-1"))])
    assert stored == result


def test_persist_benchmark_result_never_raises_without_a_bucket(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert mvb._persist_benchmark_result("bench-1", {"ok": True}) is None


def test_persist_benchmark_result_never_raises_without_a_benchmark_id(monkeypatch):
    _install_fake_s3(monkeypatch)
    assert mvb._persist_benchmark_result("", {"ok": True}) is None


def test_load_persisted_benchmark_result_round_trips(monkeypatch):
    _install_fake_s3(monkeypatch)
    result = {"ok": True, "selected_count": 7}
    mvb._persist_benchmark_result("bench-2", result)

    loaded = mvb._load_persisted_benchmark_result("bench-2")

    assert loaded == result


def test_load_persisted_benchmark_result_returns_none_when_absent(monkeypatch):
    _install_fake_s3(monkeypatch)
    assert mvb._load_persisted_benchmark_result("never-persisted") is None


# 5(a): Modal function completes while GH status appears stale -- the
# workflow's own job-step status is never consulted by these helpers at
# all; a completed remote result is recoverable purely from S3, exactly
# what a stale-status recovery path (the workflow's fallback download)
# relies on.
def test_persisted_result_recoverable_independent_of_caller_state(monkeypatch):
    _install_fake_s3(monkeypatch)
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    fake_module.run_op = lambda op, payload: {"ok": True, "selected_count": 9}
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    # Simulate the remote function completing (the ONLY thing that ever
    # calls .remote() successfully) ...
    mvb.run_video00_benchmark.remote({"op": "focused", "benchmark_id": "bench-stale"})

    # ... then a totally separate read, as if performed by a workflow step
    # that never saw the (possibly stale-reporting) job status at all.
    recovered = mvb._load_persisted_benchmark_result("bench-stale")
    assert recovered is not None
    assert recovered["ok"] is True
    assert recovered["selected_count"] == 9


# 5(b): result exists (in S3) before the local wrapper (main()) exits --
# i.e. persistence happens INSIDE the remote function call, not after.
def test_result_is_persisted_before_remote_call_returns_to_caller(monkeypatch):
    fake_s3 = _install_fake_s3(monkeypatch)
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    fake_module.run_op = lambda op, payload: {"ok": True, "selected_count": 3}
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    assert fake_s3.put_calls == []  # nothing persisted yet
    mvb.run_video00_benchmark.remote({"op": "focused", "benchmark_id": "bench-order"})
    # The put_object call already happened as part of .remote() itself --
    # a caller that dies the instant .remote() returns (the exact D-056
    # Run C failure mode) can never observe an un-persisted result.
    assert len(fake_s3.put_calls) == 1


# 5(c): cancellation never destroys an already-completed result -- once
# persisted, nothing in this module's code path ever deletes/overwrites an
# S3 object with a "not done" placeholder, and a later local-side failure
# on the SAME benchmark_id still finds it.
def test_cancellation_after_persistence_never_loses_the_result(monkeypatch):
    _install_fake_s3(monkeypatch)
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    fake_module.run_op = lambda op, payload: {"ok": True, "selected_count": 11}
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    mvb.run_video00_benchmark.remote({"op": "focused", "benchmark_id": "bench-cancel"})

    # Simulate the local wrapper being cancelled/killed AFTER persistence
    # (a no-op here: nothing to run) -- the persisted copy is untouched.
    recovered = mvb._load_persisted_benchmark_result("bench-cancel")
    assert recovered["ok"] is True
    assert recovered["selected_count"] == 11


# 5(d): no duplicate paid rerun when a result already exists for this
# exact benchmark_id -- main() must never call .remote() again.
def test_main_skips_duplicate_dispatch_when_result_already_persisted(monkeypatch, tmp_path):
    _install_fake_s3(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mvb._persist_benchmark_result("bench-dup", {"ok": True, "selected_count": 42})

    payload = {"op": "focused", "benchmark_id": "bench-dup"}
    monkeypatch.setenv(cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV, json.dumps(payload))

    def _fail_if_dispatched(*args, **kwargs):
        raise AssertionError("main() must not dispatch a duplicate paid Modal call")

    monkeypatch.setattr(mvb.run_video00_benchmark, "remote", _fail_if_dispatched)

    mvb.main()

    with open("modal-video00-result.json", "r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written == {"ok": True, "selected_count": 42}


# 5(e): a timeout/interruption on the LOCAL side (e.g. a Modal
# FunctionTimeoutError surfacing from .remote() itself) must still produce
# a deterministic, well-shaped diagnostic result -- never an uncaught
# traceback and an empty local file.
def test_main_produces_deterministic_diagnostic_result_on_timeout(monkeypatch, tmp_path):
    fake_s3 = _install_fake_s3(monkeypatch)
    monkeypatch.chdir(tmp_path)
    payload = {"op": "focused", "benchmark_id": "bench-timeout"}
    monkeypatch.setenv(cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV, json.dumps(payload))

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated modal.exception.FunctionTimeoutError")

    monkeypatch.setattr(mvb.run_video00_benchmark, "remote", _raise_timeout)

    mvb.main()

    with open("modal-video00-result.json", "r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["ok"] is False
    assert written["error_type"] == "TimeoutError"
    assert written["terminal_state"] == "local_wrapper_exception"
    assert written["benchmark_id"] == "bench-timeout"
    # The deterministic diagnostic result is itself persisted -- a later
    # poll/read for this benchmark_id recovers it instead of nothing. The
    # persisted copy predates the `benchmark_result_uri` field being set
    # on the local `written` dict (that field names the persisted copy's
    # own URI, so it cannot be included in the payload it is naming) --
    # every other field matches exactly.
    assert len(fake_s3.put_calls) == 1
    recovered = mvb._load_persisted_benchmark_result("bench-timeout")
    assert recovered == {k: v for k, v in written.items() if k != "benchmark_result_uri"}


def test_main_dispatches_normally_when_no_existing_result(monkeypatch, tmp_path):
    _install_fake_s3(monkeypatch)
    monkeypatch.chdir(tmp_path)
    fake_module = types.ModuleType("cutsell_worker.serverless_handler")
    fake_module.run_op = lambda op, payload: {"ok": True, "selected_count": 15}
    monkeypatch.setitem(sys.modules, "cutsell_worker.serverless_handler", fake_module)

    payload = {"op": "focused", "benchmark_id": "bench-fresh"}
    monkeypatch.setenv(cfg.CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV, json.dumps(payload))

    mvb.main()

    with open("modal-video00-result.json", "r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["ok"] is True
    assert written["selected_count"] == 15
    assert written["benchmark_result_uri"] == "s3://test-bucket/cutsell/benchmark-results/bench-fresh/compact-result.json"
