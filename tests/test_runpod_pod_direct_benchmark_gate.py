"""D-042 follow-up ("restore the known-working execution model"): the
direct-execution Pod benchmark gate -- S3-polled readiness/completion,
never HTTP, never container logs. Fully mocked: no network, no GPU, no
real boto3 client."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import runpod_pod_direct_benchmark_gate as gate


# ---------------------------------------------------------------------------
# build_direct_exec_config -- pure function
# ---------------------------------------------------------------------------
BASE_TEMPLATE = {
    "id": "5moabglc4m",
    "name": "CutSell-Pod-QA",
    "imageName": "ghcr.io/automatedretailservices/cutsell-serverless@sha256:deadbeef",
    "containerDiskInGb": 80,
    "env": {
        "S3_BUCKET": "real-bucket",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "AKIA-fake",
        "AWS_SECRET_ACCESS_KEY": "fake-secret",
        "GEMINI_API_KEY": "real-key",
    },
}


def test_build_direct_exec_config_inherits_image_disk_and_env_from_template():
    payload = {"op": "focused", "source_key": "x", "benchmark_id": "bench-1"}
    config = gate.build_direct_exec_config(
        BASE_TEMPLATE, api_key="fake-key", pod_name="n", payload=payload, cost_ceiling_usd_per_hr=1.5
    )
    assert config.image == BASE_TEMPLATE["imageName"]
    assert config.container_disk_gb == 80
    assert config.env["GEMINI_API_KEY"] == "real-key"  # inherited, not dropped
    assert config.env["S3_BUCKET"] == "real-bucket"
    assert json.loads(config.env["CUTSELL_BENCHMARK_PAYLOAD_JSON"]) == payload
    assert config.start_command == "python3 -m cutsell_worker.pod_direct_benchmark_entrypoint"
    assert config.template_id is None  # deliberately never uses template_id mode


def test_build_direct_exec_config_never_mutates_base_template():
    original_env = dict(BASE_TEMPLATE["env"])
    gate.build_direct_exec_config(
        BASE_TEMPLATE, api_key="k", pod_name="n", payload={"benchmark_id": "b"}, cost_ceiling_usd_per_hr=1.5
    )
    assert BASE_TEMPLATE["env"] == original_env


def test_build_direct_exec_config_defaults_disk_when_template_omits_it():
    template = {k: v for k, v in BASE_TEMPLATE.items() if k != "containerDiskInGb"}
    config = gate.build_direct_exec_config(
        template, api_key="k", pod_name="n", payload={"benchmark_id": "b"}, cost_ceiling_usd_per_hr=1.5
    )
    assert config.container_disk_gb == 80


def test_build_direct_exec_config_cloud_types_unset_keeps_default_sweep():
    # D-042 controlled SECURE-cloud test: every other caller of this
    # function must be completely unaffected -- omitting cloud_types keeps
    # PodExecutionConfig's own default (the existing COMMUNITY-then-SECURE
    # sweep), not some narrowed value.
    config = gate.build_direct_exec_config(
        BASE_TEMPLATE, api_key="k", pod_name="n", payload={"benchmark_id": "b"}, cost_ceiling_usd_per_hr=1.5
    )
    assert config.cloud_types == gate.POD_CLOUD_TYPES


def test_build_direct_exec_config_forwards_explicit_cloud_types():
    config = gate.build_direct_exec_config(
        BASE_TEMPLATE,
        api_key="k",
        pod_name="n",
        payload={"benchmark_id": "b"},
        cost_ceiling_usd_per_hr=1.5,
        cloud_types=("SECURE",),
    )
    assert config.cloud_types == ("SECURE",)


# ---------------------------------------------------------------------------
# S3 polling primitives
# ---------------------------------------------------------------------------
class _FakeClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": str(code)}}


class FakeS3Client:
    def __init__(self, existing_keys=()):
        self.objects = {k: "{}" for k in existing_keys}
        self.head_calls = []
        self.download_calls = []

    def head_object(self, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if Key not in self.objects:
            raise _FakeClientError(404)
        return {}

    def download_file(self, Bucket, Key, Filename):
        self.download_calls.append((Bucket, Key, Filename))
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_text(self.objects[Key], encoding="utf-8")

    def raise_non_404(self, Bucket=None, Key=None):
        raise _FakeClientError(403)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_s3_key_exists_true_for_present_key():
    client = FakeS3Client(existing_keys=["a/b.json"])
    assert gate.s3_key_exists(client, "bucket", "a/b.json") is True


def test_s3_key_exists_false_for_404():
    client = FakeS3Client(existing_keys=[])
    assert gate.s3_key_exists(client, "bucket", "missing.json") is False


def test_s3_key_exists_reraises_non_404_errors():
    client = FakeS3Client()
    client.head_object = client.raise_non_404
    with pytest.raises(_FakeClientError):
        gate.s3_key_exists(client, "bucket", "x")


def test_poll_for_first_existing_key_finds_immediately():
    client = FakeS3Client(existing_keys=["found.json"])
    clock = FakeClock()
    result = gate.poll_for_first_existing_key(
        client, "bucket", ["found.json"], timeout_s=60, interval_s=5, now=clock.now, sleep=clock.sleep
    )
    assert result == "found.json"


def test_poll_for_first_existing_key_checks_multiple_keys_in_order():
    client = FakeS3Client(existing_keys=["second.json"])
    clock = FakeClock()
    result = gate.poll_for_first_existing_key(
        client, "bucket", ["first.json", "second.json"], timeout_s=60, interval_s=5, now=clock.now, sleep=clock.sleep
    )
    assert result == "second.json"


def test_poll_for_first_existing_key_times_out_returns_none():
    client = FakeS3Client(existing_keys=[])
    clock = FakeClock()
    result = gate.poll_for_first_existing_key(
        client, "bucket", ["never.json"], timeout_s=20, interval_s=5, now=clock.now, sleep=clock.sleep
    )
    assert result is None
    assert clock.t >= 20


# ---------------------------------------------------------------------------
# main() -- full orchestration, fully mocked
# ---------------------------------------------------------------------------
class _FakeLifecycleResult:
    def __init__(self, pod_id, classification="POD_CREATED_FRESH"):
        self.pod_id = pod_id
        self.classification = classification


class _FakeProvider:
    instances = []

    def __init__(self, transport, config, *, existing_pod_id=None, log=None):
        self.config = config
        self.teardown_called = False
        self._pod_id = "pod-direct-1"
        self._lifecycle_result = _FakeLifecycleResult("pod-direct-1")
        _FakeProvider.instances.append(self)

    @property
    def pod_id(self):
        return self._pod_id

    def ensure_ready(self):
        return self._lifecycle_result

    def teardown(self):
        self.teardown_called = True


@pytest.fixture(autouse=True)
def _patch_provider_and_template(monkeypatch):
    _FakeProvider.instances.clear()
    monkeypatch.setattr(gate, "RunPodPodExecutionProvider", _FakeProvider)
    monkeypatch.setattr(gate, "UrllibTransport", lambda: object())
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: dict(BASE_TEMPLATE))


@pytest.fixture
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("BENCHMARK_ID", "bench-live-1")
    monkeypatch.setenv("SOURCE_KEY", "some/video.mp4")
    monkeypatch.setenv("SANITY_TIMEOUT_S", "0.05")
    monkeypatch.setenv("BENCHMARK_TIMEOUT_S", "0.05")
    monkeypatch.setenv("SANITY_POLL_INTERVAL_S", "0.01")
    monkeypatch.setenv("BENCHMARK_POLL_INTERVAL_S", "0.01")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _patch_s3(monkeypatch, client):
    monkeypatch.setattr(gate, "_make_s3_client", lambda **kwargs: client)


def test_template_not_found_aborts_before_any_pod_action(monkeypatch, _env):
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: None)
    exit_code = gate.main()
    assert exit_code == 1
    assert _FakeProvider.instances == []
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "TEMPLATE_NOT_FOUND"


def test_lifecycle_failure_still_tears_down_and_reports(monkeypatch, _env):
    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        self.config = config
        self.teardown_called = False
        self._pod_id = None
        self._lifecycle_result = _FakeLifecycleResult(None, classification="POD_CAPACITY_UNAVAILABLE")
        _FakeProvider.instances.append(self)

    monkeypatch.setattr(_FakeProvider, "__init__", _init)
    _patch_s3(monkeypatch, FakeS3Client())

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "POD_LIFECYCLE_FAILED"


def test_sanity_timeout_tears_down_and_never_polls_for_benchmark_result(monkeypatch, _env):
    client = FakeS3Client(existing_keys=[])
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "SANITY_CHECK_TIMEOUT"
    assert not any("run_output" in k or "pod-execution-error" in k for _, k in client.head_calls)


def test_sanity_failure_stops_before_benchmark_polling(monkeypatch, _env):
    client = FakeS3Client(existing_keys=["cutsell/serverless/bench-live-1/sanity_check.json"])
    client.objects["cutsell/serverless/bench-live-1/sanity_check.json"] = json.dumps({"ok": False, "cuda_available": False})
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "SANITY_CHECK_FAILED"
    assert summary["sanity_check"]["ok"] is False


def test_benchmark_timeout_after_sanity_passes(monkeypatch, _env):
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 1
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "BENCHMARK_TIMEOUT"
    assert summary["sanity_check"]["ok"] is True


def test_benchmark_exception_reported_and_torn_down(monkeypatch, _env):
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/pod-execution-error.json"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/pod-execution-error.json"] = json.dumps({"ok": False, "error": "boom", "error_type": "RuntimeError"})
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "BENCHMARK_EXCEPTION"
    assert summary["error"]["error"] == "boom"


def test_benchmark_success_downloads_result_and_preview(monkeypatch, _env):
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(
        existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/run_output.json"]
    )
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/run_output.json"] = json.dumps(
        {
            "ok": True,
            "deliverable": True,
            "delivery_status": "DELIVERABLE",
            "result_uri": "s3://real-bucket/cutsell/serverless/bench-live-1/result.json",
            "preview_uri": "s3://real-bucket/cutsell/serverless/bench-live-1/preview.mp4",
        }
    )
    client.objects["cutsell/serverless/bench-live-1/result.json"] = json.dumps({"selected_count": 5})
    client.objects["cutsell/serverless/bench-live-1/preview.mp4"] = "fake-mp4-bytes"
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 0
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "BENCHMARK_COMPLETED"
    assert (_env / "artifact" / "video00-pod-direct.json").exists()
    assert (_env / "artifact" / "video00-pod-direct.mp4").exists()


def test_benchmark_not_deliverable_downloads_diagnostic_preview_not_the_normal_one(monkeypatch, _env):
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/run_output.json"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/run_output.json"] = json.dumps(
        {
            "ok": True,
            "deliverable": False,
            "delivery_status": "NOT_DELIVERABLE_qc_failed",
            "result_uri": "s3://real-bucket/cutsell/serverless/bench-live-1/result.json",
            "diagnostic_preview_uri": "s3://real-bucket/cutsell/serverless/bench-live-1/diagnostic-invalidated-preview.mp4",
        }
    )
    client.objects["cutsell/serverless/bench-live-1/result.json"] = json.dumps({})
    client.objects["cutsell/serverless/bench-live-1/diagnostic-invalidated-preview.mp4"] = "fake"
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 0  # ok=True even though not deliverable -- pipeline ran successfully
    assert (_env / "artifact" / "video00-pod-direct-DIAGNOSTIC-INVALIDATED.mp4").exists()
    assert not (_env / "artifact" / "video00-pod-direct.mp4").exists()


def test_template_missing_s3_config_aborts_before_polling(monkeypatch, _env):
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: {"imageName": "img", "env": {}})
    exit_code = gate.main()
    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["classification"] == "TEMPLATE_MISSING_S3_CONFIG"


def test_existing_pod_id_env_var_threaded_through(monkeypatch, _env):
    monkeypatch.setenv("EXISTING_POD_ID", "pod-reuse-me")
    captured = {}

    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        captured["existing_pod_id"] = existing_pod_id
        self.teardown_called = False
        self._pod_id = "pod-reuse-me"
        self._lifecycle_result = _FakeLifecycleResult("pod-reuse-me")
        _FakeProvider.instances.append(self)

    monkeypatch.setattr(_FakeProvider, "__init__", _init)
    _patch_s3(monkeypatch, FakeS3Client())

    gate.main()

    assert captured["existing_pod_id"] == "pod-reuse-me"


def test_qa_pod_cloud_type_env_var_forces_secure_only(monkeypatch, _env):
    # D-042 controlled SECURE-cloud test: setting QA_POD_CLOUD_TYPE=SECURE
    # must reach the Pod config as cloud_types=("SECURE",), the one
    # variable under test, not silently fall back to the default sweep.
    monkeypatch.setenv("QA_POD_CLOUD_TYPE", "SECURE")
    _patch_s3(monkeypatch, FakeS3Client())

    gate.main()

    assert _FakeProvider.instances[0].config.cloud_types == ("SECURE",)
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert summary["cloud_types_requested"] == ["SECURE"]


def test_qa_pod_cloud_type_env_var_is_case_insensitive(monkeypatch, _env):
    monkeypatch.setenv("QA_POD_CLOUD_TYPE", "secure")
    _patch_s3(monkeypatch, FakeS3Client())

    gate.main()

    assert _FakeProvider.instances[0].config.cloud_types == ("SECURE",)


def test_qa_pod_cloud_type_env_var_unset_keeps_default_sweep(monkeypatch, _env):
    _patch_s3(monkeypatch, FakeS3Client())

    gate.main()

    assert _FakeProvider.instances[0].config.cloud_types == gate.POD_CLOUD_TYPES
    summary = json.loads((_env / "pod-direct-benchmark-summary.json").read_text())
    assert "cloud_types_requested" not in summary


def test_qa_pod_cloud_type_env_var_rejects_invalid_value(monkeypatch, _env):
    monkeypatch.setenv("QA_POD_CLOUD_TYPE", "BOGUS")

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances == []  # never even fetched the template / created a Pod


def test_human_gold_reference_downloaded_best_effort_after_teardown(monkeypatch, _env):
    monkeypatch.setenv("HUMAN_GOLD_KEY", "gold/reference.mp4")
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/run_output.json", "gold/reference.mp4"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/run_output.json"] = json.dumps({"ok": True})
    client.objects["gold/reference.mp4"] = "fake-gold-bytes"
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 0
    assert (_env / "artifact" / "human-gold-video00.mp4").read_text() == "fake-gold-bytes"


def test_human_gold_download_failure_never_masks_the_real_result(monkeypatch, _env, capsys):
    monkeypatch.setenv("HUMAN_GOLD_KEY", "gold/reference.mp4")
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/run_output.json"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/run_output.json"] = json.dumps({"ok": True})
    # gold/reference.mp4 deliberately absent -- download_file raises KeyError
    _patch_s3(monkeypatch, client)

    exit_code = gate.main()

    assert exit_code == 0  # the real benchmark result is unaffected
    assert "Human Gold reference download failed" in capsys.readouterr().out


def test_no_human_gold_env_var_skips_download_silently(monkeypatch, _env):
    monkeypatch.delenv("HUMAN_GOLD_KEY", raising=False)
    prefix = "cutsell/serverless/bench-live-1"
    client = FakeS3Client(existing_keys=[f"{prefix}/sanity_check.json", f"{prefix}/run_output.json"])
    client.objects[f"{prefix}/sanity_check.json"] = json.dumps({"ok": True, "cuda_available": True})
    client.objects[f"{prefix}/run_output.json"] = json.dumps({"ok": True})
    _patch_s3(monkeypatch, client)

    gate.main()

    assert not (_env / "artifact" / "human-gold-video00.mp4").exists()


def test_summary_never_includes_raw_template_env(monkeypatch, _env):
    # Security regression, same class as D-042's live secret-leak incident:
    # this gate script must never write the template's raw env (real AWS/
    # GEMINI/etc. values) into its own summary artifact.
    _patch_s3(monkeypatch, FakeS3Client())
    gate.main()
    summary_text = (_env / "pod-direct-benchmark-summary.json").read_text()
    assert "real-key" not in summary_text  # BASE_TEMPLATE's GEMINI_API_KEY value
    assert "AWS_SECRET_ACCESS_KEY" not in summary_text
    assert "GEMINI_API_KEY" not in summary_text
