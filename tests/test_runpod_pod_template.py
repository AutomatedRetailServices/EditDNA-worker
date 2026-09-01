"""D-042 follow-up: CutSell-Pod-QA template cloned from EditDNA-Worker-2.
Fully scripted fake Transport -- no network, no RunPod credentials.
"""
from __future__ import annotations

from runpod_orchestration import TransportResponse
from runpod_pod_template import (
    PodTemplateOverrides,
    build_pod_template_payload,
    create_pod_template,
    find_template_by_name,
    list_templates,
    redact_template_env,
)


def _noop_log(event):
    pass


class FakeTemplateTransport:
    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.get_templates: list[TransportResponse] = []
        self.post_templates: list[TransportResponse] = []

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append((method, url, json_body))
        if method == "GET" and url.endswith("/templates"):
            return self.get_templates.pop(0)
        if method == "POST" and url.endswith("/templates"):
            return self.post_templates.pop(0)
        raise AssertionError(f"unscripted call: {method} {url}")


BASE_TEMPLATE = {
    "id": "07g9dovc17",
    "name": "EditDNA-Worker-2",
    "imageName": "madiator2011/better-pytorch:cuda12.4-torch2.6.0",
    "dockerStartCmd": ["/start.sh"],
    "ports": "8080/http,22/tcp",
    "containerDiskInGb": 80,
    "volumeInGb": 60,
    "volumeMountPath": "/workspace",
    "category": "NVIDIA",
    "env": {
        "AWS_ACCESS_KEY_ID": "AKIA-real-value",
        "AWS_SECRET_ACCESS_KEY": "real-secret",
        "S3_BUCKET": "editdna-prod",
        "GEMINI_API_KEY": "real-key",
    },
}


# ---------------------------------------------------------------------------
# Read-only fetch
# ---------------------------------------------------------------------------
def test_list_templates_returns_empty_on_unparseable_response():
    transport = FakeTemplateTransport()
    transport.get_templates = [TransportResponse(500, None)]
    assert list_templates(transport, "fake-key") == []


def test_find_template_by_name_returns_exact_match():
    transport = FakeTemplateTransport()
    transport.get_templates = [TransportResponse(200, [BASE_TEMPLATE, {"name": "other"}])]
    found = find_template_by_name(transport, "fake-key", "EditDNA-Worker-2")
    assert found == BASE_TEMPLATE


def test_find_template_by_name_returns_none_when_absent():
    transport = FakeTemplateTransport()
    transport.get_templates = [TransportResponse(200, [{"name": "other"}])]
    assert find_template_by_name(transport, "fake-key", "EditDNA-Worker-2") is None


def test_redact_template_env_never_leaks_values():
    redacted = redact_template_env(BASE_TEMPLATE)
    assert redacted["env"] == {
        "AWS_ACCESS_KEY_ID": "<redacted>",
        "AWS_SECRET_ACCESS_KEY": "<redacted>",
        "GEMINI_API_KEY": "<redacted>",
        "S3_BUCKET": "<redacted>",
    }
    # every other field passes through untouched
    assert redacted["imageName"] == BASE_TEMPLATE["imageName"]
    assert redacted["ports"] == BASE_TEMPLATE["ports"]
    # original dict is never mutated
    assert BASE_TEMPLATE["env"]["AWS_ACCESS_KEY_ID"] == "AKIA-real-value"


# ---------------------------------------------------------------------------
# Payload construction (pure, no network)
# ---------------------------------------------------------------------------
def test_build_payload_preserves_disk_volume_mount_category_from_base():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["containerDiskInGb"] == 80
    assert payload["volumeInGb"] == 60
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["category"] == "NVIDIA"
    assert payload["isServerless"] is False


def test_build_payload_preserves_base_start_command_when_no_override():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["dockerStartCmd"] == ["/start.sh"]


def test_build_payload_uses_explicit_start_command_override():
    overrides = PodTemplateOverrides(
        name="CutSell-Pod-QA",
        image="ghcr.io/x@sha256:new",
        start_command=["python3", "-m", "cutsell_worker.pod_job_server"],
    )
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["dockerStartCmd"] == ["python3", "-m", "cutsell_worker.pod_job_server"]


def test_build_payload_normalizes_string_ports_to_list_and_ensures_required():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["ports"] == ["8080/http", "22/tcp"]  # already present, not duplicated


def test_build_payload_appends_missing_required_port():
    base = dict(BASE_TEMPLATE, ports="22/tcp")
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(base, overrides)
    assert payload["ports"] == ["22/tcp", "8080/http"]


def test_build_payload_handles_missing_ports_field():
    base = {k: v for k, v in BASE_TEMPLATE.items() if k != "ports"}
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(base, overrides)
    assert payload["ports"] == ["8080/http"]


def test_build_payload_merges_env_overrides_over_base_never_dropping_base_keys():
    overrides = PodTemplateOverrides(
        name="CutSell-Pod-QA",
        image="ghcr.io/x@sha256:new",
        env_overrides={"CUTSELL_BRAIN_BACKEND": "runpod_local", "S3_BUCKET": "cutsell-qa"},
    )
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["env"]["AWS_ACCESS_KEY_ID"] == "AKIA-real-value"  # preserved from base
    assert payload["env"]["S3_BUCKET"] == "cutsell-qa"  # overridden
    assert payload["env"]["CUTSELL_BRAIN_BACKEND"] == "runpod_local"  # added


def test_build_payload_never_mutates_base_dict():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new", env_overrides={"X": "1"})
    original_env = dict(BASE_TEMPLATE["env"])
    build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert BASE_TEMPLATE["env"] == original_env


def test_build_payload_preserves_registry_auth_and_ssh_jupyter_flags_from_base():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["containerRegistryAuthId"] == ""
    assert payload["startSsh"] is True
    assert payload["startJupyter"] is False


def test_build_payload_sets_start_command_when_base_has_none():
    # EditDNA-Worker-2's actual live shape: no dockerStartCmd at all --
    # startup parity means we must explicitly set our own, not silently
    # omit it (which would leave the Pod running no application at all).
    base = {k: v for k, v in BASE_TEMPLATE.items() if k != "dockerStartCmd"}
    overrides = PodTemplateOverrides(
        name="CutSell-Pod-QA",
        image="ghcr.io/x@sha256:new",
        start_command=["python3", "-m", "cutsell_worker.pod_job_server"],
    )
    payload = build_pod_template_payload(base, overrides)
    assert payload["dockerStartCmd"] == ["python3", "-m", "cutsell_worker.pod_job_server"]


def test_build_payload_container_disk_override_wins_over_base():
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new", container_disk_gb=40)
    payload = build_pod_template_payload(BASE_TEMPLATE, overrides)
    assert payload["containerDiskInGb"] == 40


# ---------------------------------------------------------------------------
# create_pod_template (network boundary)
# ---------------------------------------------------------------------------
def test_create_pod_template_success():
    transport = FakeTemplateTransport()
    transport.post_templates = [TransportResponse(201, {"id": "new-tmpl-id", "name": "CutSell-Pod-QA"})]
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    template, error = create_pod_template(transport, "fake-key", base=BASE_TEMPLATE, overrides=overrides, log=_noop_log)
    assert error is None
    assert template["id"] == "new-tmpl-id"


def test_create_pod_template_never_calls_base_mutation_endpoint():
    # The only POST this function ever issues is to the generic
    # /v1/templates create endpoint -- never a PATCH/PUT against the base
    # template's own id.
    transport = FakeTemplateTransport()
    transport.post_templates = [TransportResponse(201, {"id": "new-tmpl-id"})]
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    create_pod_template(transport, "fake-key", base=BASE_TEMPLATE, overrides=overrides, log=_noop_log)
    assert transport.calls == [("POST", "https://rest.runpod.io/v1/templates", transport.calls[0][2])]
    assert "07g9dovc17" not in str(transport.calls)


def test_create_pod_template_returns_error_detail_on_failure():
    transport = FakeTemplateTransport()
    transport.post_templates = [TransportResponse(400, {"error": "bad request"})]
    overrides = PodTemplateOverrides(name="CutSell-Pod-QA", image="ghcr.io/x@sha256:new")
    template, error = create_pod_template(transport, "fake-key", base=BASE_TEMPLATE, overrides=overrides, log=_noop_log)
    assert template is None
    assert "400" in error
