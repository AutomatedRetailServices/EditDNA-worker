"""Manual entrypoint for the "CutSell Video00 Pod RAW" workflow's
health-only gate (D-042: CutSell QA GPU execution fallback -- RunPod Pod
On-Demand automation).

Reads configuration from environment variables (set by the workflow
step), runs `RunPodPodExecutionProvider.ensure_ready()` + `health_check()`,
ALWAYS tears the Pod down in `finally` regardless of outcome, and writes a
JSON summary the workflow uploads as an artifact and prints. This script
is deliberately health-only: per the standing directive, full Video00
benchmarking on a Pod is not run here and stays gated on separate,
explicit authorization -- see docs/CUTSELL_DECISIONS.md D-042.

Infrastructure/orchestration only -- no CutSell editorial logic lives
here; the canonical pipeline itself is invoked (when authorized) through
`cutsell_worker.serverless_handler.run_op`, unchanged either way.
"""
from __future__ import annotations

import json
import os
import sys

from runpod_orchestration import UrllibTransport, _default_log
from runpod_pod_provider import (
    DEFAULT_COST_CEILING_USD_PER_HR,
    PodExecutionConfig,
    RunPodPodExecutionProvider,
    fetch_pod_logs,
    get_pod,
)
from runpod_pod_template import (
    PodTemplateOverrides,
    create_pod_template,
    find_template_by_name,
    redact_template_env,
)


def _diagnose_pod_logs(api_key: str, pod_id: str) -> int:
    """Zero-cost, read-only diagnostic path: fetch state + best-effort
    container logs for an ALREADY-EXISTING Pod (typically one a prior run
    created and stopped) -- never creates or starts anything. Used to
    root-cause a Pod that was reachable at the API level but never
    answered its health endpoint, without provisioning another paid Pod
    just to keep guessing."""
    transport = UrllibTransport()
    print(f"--- diagnosing existing Pod {pod_id} (read-only, no create/start) ---")
    try:
        pod = get_pod(transport, api_key, pod_id)
    except RuntimeError as exc:
        print(f"GET pod failed: {exc}")
        pod = None
    print("--- pod state ---")
    print(json.dumps(pod, indent=2, sort_keys=True, default=str))

    url, status_code, body = fetch_pod_logs(transport, api_key, pod_id, log=_default_log)
    print(f"--- logs fetch: {url} -> http {status_code} ---")
    print(json.dumps(body, indent=2, sort_keys=True, default=str) if body is not None else "(no body)")
    return 0


def _fetch_base_template(api_key: str, name: str) -> int:
    """Read-only, zero-cost: fetch the live template config by name and
    print it with every env VALUE redacted (names preserved). Never
    creates, never mutates -- pure inspection, per D-042 follow-up step 1
    ("FETCH THE LIVE EDITDNA-WORKER-2 TEMPLATE")."""
    transport = UrllibTransport()
    print(f"--- fetching live template '{name}' (read-only) ---")
    template = find_template_by_name(transport, api_key, name)
    if template is None:
        print(f"No template named '{name}' found in this account's live template catalog.")
        return 1
    print("--- template config (env values redacted, names preserved) ---")
    print(json.dumps(redact_template_env(template), indent=2, sort_keys=True, default=str))
    return 0


def _create_qa_template(api_key: str, base_name: str, qa_name: str, image: str) -> int:
    """Clone `base_name`'s live config into a new `qa_name` template,
    overriding only image/start-command/env explicitly requested via
    QA_TEMPLATE_* env vars -- everything else is preserved verbatim from
    the base. Never mutates the base template (only ever POSTs to the
    generic /v1/templates create endpoint). Per D-042 follow-up step 2."""
    transport = UrllibTransport()
    base = find_template_by_name(transport, api_key, base_name)
    if base is None:
        print(f"Base template '{base_name}' not found -- refusing to guess a configuration. Aborting.")
        return 1

    start_command_raw = os.environ.get("QA_TEMPLATE_START_COMMAND") or ""
    start_command = start_command_raw.split() if start_command_raw else None
    env_overrides_raw = os.environ.get("QA_TEMPLATE_ENV_OVERRIDES_JSON") or "{}"
    try:
        env_overrides = json.loads(env_overrides_raw)
    except ValueError as exc:
        print(f"QA_TEMPLATE_ENV_OVERRIDES_JSON is not valid JSON: {exc}")
        return 1
    if not isinstance(env_overrides, dict):
        print("QA_TEMPLATE_ENV_OVERRIDES_JSON must be a JSON object.")
        return 1

    overrides = PodTemplateOverrides(
        name=qa_name,
        image=image,
        start_command=start_command,
        env_overrides=env_overrides,
    )
    print(f"--- creating '{qa_name}' from live base '{base_name}' ({base.get('id')}) ---")
    print("--- base config used (env values redacted) ---")
    print(json.dumps(redact_template_env(base), indent=2, sort_keys=True, default=str))
    template, error = create_pod_template(transport, api_key, base=base, overrides=overrides, log=_default_log)
    if template is None:
        print(f"Template creation failed: {error}")
        return 1
    print("--- created template (env values redacted) ---")
    print(json.dumps(redact_template_env(template), indent=2, sort_keys=True, default=str))
    return 0


def main() -> int:
    diagnose_pod_id = os.environ.get("DIAGNOSE_POD_LOGS_ID") or None
    if diagnose_pod_id:
        return _diagnose_pod_logs(os.environ["RUNPOD_API_KEY"], diagnose_pod_id)

    template_action = os.environ.get("TEMPLATE_ACTION") or None
    if template_action == "fetch_base":
        return _fetch_base_template(
            os.environ["RUNPOD_API_KEY"], os.environ.get("BASE_TEMPLATE_NAME", "EditDNA-Worker-2")
        )
    if template_action == "create_qa_template":
        return _create_qa_template(
            os.environ["RUNPOD_API_KEY"],
            os.environ.get("BASE_TEMPLATE_NAME", "EditDNA-Worker-2"),
            os.environ.get("QA_TEMPLATE_NAME", "CutSell-Pod-QA"),
            os.environ["QA_TEMPLATE_IMAGE"],
        )

    api_key = os.environ["RUNPOD_API_KEY"]
    existing_pod_id = os.environ.get("EXISTING_POD_ID") or None
    cost_ceiling = float(os.environ.get("QA_POD_COST_CEILING_USD_PER_HR", str(DEFAULT_COST_CEILING_USD_PER_HR)))

    # D-042 Step 7: when POD_TEMPLATE_ID is set, the Pod is created FROM
    # that RunPod template (e.g. CutSell-Pod-QA) -- image/ports/env/
    # dockerStartCmd/disk all come from the template itself and are never
    # sent by create_pod() in this mode (see its docstring). POD_IMAGE is
    # then purely informational/logging -- this path deliberately does not
    # require a freshly-built image just to run a template-based test.
    template_id = os.environ.get("POD_TEMPLATE_ID") or None
    if template_id:
        image = os.environ.get("POD_IMAGE") or "<inherited from template>"
        start_command = os.environ.get("POD_START_COMMAND") or None
    else:
        image = os.environ["POD_IMAGE"]
        start_command = os.environ.get("POD_START_COMMAND") or "python3 -m cutsell_worker.pod_job_server"

    config = PodExecutionConfig(
        api_key=api_key,
        image=image,
        pod_name=os.environ.get("POD_NAME", "cutsell-qa-pod"),
        start_command=start_command,
        cost_ceiling_usd_per_hr=cost_ceiling,
        template_id=template_id,
    )
    provider = RunPodPodExecutionProvider(
        UrllibTransport(),
        config,
        existing_pod_id=existing_pod_id,
        log=_default_log,
    )

    summary: dict = {"execution_provider": "RUNPOD_POD", "template_id": template_id}
    exit_code = 1
    try:
        result = provider.health_check()
        summary.update(
            {
                "passed": result.passed,
                "classification": result.classification,
                "elapsed_s": result.elapsed_s,
                "detail": result.detail,
            }
        )
        exit_code = 0 if result.passed else 1
    finally:
        # Guaranteed cleanup: this runs whether health_check passed,
        # failed, or raised. provider.teardown() itself never raises.
        provider.teardown()
        summary["pod_id"] = provider.pod_id
        with open("pod-health-summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print("--- pod-health-summary.json ---")
        print(json.dumps(summary, indent=2, sort_keys=True))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
