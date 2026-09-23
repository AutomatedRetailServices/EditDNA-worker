"""RunPod Pod DIRECT-EXECUTION benchmark gate (D-042 follow-up: "restore
the known-working execution model" -- bypasses `cutsell_worker.
pod_job_server`/HTTP port 8080 entirely for the QA benchmark path, per
the standing directive not to keep debugging that transport).

Lifecycle: fetch the live CutSell-Pod-QA template (read-only) -> build an
INLINE Pod config from it (image/env/disk inherited verbatim, per D-042's
"prefer exact inheritance" precedent, plus this run's own
CUTSELL_BENCHMARK_PAYLOAD_JSON) -> `RunPodPodExecutionProvider`'s already-
tested reuse/restart/create/GPU-search/cost-ceiling lifecycle
(`ensure_ready()`, completely unchanged -- no HTTP health check here) ->
poll S3 (never HTTP, never container logs) for the entrypoint's own
`sanity_check.json`, then for its terminal `run_output.json`/
`pod-execution-error.json` -> download the result JSON + rendered MP4 ->
ALWAYS `teardown()` in `finally`, exactly like the health-only gate's own
guaranteed-stop pattern.

Deliberately does NOT use `template_id=` on `PodExecutionConfig`: that
combination (a Pod created from a template AND per-run env/start-command
overrides) has never been exercised against RunPod's live schema. The
inline path (image/env/disk supplied directly, read from the live
template so nothing is reinvented) is the one every earlier D-042 live
Pod test has already proven works end to end.

No editorial/pipeline code lives here -- the canonical CutSell pipeline
runs entirely inside the Pod via
`cutsell_worker.pod_direct_benchmark_entrypoint`, which itself only calls
`cutsell_worker.serverless_handler.run_op(op, payload)`, the exact same
dispatcher RunPod Serverless uses. This script is infrastructure/
orchestration only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from runpod_orchestration import UrllibTransport, _default_log
from runpod_pod_provider import (
    DEFAULT_COST_CEILING_USD_PER_HR,
    POD_CLOUD_TYPES,
    PodExecutionConfig,
    RunPodPodExecutionProvider,
)
from runpod_pod_template import find_template_by_name


def build_direct_exec_config(
    template: dict,
    *,
    api_key: str,
    pod_name: str,
    payload: dict,
    cost_ceiling_usd_per_hr: float,
    cloud_types: Optional[tuple[str, ...]] = None,
) -> PodExecutionConfig:
    """Pure function: the live template's own image/env/disk are inherited
    verbatim (never reinvented); the per-run payload is added as one extra
    env var. Never mutates `template`.

    `cloud_types` defaults (None) to PodExecutionConfig's own default --
    the existing COMMUNITY-then-SECURE sweep, unchanged for every other
    caller. A caller running a controlled single-variable cloud-type test
    (D-042: comparing COMMUNITY vs. SECURE container-execution behavior
    directly) passes an explicit narrowed tuple, e.g. ("SECURE",)."""
    env = dict(template.get("env") or {})
    env["CUTSELL_BENCHMARK_PAYLOAD_JSON"] = json.dumps(payload)
    kwargs = dict(
        api_key=api_key,
        image=str(template.get("imageName") or ""),
        pod_name=pod_name,
        start_command="python3 -m cutsell_worker.pod_direct_benchmark_entrypoint",
        container_disk_gb=int(template.get("containerDiskInGb") or 80),
        env=env,
        cost_ceiling_usd_per_hr=cost_ceiling_usd_per_hr,
    )
    if cloud_types is not None:
        kwargs["cloud_types"] = cloud_types
    return PodExecutionConfig(**kwargs)


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    """True iff the object exists. Any error other than a clean "not
    found" shape propagates -- a real permissions/config problem must
    never be silently treated as "still running"."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001 -- boto3 raises ClientError; kept generic so fakes in tests need not import botocore
        response = getattr(exc, "response", None) or {}
        code = str((response.get("Error") or {}).get("Code") or "")
        status = str((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or "")
        if code in ("404", "NoSuchKey", "NotFound") or status == "404":
            return False
        raise


def poll_for_first_existing_key(
    s3_client,
    bucket: str,
    keys: list[str],
    *,
    timeout_s: float,
    interval_s: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[str]:
    """Polls `keys` (checked in order, first match wins) every
    `interval_s` until one exists or `timeout_s` elapses. Returns the
    found key, or None on timeout -- never raises on a plain timeout."""
    start = now()
    while True:
        for key in keys:
            if s3_key_exists(s3_client, bucket, key):
                return key
        if now() - start >= timeout_s:
            return None
        sleep(interval_s)


def _download_s3_uri(s3_client, uri: str, local_path: str) -> None:
    assert uri.startswith("s3://"), f"not an s3 URI: {uri}"
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, local_path)


def _make_s3_client(*, region: str, access_key: str, secret_key: str):
    import boto3  # local import: keeps this module importable without boto3 for pure-function tests

    return boto3.client("s3", region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    existing_pod_id = os.environ.get("EXISTING_POD_ID") or None
    cost_ceiling = float(os.environ.get("QA_POD_COST_CEILING_USD_PER_HR", str(DEFAULT_COST_CEILING_USD_PER_HR)))
    template_name = os.environ.get("QA_TEMPLATE_NAME", "CutSell-Pod-QA")
    benchmark_id = os.environ["BENCHMARK_ID"]
    source_key = os.environ["SOURCE_KEY"]
    op = os.environ.get("CUTSELL_BENCHMARK_OP", "focused")
    auto_microtrim = (os.environ.get("CUTSELL_AUTO_MICROTRIM", "true").strip().lower() == "true")
    sanity_timeout_s = float(os.environ.get("SANITY_TIMEOUT_S", "300"))
    benchmark_timeout_s = float(os.environ.get("BENCHMARK_TIMEOUT_S", "5400"))
    sanity_poll_interval_s = float(os.environ.get("SANITY_POLL_INTERVAL_S", "10"))
    benchmark_poll_interval_s = float(os.environ.get("BENCHMARK_POLL_INTERVAL_S", "15"))
    # D-042 controlled SECURE-cloud test: unset (default) preserves the
    # existing COMMUNITY-then-SECURE sweep for every other caller of this
    # script. Set to force a single cloud type as the ONE variable under
    # test -- e.g. "SECURE" -- so the Pod created is genuinely that cloud
    # type only, never landing back on COMMUNITY first.
    qa_pod_cloud_type = (os.environ.get("QA_POD_CLOUD_TYPE") or "").strip().upper()
    cloud_types: Optional[tuple[str, ...]] = None
    if qa_pod_cloud_type:
        if qa_pod_cloud_type not in POD_CLOUD_TYPES:
            print(
                f"QA_POD_CLOUD_TYPE={qa_pod_cloud_type!r} is not one of {POD_CLOUD_TYPES} -- refusing to guess. Aborting.",
                flush=True,
            )
            return 1
        cloud_types = (qa_pod_cloud_type,)

    transport = UrllibTransport()
    template = find_template_by_name(transport, api_key, template_name)
    summary: dict = {"benchmark_id": benchmark_id, "template_name": template_name}
    if template is None:
        summary["classification"] = "TEMPLATE_NOT_FOUND"
        print(f"Template '{template_name}' not found -- refusing to guess a configuration. Aborting.", flush=True)
        Path("pod-direct-benchmark-summary.json").write_text(json.dumps(summary, indent=2))
        return 1

    if cloud_types is not None:
        summary["cloud_types_requested"] = list(cloud_types)

    payload = {
        "op": op,
        "source_key": source_key,
        "benchmark_id": benchmark_id,
        "auto_speech_visual_microtrim": auto_microtrim,
    }
    config = build_direct_exec_config(
        template,
        api_key=api_key,
        pod_name=os.environ.get("POD_NAME", "cutsell-qa-pod-direct"),
        payload=payload,
        cost_ceiling_usd_per_hr=cost_ceiling,
        cloud_types=cloud_types,
    )
    provider = RunPodPodExecutionProvider(transport, config, existing_pod_id=existing_pod_id, log=_default_log)

    template_env = template.get("env") or {}
    bucket = str(template_env.get("S3_BUCKET") or "")
    region = str(template_env.get("AWS_REGION") or "us-east-1")
    aws_access_key = str(template_env.get("AWS_ACCESS_KEY_ID") or "")
    aws_secret_key = str(template_env.get("AWS_SECRET_ACCESS_KEY") or "")
    human_gold_key = os.environ.get("HUMAN_GOLD_KEY") or ""

    # Declared before the try so `finally` can safely check them even if
    # the S3-config guard below returns early -- the Human Gold download
    # (QA-only, never fed into production logic) always goes through this
    # script's own boto3 client, never through a shell step, so raw AWS
    # credentials are never routed through bash/env interpolation at all.
    s3_client = None
    exit_code = 1
    try:
        if not (bucket and aws_access_key and aws_secret_key):
            summary["classification"] = "TEMPLATE_MISSING_S3_CONFIG"
            print("Template is missing S3_BUCKET/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY -- cannot poll for results.", flush=True)
            return exit_code

        s3_client = _make_s3_client(region=region, access_key=aws_access_key, secret_key=aws_secret_key)
        prefix = f"cutsell/serverless/{benchmark_id}"

        lifecycle = provider.ensure_ready()
        summary["pod_id"] = lifecycle.pod_id
        summary["lifecycle_classification"] = lifecycle.classification
        if not lifecycle.pod_id:
            summary["classification"] = "POD_LIFECYCLE_FAILED"
            print("Pod lifecycle failed before any container could run.", flush=True)
            return exit_code

        print(f"--- [pod-direct-gate] polling S3 for sanity_check.json (bounded {sanity_timeout_s}s) ---", flush=True)
        found = poll_for_first_existing_key(
            s3_client, bucket, [f"{prefix}/sanity_check.json"], timeout_s=sanity_timeout_s, interval_s=sanity_poll_interval_s,
        )
        if not found:
            summary["classification"] = "SANITY_CHECK_TIMEOUT"
            print("Sanity check never appeared in S3 within the bound -- container likely never became runtime-ready.", flush=True)
            return exit_code
        s3_client.download_file(bucket, found, "sanity_check.json")
        sanity = json.loads(Path("sanity_check.json").read_text())
        summary["sanity_check"] = sanity
        print(json.dumps(sanity, indent=2), flush=True)
        if not sanity.get("ok"):
            summary["classification"] = "SANITY_CHECK_FAILED"
            print("Sanity checks failed inside the Pod -- not proceeding to Video00.", flush=True)
            return exit_code

        print(f"--- [pod-direct-gate] sanity passed; polling S3 for run_output.json/pod-execution-error.json (bounded {benchmark_timeout_s}s) ---", flush=True)
        found = poll_for_first_existing_key(
            s3_client,
            bucket,
            [f"{prefix}/run_output.json", f"{prefix}/pod-execution-error.json"],
            timeout_s=benchmark_timeout_s,
            interval_s=benchmark_poll_interval_s,
        )
        if not found:
            summary["classification"] = "BENCHMARK_TIMEOUT"
            print("Benchmark never reached a terminal state in S3 within the bound.", flush=True)
            return exit_code

        if found.endswith("pod-execution-error.json"):
            s3_client.download_file(bucket, found, "pod-execution-error.json")
            error = json.loads(Path("pod-execution-error.json").read_text())
            summary["classification"] = "BENCHMARK_EXCEPTION"
            summary["error"] = error
            print(json.dumps(error, indent=2), flush=True)
            return exit_code

        s3_client.download_file(bucket, found, "run_output.json")
        run_output = json.loads(Path("run_output.json").read_text())
        summary["run_output"] = run_output
        ok = bool(run_output.get("ok"))
        summary["classification"] = "BENCHMARK_COMPLETED" if ok else "BENCHMARK_NOT_OK"
        exit_code = 0 if ok else 1

        if run_output.get("result_uri"):
            _download_s3_uri(s3_client, run_output["result_uri"], "artifact/video00-pod-direct.json")
        if run_output.get("preview_uri"):
            _download_s3_uri(s3_client, run_output["preview_uri"], "artifact/video00-pod-direct.mp4")
        elif run_output.get("diagnostic_preview_uri"):
            _download_s3_uri(
                s3_client, run_output["diagnostic_preview_uri"], "artifact/video00-pod-direct-DIAGNOSTIC-INVALIDATED.mp4"
            )
        return exit_code
    finally:
        # Guaranteed cleanup: runs whether the benchmark passed, failed, or
        # this function raised. provider.teardown() itself never raises.
        provider.teardown()
        # QA-only Human Gold reference download (Watch+Listen comparison,
        # never fed into production Selection/Boundary logic) -- best-
        # effort, through this script's own boto3 client so raw AWS
        # credentials never pass through a shell step for this workflow.
        if s3_client is not None and human_gold_key and bucket:
            try:
                _download_s3_uri(s3_client, f"s3://{bucket}/{human_gold_key}", "artifact/human-gold-video00.mp4")
            except Exception as exc:  # noqa: BLE001 -- never let this mask the real result
                print(f"Human Gold reference download failed (non-blocking): {exc}", flush=True)
        summary["pod_id"] = summary.get("pod_id") or provider.pod_id
        Path("pod-direct-benchmark-summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("--- pod-direct-benchmark-summary.json ---", flush=True)
        print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
