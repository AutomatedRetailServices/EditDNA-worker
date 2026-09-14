"""Modal full Video00 CutSell benchmark (D-043: "BUILD MODAL FULL VIDEO00
EXECUTION AND RUN ONE FULL BENCHMARK"). Resumes the exact full Video00
benchmark RunPod Serverless was supposed to execute -- NOT a new
benchmark, NOT a new editor, NOT a reduced Modal-specific pipeline. Modal
is transport only.

Conceptually: GitHub -> ModalExecutionProvider (future in-process wiring;
this script uses the same `modal run` CLI invocation the D-043 minimal
smoke test already validated live) -> Modal L4 -> SAME
`cutsell_worker.serverless_handler.run_op("focused", canonical payload)`
-> SAME result contract -> S3 artifacts -> scale-to-zero.

Image strategy: extends the already-live-validated minimal-test image
(same `modal_gpu_config.CUTSELL_BASE_IMAGE`, same
`.add_local_python_source` mounting technique) with the ACTUAL CutSell
runtime: the same apt packages, the same `requirements.cutsell.worker.txt`
(read directly by Modal's own `pip_install_from_requirements` -- not a
copied list), the same separate `runpod` pip spec, and the `cutsell_worker`
package itself mounted whole via `add_local_python_source("cutsell_worker")`.
See modal_gpu_config.py's own "full Video00 execution phase" section for
every one of these constants and the Dockerfile-consistency test that
guards them from drifting.

Import discipline (same reasoning as modal_gpu_diagnostics.py's own torch
import): `cutsell_worker` pulls in torch/mediapipe/faster-whisper/etc.,
none of which are installed in the plain `modal run` CLI process (only the
`modal` package itself is). The import of `cutsell_worker.serverless_handler`
happens INSIDE the remote function body, never at this module's top level,
so this script stays importable (and testable, with `modal` stubbed) in a
plain environment.

Return-value discipline: `serverless_handler._focused()` already returns a
small, plain-JSON-native compact summary (the exact same shape RunPod
Serverless returns) -- the FULL diagnostics tree is written by run_op()
itself to S3 as `result.json` (see `result_uri` in the returned dict), not
returned in-process. This sidesteps the D-043 DeserializationError class of
bug entirely: nothing torch-typed is ever part of the return value in the
first place. The same defensive `json.loads(json.dumps(..., default=str))`
round-trip is still applied here as a second, independent guarantee,
exactly like modal_gpu_diagnostics.collect_gpu_diagnostics().

Secrets: the actual runtime configuration (S3_BUCKET, AWS creds,
GEMINI_API_KEY, FFMPEG_BIN, and every other value the canonical Video00
path reads) is never hand-typed here. The GitHub Actions workflow
(cutsell-video00-modal-raw.yml) fetches the live RunPod template's own env
dict (the single existing source of truth, same template
runpod_pod_template.find_template_by_name already reads), masks every
value in the CI log, and writes it to a local JSON file whose path is
passed via CUTSELL_ENV_JSON_PATH. This module reads that file and builds
a `modal.Secret.from_dict(...)` from it -- values never printed, never
baked into the image, never a second static Modal Secret store to keep in
sync with the template by hand.

Execution safety: exactly one approved GPU type (L4, via
`require_modal_gpu_type`), `retries=0` (same crash-loop protection the
minimal smoke test's own live failure required), and a bounded timeout
(`modal_gpu_config.DEFAULT_MODAL_VIDEO00_TIMEOUT_S` -- mirrors RunPod
Serverless RAW's own 5400s/90-minute poll bound for this exact six-minute
source video, not a new number). No persistent container: Modal's own
scale-to-zero behavior applies the moment this one ephemeral `modal run`
invocation's function call returns.

Result persistence (D-056.1 "BENCHMARK EXECUTION RELIABILITY ONLY"): D-056
found that cancelling the local `modal run` CLI process at ANY point
before its blocking `.remote()` call returned discarded the result
permanently, even when the remote computation had already fully
succeeded -- the ONLY place the compact summary was ever written was a
local file, written only on a clean return. `run_video00_benchmark` now
also persists that same compact summary to S3 (see
`_persist_benchmark_result`, `modal_gpu_config.benchmark_result_s3_key`)
from INSIDE the remote function, before it ever returns -- independent of
whether the local process survives. `main()` checks for an
already-persisted result before dispatching (never pays for a duplicate
Modal invocation for the same `benchmark_id`) and persists a deterministic
diagnostic result on its own exception path too (e.g. a Modal timeout).
Purely additive/observability -- no editorial field of the compact
summary changes shape or meaning.
"""
from __future__ import annotations

import json
import os

import modal

from modal_gpu_config import (
    CUTSELL_APT_PACKAGES,
    CUTSELL_BASE_IMAGE,
    CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV,
    CUTSELL_ENV_JSON_PATH_ENV,
    CUTSELL_REQUIREMENTS_FILE,
    CUTSELL_RUNPOD_PIP_SPEC,
    DEFAULT_MODAL_VIDEO00_TIMEOUT_S,
    MODAL_VIDEO00_APP_NAME,
    benchmark_result_s3_key,
    require_modal_gpu_type,
    require_modal_video00_timeout,
)

MODAL_GPU_TYPE = "L4"

# Fail fast at import time if these ever drift from the approved pool --
# same discipline as modal_gpu_minimal_test.py.
require_modal_gpu_type(MODAL_GPU_TYPE)
require_modal_video00_timeout(DEFAULT_MODAL_VIDEO00_TIMEOUT_S)

app = modal.App(MODAL_VIDEO00_APP_NAME)

image = (
    modal.Image.from_registry(CUTSELL_BASE_IMAGE)
    .apt_install(*CUTSELL_APT_PACKAGES)
    .pip_install_from_requirements(CUTSELL_REQUIREMENTS_FILE)
    .pip_install(CUTSELL_RUNPOD_PIP_SPEC)
    # Mounts the ENTIRE cutsell_worker package (from this exact checked-out
    # git commit -- the workflow checks out `ref: github.sha` before this
    # script ever runs) into the remote container. This is the "exact test
    # head" guarantee for Modal: no Docker build/push/digest-pin is needed
    # the way the RunPod Serverless RAW workflow requires, because the
    # local-source mount embeds the exact files present at invocation time.
    #
    # `modal_gpu_config` is ALSO mounted here -- not just cutsell_worker --
    # because this script's own top-level code (below) imports it, and
    # Modal re-imports the entire user module remotely to hydrate the
    # Function. Omitting it crash-loops the container at import time
    # before the function body ever runs (live evidence, D-043 full
    # Video00 dispatch: three container attempts over ~90 minutes, run
    # ap-WX9iPZfMQnhPQJsM1rZwLm, function fu-UtPpmyf89ZpgT6JFPc8mDr, each
    # dying within ~2s with `ModuleNotFoundError: No module named
    # 'modal_gpu_config'`) -- the exact same class of bug already found
    # and fixed once for modal_gpu_minimal_test.py, which correctly
    # mounts its own sibling modules. Never omit a top-level sibling
    # import from this list again.
    .add_local_python_source("modal_gpu_config", "cutsell_worker")
)


def _resolve_env_secret() -> "modal.Secret":
    """Builds the injected Modal Secret from the JSON file the workflow
    wrote from the live RunPod template's own env dict. Returns an empty
    secret (not an error) when CUTSELL_ENV_JSON_PATH is unset -- keeps this
    module importable in tests/local dry-checks that stub `modal` entirely
    and never intend to actually invoke the remote function. The live
    workflow ALWAYS sets this before invoking `modal run` -- see
    cutsell-video00-modal-raw.yml."""
    env_json_path = os.environ.get(CUTSELL_ENV_JSON_PATH_ENV, "").strip()
    if not env_json_path:
        return modal.Secret.from_dict({})
    with open(env_json_path, "r", encoding="utf-8") as fh:
        env_dict = json.load(fh)
    if not isinstance(env_dict, dict) or not env_dict:
        raise RuntimeError(f"{env_json_path} (from {CUTSELL_ENV_JSON_PATH_ENV}) must contain a non-empty JSON object of env vars.")
    return modal.Secret.from_dict({str(k): str(v) for k, v in env_dict.items()})


cutsell_env_secret = _resolve_env_secret()


def _persist_benchmark_result(benchmark_id: str, result: dict) -> str | None:
    """D-056.1 item 1: persists the compact run_op() summary to S3 at the
    deterministic key `benchmark_result_s3_key(benchmark_id)`, independent
    of whether the local `modal run` CLI process survives to write its own
    `modal-video00-result.json`. This is called from INSIDE the remote
    `run_video00_benchmark` function below, before it returns to the local
    caller -- so the persisted copy exists even if the local process is
    SIGTERM'd (workflow cancellation, runner loss, a timeout) the very
    instant control returns to it. Uses the SAME AWS credentials already
    present in this remote container's environment (injected via
    `cutsell_env_secret`, built from the full RunPod template env dict --
    identical mechanism to `cutsell_worker.serverless_handler._upload_
    artifact`, never a second credential path).

    Also called from `main()` (the LOCAL process) both to persist a
    deterministic diagnostic result on a local-side exception (see item
    5(e)) and, indirectly via `_load_persisted_benchmark_result`, to avoid
    a duplicate paid dispatch (item 5(d)) -- the live workflow exports the
    same 4 AWS env vars into that step's shell before invoking `modal run`
    so this function works identically on both sides.

    Never raises: persistence is purely an additive reliability/
    observability mechanism and must never turn an otherwise-successful
    (or already-reported) benchmark result into a harder failure. Returns
    the `s3://...` URI on success, None otherwise (including when
    S3_BUCKET/boto3/benchmark_id are unavailable)."""
    benchmark_id = str(benchmark_id or "").strip()
    if not benchmark_id:
        return None
    try:
        import boto3

        bucket = str(os.environ.get("S3_BUCKET") or "").strip()
        if not bucket:
            return None
        region = str(os.environ.get("AWS_REGION") or "us-east-1")
        key = benchmark_result_s3_key(benchmark_id)
        s3 = boto3.client("s3", region_name=region)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(result, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"
    except Exception:  # noqa: BLE001 -- observability only, never fatal
        return None


def _load_persisted_benchmark_result(benchmark_id: str) -> dict | None:
    """D-056.1 items 2/5(d): reads back a previously-persisted compact
    result for this exact `benchmark_id`, if one already exists at
    `benchmark_result_s3_key(benchmark_id)`. `main()` uses this BEFORE
    dispatching to skip a duplicate paid Modal invocation for a
    benchmark_id that already has a durable result -- e.g. the workflow's
    own deterministic `BENCHMARK_ID` (known before dispatch) being reused
    by a resumed or re-invoked local wrapper. The same function is what a
    stale-GH-status recovery path relies on: since the remote function
    persists before returning (see `_persist_benchmark_result`), this can
    recover a completed result even when the local process/GitHub step
    status never reflects that completion. Never raises -- returns None on
    any failure (bucket/creds/boto3 unavailable, key not found, malformed
    JSON)."""
    benchmark_id = str(benchmark_id or "").strip()
    if not benchmark_id:
        return None
    try:
        import boto3

        bucket = str(os.environ.get("S3_BUCKET") or "").strip()
        if not bucket:
            return None
        region = str(os.environ.get("AWS_REGION") or "us-east-1")
        key = benchmark_result_s3_key(benchmark_id)
        s3 = boto3.client("s3", region_name=region)
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except Exception:  # noqa: BLE001 -- "not found yet" and "can't check" both mean: dispatch normally
        return None


@app.function(
    gpu=MODAL_GPU_TYPE,
    image=image,
    timeout=DEFAULT_MODAL_VIDEO00_TIMEOUT_S,
    retries=0,
    secrets=[cutsell_env_secret],
)
def run_video00_benchmark(payload: dict) -> dict:
    # Local import: cutsell_worker (torch/mediapipe/faster-whisper/etc.) is
    # only ever available inside the Modal container, never in the plain
    # `modal run` CLI process that defines this App/Function.
    from cutsell_worker.serverless_handler import run_op

    op = str(payload.get("op") or "focused")
    try:
        result = run_op(op, payload)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole benchmark invocation
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    # D-056.1 item 1: persist BEFORE returning -- see _persist_benchmark_
    # result's docstring for why this specific ordering is the fix.
    benchmark_id = str(payload.get("benchmark_id") or "").strip()
    persisted_uri = _persist_benchmark_result(benchmark_id, result) if benchmark_id else None

    # Second, independent guarantee (belt-and-suspenders on top of
    # _focused() already returning a plain-JSON-native compact summary):
    # only plain JSON-native types ever cross the Modal serialization
    # boundary back to a caller that has neither torch nor cutsell_worker
    # installed locally -- same discipline as modal_gpu_diagnostics.py.
    # `benchmark_result_uri` is additive observability only -- never an
    # editorial field, never consulted by any Selection/Boundary/Freeze
    # decision.
    return json.loads(json.dumps({**result, "benchmark_result_uri": persisted_uri}, default=str))


@app.local_entrypoint()
def main() -> None:
    payload_json = os.environ.get(CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV, "").strip()
    if not payload_json:
        raise RuntimeError(f"{CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV} is required (JSON-encoded run_op() payload).")
    payload = json.loads(payload_json)
    benchmark_id = str(payload.get("benchmark_id") or "").strip()

    # D-056.1 item 5(d): never dispatch a second paid Modal invocation for
    # a benchmark_id that already has a durably-persisted result.
    existing = _load_persisted_benchmark_result(benchmark_id) if benchmark_id else None
    if existing is not None:
        print(
            f"Reusing already-persisted result for benchmark_id={benchmark_id!r} -- "
            "no duplicate paid Modal dispatch."
        )
        result = existing
    else:
        try:
            result = run_video00_benchmark.remote(payload)
        except Exception as exc:  # noqa: BLE001 -- D-056.1 item 5(e): a Modal-side
            # timeout/interruption (e.g. modal.exception.FunctionTimeoutError,
            # a lost connection) must still produce a deterministic,
            # recoverable diagnostic result -- never an uncaught traceback
            # and an empty local file.
            result = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "benchmark_id": benchmark_id or None,
                "terminal_state": "local_wrapper_exception",
            }
            if benchmark_id:
                result["benchmark_result_uri"] = _persist_benchmark_result(benchmark_id, result)

    print(json.dumps(result, indent=2, default=str))
    with open("modal-video00-result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
