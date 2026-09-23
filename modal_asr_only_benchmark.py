"""Modal ASR-only benchmark (D-053 Section 5: "Build an isolated ASR
benchmark harness"). Runs ONLY cutsell_worker.asr_only_benchmark's isolated
ASR stage -- download the same Video00 source, transcribe, build
CanonicalASREvidence, stop -- never the full canonical
serverless_handler.run_op() path. NOT a full Video00 RAW: no
AttemptReconstructor, no local-performance/vision analysis, no hybrid
editorial, no Unified Realization Resolver, no Render/QC. This makes one
ASR determinism test materially cheaper and faster than a full RAW, per
D-053 Section 5's explicit requirement.

Mirrors modal_video00_full_benchmark.py's image/app/secret pattern exactly
(same base image, same apt packages, same requirements file, same
add_local_python_source mounting technique) -- a SEPARATE Modal App
(MODAL_ASR_ONLY_APP_NAME, never MODAL_VIDEO00_APP_NAME) so its
billing/observability never mixes with the full-engine benchmark app, and a
separate, shorter timeout ceiling (DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S) since
this harness does far less work per invocation.

Import discipline: cutsell_worker.asr_only_benchmark pulls in faster-whisper
(and, indirectly, ctranslate2), neither of which are installed in the plain
`modal run` CLI process -- the import of cutsell_worker.asr_only_benchmark
happens INSIDE the remote function
body, never at this module's top level, same as
modal_video00_full_benchmark.py's own import of serverless_handler.

Return-value discipline: run_asr_only_benchmark() already returns a plain-
JSON-native dict (no torch-typed value at any key) -- the same defensive
json.loads(json.dumps(..., default=str)) round-trip is still applied here
as a second, independent guarantee.

Secrets: reuses the exact same CUTSELL_ENV_JSON_PATH_ENV mechanism
modal_video00_full_benchmark.py already established (the GitHub Actions
workflow writes the live RunPod template's own env dict to a local JSON
file; this module never hand-types runtime configuration).
"""
from __future__ import annotations

import json
import os

import modal

from modal_gpu_config import (
    CUTSELL_APT_PACKAGES,
    CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV,
    CUTSELL_BASE_IMAGE,
    CUTSELL_ENV_JSON_PATH_ENV,
    CUTSELL_REQUIREMENTS_FILE,
    CUTSELL_RUNPOD_PIP_SPEC,
    DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S,
    MODAL_ASR_ONLY_APP_NAME,
    require_modal_asr_only_timeout,
    require_modal_gpu_type,
)

MODAL_GPU_TYPE = "L4"

# Fail fast at import time if these ever drift from the approved pool --
# same discipline as modal_video00_full_benchmark.py.
require_modal_gpu_type(MODAL_GPU_TYPE)
require_modal_asr_only_timeout(DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S)

app = modal.App(MODAL_ASR_ONLY_APP_NAME)

image = (
    modal.Image.from_registry(CUTSELL_BASE_IMAGE)
    .apt_install(*CUTSELL_APT_PACKAGES)
    .pip_install_from_requirements(CUTSELL_REQUIREMENTS_FILE)
    .pip_install(CUTSELL_RUNPOD_PIP_SPEC)
    # Same "exact test head" mounting technique as
    # modal_video00_full_benchmark.py -- modal_gpu_config is mounted too
    # because this script's own top-level code imports it, and Modal
    # re-imports the entire user module remotely to hydrate the Function.
    .add_local_python_source("modal_gpu_config", "cutsell_worker")
)


def _resolve_env_secret() -> "modal.Secret":
    """Identical mechanism to modal_video00_full_benchmark.py's own
    _resolve_env_secret -- reads the same CUTSELL_ENV_JSON_PATH file the
    workflow writes from the live RunPod template's own env dict. Returns
    an empty secret (not an error) when unset, keeping this module
    importable in dry-checks that stub `modal` entirely."""
    env_json_path = os.environ.get(CUTSELL_ENV_JSON_PATH_ENV, "").strip()
    if not env_json_path:
        return modal.Secret.from_dict({})
    with open(env_json_path, "r", encoding="utf-8") as fh:
        env_dict = json.load(fh)
    if not isinstance(env_dict, dict) or not env_dict:
        raise RuntimeError(f"{env_json_path} (from {CUTSELL_ENV_JSON_PATH_ENV}) must contain a non-empty JSON object of env vars.")
    return modal.Secret.from_dict({str(k): str(v) for k, v in env_dict.items()})


cutsell_env_secret = _resolve_env_secret()


@app.function(
    gpu=MODAL_GPU_TYPE,
    image=image,
    timeout=DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S,
    retries=0,
    secrets=[cutsell_env_secret],
)
def run_asr_only_benchmark(payload: dict) -> dict:
    # Local import: cutsell_worker (faster-whisper/ctranslate2) is only ever
    # available inside the Modal container, never in the plain `modal run`
    # CLI process that defines this App/Function.
    from cutsell_worker.asr_only_benchmark import run_asr_only_benchmark as _run

    try:
        result = _run(payload)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole benchmark invocation
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    # Second, independent guarantee (belt-and-suspenders on top of
    # run_asr_only_benchmark() already returning a plain-JSON-native dict):
    # only plain JSON-native types ever cross the Modal serialization
    # boundary back to a caller that has neither torch nor cutsell_worker
    # installed locally.
    return json.loads(json.dumps(result, default=str))


@app.local_entrypoint()
def main() -> None:
    payload_json = os.environ.get(CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV, "").strip()
    if not payload_json:
        raise RuntimeError(f"{CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV} is required (JSON-encoded asr_only_benchmark payload).")
    payload = json.loads(payload_json)
    result = run_asr_only_benchmark.remote(payload)
    print(json.dumps(result, indent=2, default=str))
    with open("modal-asr-only-result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
