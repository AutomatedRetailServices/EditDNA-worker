"""Shared, modal-package-free configuration/validation for the Modal GPU
execution backend (D-043: CutSell Modal GPU execution -- first live
validation). Deliberately has NO dependency on the `modal` package itself
(unlike `modal_gpu_minimal_test.py`, which needs it to actually define an
App/Function) -- every constant and validator here is plain Python so it
stays importable and unit-testable in any environment, exactly like this
repo's existing RunPod modules never require the `runpod` package for
their own config/validation logic.

Per the standing cost-safety directive: this phase approves exactly ONE
GPU type (L4) and nothing else. Widening the pool (A100/H100/H200/L40S)
requires a future, explicit authorization -- this module hard-rejects
them rather than silently accepting whatever string a caller passes, the
same "approved pool, not a free-for-all" posture as
`runpod_pod_provider.APPROVED_POD_GPU_TYPE_IDS`.
"""
from __future__ import annotations

from typing import Mapping

# Cost-safety allowlist (D-043 first live validation phase). Any GPU type
# outside this tuple is rejected -- see require_modal_gpu_type below.
APPROVED_MODAL_GPU_TYPES: tuple[str, ...] = ("L4",)

# Named explicitly (not just "everything else") so a reader sees exactly
# which higher-cost types this phase deliberately excludes, mirroring the
# RunPod Pod provider's own EXCLUDED_POD_GPU_TYPE_IDS precedent.
EXCLUDED_MODAL_GPU_TYPES: tuple[str, ...] = ("A100", "A100-80GB", "H100", "H200", "L40S")

# Bounded smoke-test timeout: short enough that a hung function cannot run
# up meaningful cost, per "keep the function bounded with a short
# timeout." DEFAULT is what modal_gpu_minimal_test.py actually uses;
# MAX_SMOKE_TEST_TIMEOUT_S is a sanity ceiling this phase's tooling
# refuses to exceed even if a caller asks for more -- Video00-scale
# timeouts are out of scope until that integration is authorized.
DEFAULT_MODAL_TIMEOUT_S: int = 300
MAX_MODAL_SMOKE_TEST_TIMEOUT_S: int = 600

# Modal's own container-idle-shutdown window (formerly
# `container_idle_timeout`, now `scaledown_window` in current Modal SDK
# versions) -- kept short so no idle container lingers (billed) after the
# function returns, on top of Modal's own scale-to-zero default behavior.
MODAL_SCALEDOWN_WINDOW_S: int = 30

# Reused verbatim from Dockerfile.cutsell.serverless's own FROM line --
# see D-043's image/runtime audit: this phase deliberately reuses the
# EXACT SAME base image (same torch version, same CUDA version) rather
# than inventing a divergent Modal-specific environment. ffmpeg is
# installed the same way the Dockerfile installs it (apt-get ffmpeg),
# not a different version pulled from a different source.
CUTSELL_BASE_IMAGE = "madiator2011/better-pytorch:cuda12.4-torch2.6.0"

MODAL_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
MODAL_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"

EXECUTION_PROVIDER_MODAL = "MODAL"

# --- D-043 full Video00 execution phase (separately authorized) ---------
# The minimal smoke test above deliberately never installed the full
# CutSell runtime; this phase extends the SAME validated base image to run
# the actual canonical engine. Every constant below is reused verbatim by
# modal_video00_full_benchmark.py -- never a second, hand-typed dependency
# list drifting from Dockerfile.cutsell.serverless. Order matches the
# Dockerfile's own apt-get install block; a dedicated test
# (tests/test_modal_video00_full_benchmark.py) parses that Dockerfile and
# asserts this tuple stays in lockstep with it.
CUTSELL_APT_PACKAGES: tuple[str, ...] = (
    "build-essential",
    "python3-dev",
    "ffmpeg",
    "git",
    "curl",
    "pkg-config",
    "libavformat-dev",
    "libavcodec-dev",
    "libavdevice-dev",
    "libavutil-dev",
    "libavfilter-dev",
    "libswscale-dev",
    "libswresample-dev",
)

# The actual requirements FILE is read at Modal image-build time (Modal's
# own `pip_install_from_requirements`) -- this is the file path, not a
# copy of its contents, so there is nothing here that can drift from the
# canonical dependency list itself.
CUTSELL_REQUIREMENTS_FILE = "requirements.cutsell.worker.txt"

# Reused verbatim from Dockerfile.cutsell.serverless's own separate
# `pip install 'runpod>=1.7,<2'` step (kept out of
# requirements.cutsell.worker.txt in the Dockerfile too -- see that file's
# own header comment). serverless_handler.py imports `runpod` unconditionally
# at module level even though run_op() itself never calls it.
CUTSELL_RUNPOD_PIP_SPEC = "runpod>=1.7,<2"

# Full Video00 benchmark timeout: mirrors RunPod Serverless RAW's own
# poll bound (5400s / 90 minutes, cutsell-video00-raw-v5-auto-microtrim.yml)
# rather than inventing a new number for the same six-minute source video.
# Kept as a SEPARATE ceiling from MAX_MODAL_SMOKE_TEST_TIMEOUT_S above --
# that one stays scoped to the minimal smoke test and is never widened by
# this phase's own needs.
DEFAULT_MODAL_VIDEO00_TIMEOUT_S: int = 5400
MAX_MODAL_VIDEO00_TIMEOUT_S: int = 5400

MODAL_VIDEO00_APP_NAME = "cutsell-video00-modal-benchmark"

# Env var naming the local JSON file (built by the GitHub Actions workflow
# from the live RunPod template's own env dict -- never a second,
# hand-typed env list) that modal_video00_full_benchmark.py reads to build
# its injected Modal Secret.
CUTSELL_ENV_JSON_PATH_ENV = "CUTSELL_ENV_JSON_PATH"

# Env var naming the canonical run_op() payload (JSON-encoded), reusing the
# exact same env var name runpod_pod_direct_benchmark_gate.py already
# established for the equivalent RunPod Pod direct-execution path.
CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV = "CUTSELL_BENCHMARK_PAYLOAD_JSON"

# --- D-053 ASR-only isolated benchmark (separately authorized, NOT a full
# Video00 RAW) ---------------------------------------------------------
# A dedicated, materially cheaper/faster Modal app: it runs ONLY
# cutsell_worker.asr_only_benchmark.run_asr_only_benchmark (download source,
# transcribe, build CanonicalASREvidence, stop) -- never
# serverless_handler.run_op(), never AttemptReconstructor/hybrid editorial/
# resolver/render. A separate app name (never reusing
# MODAL_VIDEO00_APP_NAME) keeps its billing/observability distinct from the
# full-engine benchmark app.
MODAL_ASR_ONLY_APP_NAME = "cutsell-asr-only-modal-benchmark"

# ASR-only transcription of one ~6 minute source is a small fraction of a
# full Video00 RAW's runtime (no local-performance/vision analysis, no
# hybrid editorial, no render/QC) -- a short, separate ceiling keeps a hung
# ASR-only call bounded without borrowing the full-engine timeout.
DEFAULT_MODAL_ASR_ONLY_TIMEOUT_S: int = 900
MAX_MODAL_ASR_ONLY_TIMEOUT_S: int = 900

# Env var naming the canonical asr_only_benchmark payload (JSON-encoded),
# kept separate from CUTSELL_BENCHMARK_PAYLOAD_JSON_ENV so an ASR-only
# dispatch can never be confused with (or accidentally reuse) a full
# Video00 payload shape.
CUTSELL_ASR_ONLY_PAYLOAD_JSON_ENV = "CUTSELL_ASR_ONLY_PAYLOAD_JSON"

# --- D-056.1 "BENCHMARK EXECUTION RELIABILITY ONLY" ----------------------
# D-056's Run C dispatch saga (docs/CUTSELL_DECISIONS.md D-056 Section 8)
# root-caused the failure to a single structural gap: the ONLY place a
# Modal full-Video00 benchmark's compact result was ever persisted was a
# local file (`modal-video00-result.json`), written by the local `modal
# run` CLI process ONLY after its blocking `.remote()` call returned
# cleanly. A SIGTERM to that local process at any point before that write
# -- even after the remote computation had already fully succeeded --
# permanently discarded the result. This prefix/key names the SEPARATE,
# durable S3 location `modal_video00_full_benchmark.py` now persists that
# same compact result to, from INSIDE the remote function itself, before
# it ever returns to the local caller -- independent of whether that local
# process survives.
#
# Deliberately a different namespace from
# `cutsell_worker.serverless_handler._focused()`'s own
# `cutsell/serverless/{benchmark_id}/result.json` upload: that key holds
# the FULL diagnostics tree (run_op()'s own durable-persistence
# convention, unchanged here); this key holds only the small compact
# summary dict (the exact same shape the Modal function itself returns),
# so a workflow-side poll/read never has to guess which of the two it is
# looking at.
CUTSELL_BENCHMARK_RESULT_S3_PREFIX = "cutsell/benchmark-results"


def benchmark_result_s3_key(benchmark_id: str) -> str:
    """Deterministic S3 key for a durably-persisted compact benchmark
    result. Keyed off the SAME `benchmark_id` the calling workflow already
    computes BEFORE ever dispatching (e.g. `BENCHMARK_ID: video00-modal-
    ${{ github.run_id }}-${{ github.run_attempt }}` in
    cutsell-video00-modal-raw.yml) -- so both the Modal remote function
    (the writer) and the GitHub Actions workflow or local wrapper (the
    readers) can derive the identical key independently, with no round
    trip through a function return value or a local file either side
    might never see. Sanitized the same way
    `cutsell_worker.serverless_handler._safe_id` sanitizes a benchmark_id
    for use in an S3 key, so this never rejects a benchmark_id `run_op()`
    itself would have accepted."""
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(benchmark_id or "").strip())[:100]
    if not safe_id:
        raise ValueError("benchmark_id is required to compute a benchmark_result_s3_key.")
    return f"{CUTSELL_BENCHMARK_RESULT_S3_PREFIX}/{safe_id}/compact-result.json"


def require_modal_asr_only_timeout(timeout_s: float) -> None:
    """Raises ValueError for a non-positive or excessive timeout. Separate
    from both require_modal_timeout and require_modal_video00_timeout so
    widening one never silently widens another."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}.")
    if timeout_s > MAX_MODAL_ASR_ONLY_TIMEOUT_S:
        raise ValueError(
            f"timeout_s={timeout_s!r} exceeds the ASR-only benchmark ceiling of "
            f"{MAX_MODAL_ASR_ONLY_TIMEOUT_S}s."
        )


def require_modal_video00_timeout(timeout_s: float) -> None:
    """Raises ValueError for a non-positive or excessive timeout. Separate
    from require_modal_timeout (the minimal-smoke-test ceiling) so widening
    one never silently widens the other."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}.")
    if timeout_s > MAX_MODAL_VIDEO00_TIMEOUT_S:
        raise ValueError(
            f"timeout_s={timeout_s!r} exceeds the full Video00 benchmark ceiling of "
            f"{MAX_MODAL_VIDEO00_TIMEOUT_S}s."
        )


def require_modal_gpu_type(gpu_type: str) -> None:
    """Raises ValueError for any GPU type outside the approved pool.
    Never silently substitutes or upgrades -- a caller asking for an
    excluded type gets a clear, named refusal, not a different GPU."""
    if gpu_type not in APPROVED_MODAL_GPU_TYPES:
        if gpu_type in EXCLUDED_MODAL_GPU_TYPES:
            raise ValueError(
                f"GPU type {gpu_type!r} is explicitly excluded for this phase "
                f"(approved pool: {APPROVED_MODAL_GPU_TYPES}) -- requires explicit "
                f"future authorization before use, per the standing cost-safety directive."
            )
        raise ValueError(f"GPU type {gpu_type!r} is not in the approved pool {APPROVED_MODAL_GPU_TYPES}.")


def require_modal_timeout(timeout_s: float) -> None:
    """Raises ValueError for a non-positive or excessive timeout. Keeps
    every Modal function call for this phase genuinely bounded."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}.")
    if timeout_s > MAX_MODAL_SMOKE_TEST_TIMEOUT_S:
        raise ValueError(
            f"timeout_s={timeout_s!r} exceeds this phase's ceiling of "
            f"{MAX_MODAL_SMOKE_TEST_TIMEOUT_S}s -- Video00-scale timeouts are out of "
            f"scope until that integration is separately authorized."
        )


def require_modal_token_env(env: Mapping[str, str]) -> None:
    """Raises RuntimeError naming exactly which required token env var(s)
    are missing or empty. Never guesses/falls back to an unauthenticated
    call -- Modal auth must be explicit."""
    missing = [name for name in (MODAL_TOKEN_ID_ENV, MODAL_TOKEN_SECRET_ENV) if not (env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(f"Missing required Modal auth env var(s): {', '.join(missing)}.")
