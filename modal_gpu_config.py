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
