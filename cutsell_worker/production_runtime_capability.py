"""D-274C: Production Runtime Capability contract.

A DIFFERENT concern from D-271's `source_media_profile.LocalFfmpegCapability
Snapshot` (which is a per-SOURCE-probe-run artifact, permanently labelled
`LOCAL_SANDBOX_ONLY_NOT_PRODUCTION_VERIFIED`, and from `source_format_policy.
RuntimeCapabilityInput` (D-272's own pure policy-input type, which every
caller in this codebase still constructs with its all-`False` defaults --
Stage 28's own "no caller ... constructs a non-default instance today").

This module is the WORKER-PROCESS-LEVEL capability contract Stage 1 asks
for: what should a real, deployed worker container measure about itself
ONCE, at process startup, before it can safely claim HEVC decode capability
to D-272's policy layer. It reuses (never duplicates) D-271's own bounded,
real-subprocess ffmpeg decoder/encoder inspection mechanism (`source_media_
profile.capture_local_ffmpeg_capability`) -- the actual bytes-on-the-wire
probe technique is identical; what differs is WHO is asking and what the
result is honestly labelled.

Honest scope of this gate (Stage 1's own "Capability must represent the
ACTUAL worker runtime. Do not use developer-sandbox capability as
production truth"): this module defines the CONTRACT and the MECHANISM,
and proves both work correctly wherever they run (including this sandbox,
which genuinely has the mechanism's own real ffmpeg/ffprobe binaries). It
does NOT, and cannot inside this offline gate, prove that the REAL deployed
worker container (`Dockerfile.cutsell.worker`'s own base image) reports
`hevc_decoder_available=True` -- building/running that CUDA-based image is
itself paid/heavy compute this gate's own banners forbid, and no live
worker dispatch is authorized. `capture_production_worker_capability` is
therefore the function a REAL worker startup sequence (`rq_worker.py`'s own
`run_worker()`, per this gate's own Stage 2 seam-identification) WOULD call
-- written, tested, and ready -- but this gate deliberately does NOT wire
it into `rq_worker.py`/`entrypoint.sh` (that wiring is itself a form of
live activation the Primary Objective's own "No live worker activation"
excludes from this gate). See docs/CUTSELL_DECISIONS.md D-274C for the
full disclosure.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from . import source_media_profile as smp
from . import source_format_policy as sfp

# =============================================================================
# Stage 1 -- typed production capability contract
# =============================================================================

# The two honest capability-source labels this module ever produces. Never a
# third, ad-hoc string -- any caller asserting production truth must go
# through `capture_production_worker_capability` and accept its label.
CAPABILITY_SOURCE_LOCAL_SANDBOX = smp.RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY
CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK = "PRODUCTION_STARTUP_SELF_CHECK"

# Stage 1's own explicit disclosure: this gate builds and proves the
# MECHANISM: it never asserts that mechanism has actually been executed
# inside the real, deployed production worker container this session (no
# docker build/run of the real CUDA worker image is authorized here -- see
# module docstring). A caller inspecting a `ProductionRuntimeCapability`
# built during THIS gate's own tests/offline proof must read this field,
# not `capability_source` alone, to know whether production truth was ever
# actually established.
PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED = "PRODUCTION_CAPABILITY_NOT_YET_ESTABLISHED"
PRODUCTION_VERIFICATION_STATUS_ESTABLISHED = "PRODUCTION_CAPABILITY_ESTABLISHED"


@dataclass(frozen=True)
class ProductionRuntimeCapability:
    """D-274C Stage 1: one immutable snapshot of what a worker RUNTIME
    process can actually do -- decode HEVC, encode canonical H264 -- plus
    enough provenance (`capability_source`, `production_verification_
    status`, `ffmpeg_version`) that a caller can tell a genuine, deployed-
    container startup result apart from an offline/sandbox proof of the
    same mechanism. Never fabricates `True` from the executable merely
    existing (Stage 3's own explicit warning) -- always derived from a
    real decoder/encoder listing."""

    hevc_decoder_available: bool
    h264_encoder_available: bool
    ffmpeg_version: str | None
    capability_source: str
    production_verification_status: str
    errors: tuple[str, ...] = ()


def _snapshot_to_capability(
    snapshot: "smp.LocalFfmpegCapabilitySnapshot",
    *,
    capability_source: str,
    production_verification_status: str,
) -> ProductionRuntimeCapability:
    return ProductionRuntimeCapability(
        hevc_decoder_available=snapshot.hevc_decoder_present,
        h264_encoder_available=snapshot.libx264_present,
        ffmpeg_version=snapshot.ffmpeg_version,
        capability_source=capability_source,
        production_verification_status=production_verification_status,
        errors=snapshot.errors,
    )


# =============================================================================
# Stage 2/3/4 -- the smallest production-safe mechanism: reuse D-271's own
# bounded, real-subprocess ffmpeg decoder/encoder inspection. No provider
# call, no network, no GPU required (decoder/encoder LISTING is a pure CPU
# metadata query, never an actual decode/encode of real footage).
# =============================================================================

def capture_local_sandbox_capability_for_testing(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ProductionRuntimeCapability:
    """The HONEST label for every capability snapshot this gate's own
    offline tests produce: this sandbox's ffmpeg, not the real deployed
    worker image. Reuses `smp.capture_local_ffmpeg_capability` directly --
    no duplicated subprocess logic."""
    snapshot = smp.capture_local_ffmpeg_capability(runner)
    return _snapshot_to_capability(
        snapshot,
        capability_source=CAPABILITY_SOURCE_LOCAL_SANDBOX,
        production_verification_status=PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )


def capture_production_worker_capability(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ProductionRuntimeCapability:
    """Stage 2's own "worker-process startup self-check": the function a
    real deployed worker container's startup sequence (`rq_worker.py`'s
    `run_worker()`, identified in this gate as the exact existing seam --
    see docs/CUTSELL_DECISIONS.md D-274C) would call, once a future,
    separately-authorized activation gate wires it in. Uses the IDENTICAL
    mechanism as the sandbox path above (Stage 3: real ffmpeg decoder
    listing, never inferred from executable presence alone) -- the only
    difference is the label this result honestly carries. Calling this
    function from THIS gate's own tests (in this sandbox) still produces
    `production_verification_status=PRODUCTION_CAPABILITY_NOT_YET_
    ESTABLISHED`, because the caller -- not the function -- is the thing
    that must genuinely be the deployed production container for that
    status to become ESTABLISHED; this function cannot detect its own
    caller's identity, so it never claims ESTABLISHED on its own authority.
    A future D-274C-A activation gate, wiring this into `rq_worker.py`
    itself and confirming via a real deployed-container log, is the only
    legitimate path to `PRODUCTION_CAPABILITY_ESTABLISHED`.
    """
    snapshot = smp.capture_local_ffmpeg_capability(runner)
    return _snapshot_to_capability(
        snapshot,
        capability_source=CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )


# =============================================================================
# Stage 5/6 -- fail-closed bridge into D-272's existing RuntimeCapabilityInput
# =============================================================================

def bridge_to_runtime_capability_input(
    capability: ProductionRuntimeCapability,
) -> "sfp.RuntimeCapabilityInput":
    """Stage 6: the ONLY function that ever turns a `ProductionRuntime
    Capability` into D-272's own `RuntimeCapabilityInput`. Deliberately
    narrow: maps `hevc_decoder_available` -> `hevc_decode_confirmed` and
    nothing else -- `av1_decode_confirmed` is always left `False` here
    (out of this gate's own scope; never asserted even if this sandbox's
    ffmpeg happens to support AV1 decode, per Stage 6's own "do not
    duplicate/over-assert policy"). Fails closed: any non-ESTABLISHED
    `production_verification_status` still produces `hevc_decode_
    confirmed=False` -- Stage 1's own "capability must represent the
    ACTUAL worker runtime" is enforced HERE, not left to the caller."""
    confirmed = (
        capability.hevc_decoder_available
        and capability.production_verification_status == PRODUCTION_VERIFICATION_STATUS_ESTABLISHED
    )
    return sfp.RuntimeCapabilityInput(hevc_decode_confirmed=confirmed, av1_decode_confirmed=False)
