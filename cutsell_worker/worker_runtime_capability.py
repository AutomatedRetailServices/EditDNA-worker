"""D-274C-A: Live Runtime Capability Activation.

Establishes a REAL worker process's own HEVC/H264 capability truth,
exactly once per process lifetime, and exposes it to the live D-272
policy evaluation path (`worker_job.evaluate_source_format_gate`) --
without ever running a media job, without activating auto-normalization,
and without touching what happens AFTER a `NORMALIZE_REQUIRED` decision
(D-272A's own "stop before expensive editorial processing" behavior is
unchanged).

Startup seam correction (Stage 1). D-274C's own docstrings assumed
`rq_worker.py`'s `run_worker()` was "the" real worker startup seam. This
gate's own audit of `Dockerfile.cutsell.worker` (the actual, ffmpeg/
libav-installing CutSell worker image) found that assumption WRONG: its
own `CMD` is `bash -lc 'python -m rq.cli worker -u "$REDIS_URL"
--worker-ttl 14400 cutsell'` -- the raw RQ CLI, with no app-level Python
bootstrap hook at all. (`rq_worker.py` + `start_worker.sh` is a separate,
RunPod-oriented launcher wired via `bootstrap.sh`; `Dockerfile.worker`'s
own `entrypoint.sh` similarly execs the raw `rq worker` CLI directly.)
None of the three real launch paths offers a natural "call this before
the dequeue loop starts" hook.

The correct, launcher-independent seam is therefore MODULE IMPORT, not
any one entrypoint script: RQ's own worker resolves and imports a job's
module before executing it, and Python caches that import for the life
of the process -- so a lazily-computed, memoized capability check
(`functools.lru_cache(maxsize=1)`) attached to this module runs exactly
ONCE per worker process, on first access, regardless of which of the
three launch paths actually started it, and regardless of whether that
first access happens at literal container boot or at the first job's
own module resolution. Either way it satisfies Stage 1's "once per
worker process or equivalent bounded lifecycle" and Stage 5's "no
per-job repeated shell probe."
"""
from __future__ import annotations

from functools import lru_cache

from . import production_runtime_capability as prc
from . import source_format_policy as sfp

# Stage 15 -- bounded, secret-free startup diagnostics; never a full
# decoder/encoder dump, never a raw command line, never an env var.
_DIAGNOSTIC_FIELDS = (
    "ffmpeg_version",
    "hevc_decoder_available",
    "h264_encoder_available",
    "capability_source",
    "production_verification_status",
)


def _genuinely_succeeded(capability: "prc.ProductionRuntimeCapability") -> bool:
    """Stage 3's own fail-closed bar: a probe counts as a genuine success
    only when it captured a real ffmpeg version AND recorded zero
    subprocess/probe errors. Anything else (a missing ffmpeg binary, a
    subprocess timeout, an unexpected exception any layer below caught
    into `errors`) leaves the result at NOT_YET_ESTABLISHED -- never
    fabricated as ESTABLISHED from partial evidence."""
    return capability.ffmpeg_version is not None and len(capability.errors) == 0


@lru_cache(maxsize=1)
def get_worker_runtime_capability() -> "prc.ProductionRuntimeCapability":
    """Stage 2/3/4/5/9/14: the ONE place a real worker process establishes
    its own HEVC/H264 capability truth. Computed at most once per process
    (`lru_cache(maxsize=1)` on a zero-argument function -- Stage 4's own
    "module-level frozen snapshot" pattern, made lazy so importing this
    module never itself triggers a subprocess call as a side effect of
    mere import, e.g. during unrelated test collection). The returned
    object is itself an immutable, frozen dataclass -- Stage 14's own
    "read-only after establishment" is satisfied by construction, not by
    caller discipline.

    Fail-closed (Stage 3/9): if the underlying probe did not genuinely
    succeed (see `_genuinely_succeeded`), or if capture raises an
    unexpected exception this module did not anticipate, the returned
    capability's `production_verification_status` stays `PRODUCTION_
    CAPABILITY_NOT_YET_ESTABLISHED` -- NEVER promoted to `ESTABLISHED`
    from partial or absent evidence, and this function NEVER raises (a
    worker must be able to start and serve ordinary H264 SDR jobs even
    when HEVC capability establishment fails entirely -- Stage 9's own
    recommended Option A, "do not turn optional HEVC support into a
    global worker outage")."""
    try:
        captured = prc.capture_production_worker_capability()
    except Exception as exc:  # noqa: BLE001 -- must never crash worker startup
        return prc.ProductionRuntimeCapability(
            hevc_decoder_available=False,
            h264_encoder_available=False,
            ffmpeg_version=None,
            capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
            production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
            errors=(f"worker_capability_establishment_failed:{exc.__class__.__name__}",),
        )

    if not _genuinely_succeeded(captured):
        # Stays UNESTABLISHED -- the underlying capture already carries
        # its own `errors`/absent version; nothing fabricated here.
        return captured

    # Stage 2: the ONLY place `production_verification_status` may ever
    # become ESTABLISHED -- gated on a genuinely clean probe, in whatever
    # process this code actually runs (sandbox, CI, or the real deployed
    # worker container). This is intentionally a dynamic, self-reporting
    # check, not a hardcoded assumption: it reports exactly what THIS
    # process's own ffmpeg build can do, wherever that process is.
    return prc.ProductionRuntimeCapability(
        hevc_decoder_available=captured.hevc_decoder_available,
        h264_encoder_available=captured.h264_encoder_available,
        ffmpeg_version=captured.ffmpeg_version,
        capability_source=captured.capability_source,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
        errors=captured.errors,
    )


def get_worker_runtime_capability_input() -> "sfp.RuntimeCapabilityInput":
    """Stage 6: the one call site `worker_job.py` uses to feed the live
    D-272 policy evaluation -- always routes through `production_runtime_
    capability.bridge_to_runtime_capability_input`'s own fail-closed
    logic (Stage 7's H264-encoder requirement included); never
    re-implements the bridge here."""
    return prc.bridge_to_runtime_capability_input(get_worker_runtime_capability())


def describe_worker_capability_diagnostics() -> dict:
    """Stage 8/15: bounded, secret-free diagnostics for a startup log
    line. Exactly the five fields named in Stage 8 -- no decoder/encoder
    list dumps, no raw command lines, no environment values."""
    capability = get_worker_runtime_capability()
    return {field: getattr(capability, field) for field in _DIAGNOSTIC_FIELDS}


def _reset_for_testing() -> None:
    """Test-only: clears the process-local memoized capability so a test
    can force a fresh capture (e.g. to prove per-process isolation or to
    inject a different `runner`). Never called from production code."""
    get_worker_runtime_capability.cache_clear()
