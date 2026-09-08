"""D-155 Phase A -- parallel perception orchestration.

Per docs/CUTSELL_DECISIONS.md D-154/D-155. `flow_b.py`'s real, verified
orchestration (confirmed by direct code read, D-154) runs every
perception stage strictly serially even though Track A (ASR), the
real-signal half of Track B (`audio_silence.py`), and Track C
(`local_performance.py`) have almost no genuine cross-track dependency
-- each is a pure function of `local_paths` alone. This module makes
that ALREADY-INDEPENDENT work concurrent, changing nothing about WHAT is
computed, only WHEN.

## Dependency audit (restated from D-154, the basis for this design)

- `media_probe.probe_media` -> `asr_provider.transcribe`: real,
  necessary dependency (the `source.has_audio` gate below decides
  whether to call ASR at all, and only `probe_media`'s own measurement
  is trustworthy for that -- the request's own declared `has_audio`
  default is not). `flow_b.py` therefore still runs `probe_media` for
  every source SYNCHRONOUSLY, BEFORE this module's parallel batch, and
  passes it `hydrated_sources` (post-probe). This is a genuine
  HARD_DEPENDENCY, not appearance-parallelism avoided for its own sake.
- `asr_provider.transcribe` -> `audio_silence_events`: NO_DEPENDENCY
  (`audio_silence.py` reads the waveform directly via ffmpeg, never the
  transcript).
- `asr_provider.transcribe` -> `analyze_local_performance`: NO_
  DEPENDENCY (OpenCV/MediaPipe reads video frames only).
- `audio_silence_events` <-> `analyze_local_performance`: NO_DEPENDENCY
  on each other (disjoint inputs/outputs).

These three (ASR, audio_silence, local_performance) are therefore run
CONCURRENTLY here via a bounded `ThreadPoolExecutor` -- each is I/O/
subprocess/native-extension bound (ffmpeg subprocess, OpenCV/MediaPipe
C++ extensions, faster-whisper's own C++ decode), so GIL contention is
not a blocking concern for the kind of overlap this task measures.

## Concurrency safety

Every task below is a PURE function of already-immutable inputs
(`local_paths` -- a plain read-only mapping -- and, for ASR, the
existing `asr_provider`/`request.language_hint`) returning an immutable
result (tuples/frozen dataclasses). No task mutates shared state, writes
to a shared temp path, or shares a provider/client object across
threads (`FasterWhisperASR.transcribe` already constructs its own fresh
`WhisperModel` per call, per `asr.py`; `audio_silence.py`/
`local_performance.py` allocate no temp files at all, confirmed by
direct code read). Results are read back by TRACK NAME (never by
completion order), so the returned `ParallelPerceptionOutcome` is
byte-identical regardless of which task happens to finish first.

## Failure semantics (preserves TODAY's real behavior, invents nothing)

- **ASR (Track A): MANDATORY, hard-fail.** `asr_provider.transcribe`
  carries NO existing fail-open wrapper in `flow_b.py` today (confirmed
  by direct code read: the current per-source loop calls it bare, no
  try/except) -- an ASR failure aborts `process_local_sources` today,
  and this module preserves that exactly: the ASR future's exception is
  re-raised unmodified by `.result()`, never caught here.
- **Audio (Track B) / Visual (Track C): already fail-open in their own
  implementations** (`detect_audio_silence_intervals` never raises --
  its own docstring says so; `analyze_local_performance` returns a
  `ProviderStatus`-carrying result even on failure). This module ALSO
  wraps their retrieval in a defensive try/except (belt-and-braces,
  since a future change to either function should never be able to
  crash the whole batch silently) and reports `TRACK_STATUS_FAILED` if
  one somehow does raise -- without touching the other two tracks'
  results.
- Media (Track D) is NOT part of this module's parallel batch at all
  (see the HARD_DEPENDENCY note above) -- its own failure semantics
  (`probe_media` raises via `subprocess.run(..., check=True)`) are
  unchanged, still handled by `flow_b.py` itself, before this module
  ever runs.

`CUTSELL_PARALLEL_PERCEPTION_ENABLED` (default ON): the rollback flag.
Set to `0`/`false` to force the exact pre-D-155 SEQUENTIAL call order
(ASR, then audio_silence, then local_performance) -- byte-identical
values either way; only the scheduling changes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import time
from typing import Mapping, Tuple

from .asr import ASRProvider
from .audio_silence import audio_silence_events
from .contracts import TranscriptSegment
from .local_performance import LocalPerformanceResult, analyze_local_performance
from .raw_understanding_map import (
    TRACK_STATUS_FAILED,
    TRACK_STATUS_PASS,
    TRACK_STATUS_PARTIAL,
)
from .whole_video_analysis import TemporalEvent

_PARALLEL_PERCEPTION_ENV = "CUTSELL_PARALLEL_PERCEPTION_ENABLED"

TRACK_SPEECH = "speech"
TRACK_AUDIO = "audio"
TRACK_VISUAL = "visual"


def _env_true_default_true(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def parallel_perception_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_true(values.get(_PARALLEL_PERCEPTION_ENV))


@dataclass(frozen=True)
class TrackTiming:
    name: str
    status: str
    wall_time_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ParallelPerceptionOutcome:
    transcripts: Tuple[TranscriptSegment, ...]
    local_performance: LocalPerformanceResult
    audio_silence_by_source: Mapping[str, Tuple[TemporalEvent, ...]]
    speech_track_status: str
    audio_track_status: str
    visual_track_status: str
    parallel_enabled: bool
    track_timings: Tuple[TrackTiming, ...] = field(default_factory=tuple)
    wall_time_ms: float = 0.0
    sum_track_time_ms: float = 0.0

    def diagnostics(self) -> dict:
        """Bounded, tail-safe diagnostics -- no transcript/event payload."""
        return {
            "parallel_perception_enabled": self.parallel_enabled,
            "track_count": len(self.track_timings),
            "tracks_started": len(self.track_timings),
            "tracks_completed": sum(1 for t in self.track_timings if t.status != TRACK_STATUS_FAILED),
            "tracks_failed": sum(1 for t in self.track_timings if t.status == TRACK_STATUS_FAILED),
            "parallel_wall_time_ms": round(self.wall_time_ms, 3),
            "sum_track_time_ms": round(self.sum_track_time_ms, 3),
            "speech_track_status": self.speech_track_status,
            "audio_track_status": self.audio_track_status,
            "visual_track_status": self.visual_track_status,
        }


def _run_asr(asr_provider: ASRProvider, hydrated_sources, local_paths: Mapping[str, str], language_hint: str | None) -> Tuple[TranscriptSegment, ...]:
    transcripts = []
    for source in sorted(hydrated_sources, key=lambda item: item.source_order):
        if source.has_audio:
            transcripts.extend(asr_provider.transcribe(
                local_paths[source.source_asset_id],
                source_asset_id=source.source_asset_id,
                language_hint=language_hint,
            ))
    return tuple(transcripts)


def _timed(name: str, mandatory: bool, fn):
    """Wraps one track callable so it measures ITS OWN execution duration
    (start-to-finish inside the thread that actually runs it) -- never the
    orchestrator's own wait-for-.result() time, which would silently
    undercount a track that happened to be read back after a slower one
    already finished. Returns (value, TrackTiming); the mandatory track's
    exception is re-raised unmodified (see module docstring), never
    downgraded to a TrackTiming."""
    start = time.monotonic()
    if mandatory:
        value = fn()
        return value, TrackTiming(name, TRACK_STATUS_PASS, (time.monotonic() - start) * 1000.0)
    try:
        value = fn()
        return value, TrackTiming(name, TRACK_STATUS_PASS, (time.monotonic() - start) * 1000.0)
    except Exception as exc:  # noqa: BLE001 -- defensive; see module docstring
        return None, TrackTiming(name, TRACK_STATUS_FAILED, (time.monotonic() - start) * 1000.0, str(exc))


def run_parallel_perception(
    hydrated_sources,
    local_paths: Mapping[str, str],
    asr_provider: ASRProvider,
    language_hint: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> ParallelPerceptionOutcome:
    """Runs Track A (ASR), the real-signal half of Track B (audio_silence),
    and Track C (local_performance) -- concurrently when enabled, in the
    exact original sequential order when disabled (`CUTSELL_PARALLEL_
    PERCEPTION_ENABLED=0`). Returns byte-identical VALUES either way; only
    scheduling changes. `hydrated_sources` must already carry `probe_
    media`'s own `has_audio`/`duration_sec` (Track D's genuine, unchanged
    hard dependency -- see module docstring)."""
    enabled = parallel_perception_enabled(env)
    hydrated_sources = tuple(hydrated_sources)

    def _asr_task():
        return _run_asr(asr_provider, hydrated_sources, local_paths, language_hint)

    def _audio_task():
        return audio_silence_events(local_paths)

    def _visual_task():
        return analyze_local_performance(local_paths, target_fps=12.0)

    mandatory = {TRACK_SPEECH: True, TRACK_AUDIO: False, TRACK_VISUAL: False}
    tasks = {TRACK_SPEECH: _asr_task, TRACK_AUDIO: _audio_task, TRACK_VISUAL: _visual_task}
    results: dict[str, object] = {}
    timings: dict[str, TrackTiming] = {}
    wall_start = time.monotonic()

    if enabled:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Each future runs `_timed(...)`, so timing is measured INSIDE
            # the worker thread from the moment it actually starts, not
            # from when this loop gets around to collecting it.
            futures = {
                name: executor.submit(_timed, name, mandatory[name], fn)
                for name, fn in tasks.items()
            }
            # Deterministic read-back order (by NAME, never completion
            # order) -- see module docstring's concurrency-safety note. The
            # mandatory track's own exception (raised inside `_timed`,
            # since `mandatory=True` never catches) propagates unmodified
            # from `.result()` here.
            for name in (TRACK_SPEECH, TRACK_AUDIO, TRACK_VISUAL):
                results[name], timings[name] = futures[name].result()
    else:
        for name in (TRACK_SPEECH, TRACK_AUDIO, TRACK_VISUAL):
            results[name], timings[name] = _timed(name, mandatory[name], tasks[name])

    wall_time_ms = (time.monotonic() - wall_start) * 1000.0
    sum_track_time_ms = sum(t.wall_time_ms for t in timings.values())

    local_perf_result = results[TRACK_VISUAL]
    visual_status = timings[TRACK_VISUAL].status
    if visual_status != TRACK_STATUS_FAILED and local_perf_result is not None:
        visual_status = TRACK_STATUS_PASS if local_perf_result.status.available else TRACK_STATUS_PARTIAL
    if local_perf_result is None:
        from .local_performance import LocalPerformanceResult as _LPR
        from .providers import ProviderStatus as _PS
        local_perf_result = _LPR((), _PS("local_performance", False, False, "failed", timings[TRACK_VISUAL].error))

    audio_result = results[TRACK_AUDIO] or {}
    audio_status = timings[TRACK_AUDIO].status
    if audio_status != TRACK_STATUS_FAILED:
        audio_status = TRACK_STATUS_PASS

    return ParallelPerceptionOutcome(
        transcripts=results[TRACK_SPEECH],
        local_performance=local_perf_result,
        audio_silence_by_source=audio_result,
        speech_track_status=timings[TRACK_SPEECH].status,
        audio_track_status=audio_status,
        visual_track_status=visual_status,
        parallel_enabled=enabled,
        track_timings=tuple(timings[name] for name in (TRACK_SPEECH, TRACK_AUDIO, TRACK_VISUAL)),
        wall_time_ms=wall_time_ms,
        sum_track_time_ms=sum_track_time_ms,
    )
