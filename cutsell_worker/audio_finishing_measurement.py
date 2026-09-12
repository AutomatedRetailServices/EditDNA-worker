"""Audio Finishing -- MEASUREMENT ONLY foundation (D-247).

D-246 audited the Audio Finishing gap and found zero loudness/true-peak/
clipping observability anywhere in the codebase. This module is the first
implementation gate for that gap, and it is deliberately, strictly scoped
to MEASUREMENT:

    MEASUREMENT (this module)  !=  POLICY (a target LUFS/peak ceiling,
    an acceptable delta -- not this module)  !=  CORRECTION (a gain
    change, a limiter, a compressor, a denoiser -- not this module).

Every function here runs `ffmpeg`/`ffprobe` as a real subprocess against a
real media file (or a real caller-supplied time window of one) and reports
what it measured. Nothing here decides pass/fail, nothing here invents a
threshold, and nothing here writes so much as one modified audio sample
back to disk. `finishing_contract.py` (D-024) remains the correct future
home for the POLICY/CORRECTION layer (`FinishingSpec.target_loudness_lufs`,
`true_peak_ceiling_dbtp`) -- this module deliberately does not touch or
extend that Protocol; it exists one layer earlier, to give a future
Finishing implementation (or any other caller) real numbers to reason
about instead of no numbers at all.

## What IS real and built here

- **Loudness / true peak / loudness range**: ffmpeg's own `ebur128` filter
  (`peak=true`), parsed from its real stderr "Summary:" block -- the same
  BS.1770 measurement broadcast loudness tooling uses. Per-frame streaming
  lines are deliberately ignored; only the terminal Summary block (printed
  once, after the whole input has been measured) is parsed, located by
  finding the literal "Summary:" marker first so the per-frame lines'
  own "I:"/"LRA:" tokens are never mistaken for the summary values.
- **Sample peak**: ffmpeg's own `astats` filter's "Overall" section's
  "Peak level dB" -- a distinct quantity from `ebur128`'s BS.1770 true
  peak (true peak reconstructs the inter-sample analog peak via
  oversampling; sample peak only looks at the discrete samples actually
  stored). Both are reported, never conflated.
- **Clipping observability**: this ffmpeg build's `astats` filter (ffmpeg
  6.1.1, confirmed empirically in this sandbox via
  `ffmpeg -h filter=astats`) has NO dedicated "number of clipped samples"
  metric -- that field simply does not exist in this filter's real output
  here, so no code below pretends to parse one. What IS real and
  deterministic: `astats`'s "Peak level dB" is a literal, unrounded
  measurement of the loudest sample already present in the file. A
  digital signal cannot exceed 0 dBFS without visible information loss;
  "Peak level dB" landing at/above `_CLIPPING_PEAK_THRESHOLD_DB` (a small
  negative epsilon accounting for float rounding, not an invented
  loudness target) is a direct, physical observation that the signal is
  sitting at the digital ceiling. This is a conservative, honestly-named
  proxy (`CLIPPING_DETECTED`/`NO_CLIPPING_DETECTED`), not a certified
  "flat-topped waveform" detector -- a legitimate signal that peaks
  exactly at full scale without audible distortion is rare but possible,
  and this module says so rather than overclaiming precision no ffmpeg
  filter in this build actually provides.
- **Duration / sample rate / channel count / channel layout**: ffprobe,
  scoped to the audio stream. Duration reuses `media_probe.probe_media`
  (the existing, single project-native ffprobe-duration owner -- see
  `flow_b.py`/`validation.py`/`retry_scan.py`, all of which already call
  it) rather than re-deriving duration from a second ffprobe call;
  sample_rate/channels/channel_layout are audio-stream-specific fields
  `probe_media` does not expose (it is video-oriented: width/height/fps/
  has_audio), so this module adds one narrow, additional ffprobe call for
  exactly those three fields rather than duplicating what already exists.
- **Silence observation**: reuses `post_render_media_qc.check_accidental_silence`
  verbatim (imported, not reimplemented) so this module never carries a
  second silence-detection implementation.
- **Windowed measurement**: every measuring function accepts an optional
  `start_sec`/`end_sec` window (via ffmpeg `-ss`/`-t`), so a caller can
  measure one clip or one join region without measuring the whole file --
  the data-contract foundation a future gain-continuity/join-compatibility
  check would need, without this module trying to be that check itself.

## What is honestly NOT built here, and why

No target LUFS, no true-peak ceiling, no "acceptable" loudness range, no
PASS/FAIL verdict of any kind is computed anywhere in this module --
inventing one here would be exactly the POLICY layer this gate's own
directive excludes; that decision belongs to `finishing_contract.py`'s
dormant `FinishingSpec` once a Product Owner sets it. No gain is applied,
no `loudnorm` normalization pass is run (only measurement filters --
`ebur128`, `astats` -- are ever invoked; `loudnorm`'s own two-pass
analysis mode is not used here even in analysis-only form, to keep this
module's dependency surface to the two filters actually needed), no
limiter/compressor/denoise/hum/highpass/breath/mouth-click/plosive/
de-essing capability is added -- all remain exactly the gaps D-246 found,
now measurable but still uncorrected.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Sequence

from .media_probe import probe_media
from .post_render_media_qc import check_accidental_silence
from .post_render_watch_listen_qc import PostRenderQCResult

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"
_DEFAULT_TIMEOUT_SEC = 60.0

# A small negative epsilon, not an invented loudness target: float/dB
# rounding on a genuinely full-scale digital signal can land at e.g.
# -0.0003 dB rather than exactly 0.0 -- this only accounts for that,
# it does not define any notion of "acceptable" peak level.
_CLIPPING_PEAK_THRESHOLD_DB = -0.1

MEASUREMENT_STATUS_COMPLETE = "COMPLETE"
MEASUREMENT_STATUS_PARTIAL = "PARTIAL"
MEASUREMENT_STATUS_UNAVAILABLE = "UNAVAILABLE"
MEASUREMENT_STATUS_MEASUREMENT_ERROR = "MEASUREMENT_ERROR"

CLIPPING_STATUS_CLIPPING_DETECTED = "CLIPPING_DETECTED"
CLIPPING_STATUS_NO_CLIPPING_DETECTED = "NO_CLIPPING_DETECTED"
CLIPPING_STATUS_UNKNOWN = "UNKNOWN"


def _run(args: list[str], *, timeout_sec: float = _DEFAULT_TIMEOUT_SEC) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined stdout+stderr).
    Never raises on a nonzero exit -- callers decide what that means.
    Matches `post_render_media_qc._run`'s exact pattern verbatim."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout_sec, check=False,
    )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def _window_args(start_sec: float | None, end_sec: float | None) -> list[str]:
    args: list[str] = []
    if start_sec is not None:
        args += ["-ss", f"{max(0.0, start_sec):.6f}"]
    if start_sec is not None and end_sec is not None:
        duration = max(0.0, end_sec - start_sec)
        args += ["-t", f"{duration:.6f}"]
    return args


def _parse_db_or_inf(raw: str) -> float | None:
    raw = raw.strip()
    if raw in {"-inf", "-Infinity"}:
        return float("-inf")
    if raw in {"inf", "+inf", "Infinity"}:
        return float("inf")
    try:
        return float(raw)
    except ValueError:
        return None


_INTEGRATED_LOUDNESS_RE = re.compile(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+|-?inf)\s*LUFS")
_LOUDNESS_RANGE_RE = re.compile(r"Loudness range:\s*\n\s*LRA:\s*(-?[\d.]+)\s*LU")
_TRUE_PEAK_RE = re.compile(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-?inf)\s*dBFS")


def _parse_ebur128_summary(output: str) -> tuple[float | None, float | None, float | None, list[str]]:
    """Parse ffmpeg `ebur128=peak=true`'s real stderr. Only the terminal
    "Summary:" block is considered -- the per-frame streaming lines
    (printed continuously while ffmpeg processes the file) contain their
    own differently-scoped "I:"/"LRA:" tokens and must never be read as
    the final measurement. Returns
    (integrated_lufs, loudness_range_lu, true_peak_dbfs, errors)."""
    errors: list[str] = []
    summary_idx = output.find("Summary:")
    if summary_idx == -1:
        return None, None, None, ["ebur128 output did not contain a Summary block"]
    summary = output[summary_idx:]

    integrated_lufs: float | None = None
    match = _INTEGRATED_LOUDNESS_RE.search(summary)
    if match:
        integrated_lufs = _parse_db_or_inf(match.group(1))
        if integrated_lufs is None:
            errors.append("integrated loudness value unparseable")
    else:
        errors.append("integrated loudness not found in ebur128 Summary")

    lra: float | None = None
    match = _LOUDNESS_RANGE_RE.search(summary)
    if match:
        try:
            lra = float(match.group(1))
        except ValueError:
            errors.append("loudness range value unparseable")
    else:
        errors.append("loudness range not found in ebur128 Summary")

    true_peak: float | None = None
    match = _TRUE_PEAK_RE.search(summary)
    if match:
        true_peak = _parse_db_or_inf(match.group(1))
        if true_peak is None:
            errors.append("true peak value unparseable")
    else:
        errors.append("true peak not found in ebur128 Summary")

    return integrated_lufs, lra, true_peak, errors


def _measure_loudness(media_path: str, *, start_sec: float | None, end_sec: float | None) -> tuple[float | None, float | None, float | None, list[str]]:
    args = [_FFMPEG, "-hide_banner"] + _window_args(start_sec, end_sec) + [
        "-i", media_path, "-af", "ebur128=peak=true", "-f", "null", "-",
    ]
    returncode, output = _run(args)
    integrated, lra, true_peak, errors = _parse_ebur128_summary(output)
    if returncode != 0 and not errors:
        errors.append(f"ebur128 ffmpeg exited nonzero ({returncode})")
    return integrated, lra, true_peak, errors


_OVERALL_MARKER_RE = re.compile(r"\]\s*Overall\s*$", re.MULTILINE)
_PEAK_LEVEL_DB_RE = re.compile(r"Peak level dB:\s*(-?[\d.]+|-?inf)")


def _parse_astats_sample_peak(output: str) -> tuple[float | None, list[str]]:
    """Parse ffmpeg `astats`'s real stderr. Per-channel blocks are each
    followed (for a real multi-channel or mono file) by exactly one
    terminal "Overall" block -- only the "Peak level dB" line inside that
    LAST "Overall" section is the whole-file sample peak; the per-channel
    "Peak level dB" lines above it describe one channel only and must not
    be read as the file's overall sample peak."""
    matches = list(_OVERALL_MARKER_RE.finditer(output))
    if not matches:
        return None, ["astats output did not contain an Overall section"]
    overall_text = output[matches[-1].end():]
    match = _PEAK_LEVEL_DB_RE.search(overall_text)
    if not match:
        return None, ["Peak level dB not found in astats Overall section"]
    value = _parse_db_or_inf(match.group(1))
    if value is None:
        return None, ["astats Peak level dB value unparseable"]
    return value, []


def _measure_sample_peak(media_path: str, *, start_sec: float | None, end_sec: float | None) -> tuple[float | None, list[str]]:
    args = [_FFMPEG, "-hide_banner"] + _window_args(start_sec, end_sec) + [
        "-i", media_path, "-af", "astats", "-f", "null", "-",
    ]
    returncode, output = _run(args)
    peak, errors = _parse_astats_sample_peak(output)
    if returncode != 0 and not errors:
        errors.append(f"astats ffmpeg exited nonzero ({returncode})")
    return peak, errors


def _classify_clipping(sample_peak_dbfs: float | None) -> str:
    if sample_peak_dbfs is None:
        return CLIPPING_STATUS_UNKNOWN
    if sample_peak_dbfs >= _CLIPPING_PEAK_THRESHOLD_DB:
        return CLIPPING_STATUS_CLIPPING_DETECTED
    return CLIPPING_STATUS_NO_CLIPPING_DETECTED


def _probe_audio_stream_fields(media_path: str) -> tuple[int | None, int | None, str | None, list[str]]:
    """One narrow ffprobe call for the audio-stream fields
    `media_probe.probe_media` does not expose (it is video-oriented).
    Returns (sample_rate_hz, channel_count, channel_layout, errors)."""
    proc = subprocess.run(
        [
            _FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,channel_layout",
            "-of", "json", media_path,
        ],
        capture_output=True, text=True, timeout=_DEFAULT_TIMEOUT_SEC, check=False,
    )
    if proc.returncode != 0:
        return None, None, None, [f"ffprobe audio-stream query exited nonzero ({proc.returncode})"]
    import json
    try:
        payload = json.loads(proc.stdout or "{}")
    except ValueError:
        return None, None, None, ["ffprobe audio-stream JSON unparseable"]
    streams = payload.get("streams") or []
    if not streams:
        return None, None, None, ["no audio stream present"]
    stream = streams[0]
    sample_rate = None
    if stream.get("sample_rate") is not None:
        try:
            sample_rate = int(stream["sample_rate"])
        except (TypeError, ValueError):
            pass
    channels = stream.get("channels")
    channels = int(channels) if isinstance(channels, int) else None
    channel_layout = stream.get("channel_layout")
    channel_layout = str(channel_layout) if channel_layout else None
    return sample_rate, channels, channel_layout, []


@dataclass(frozen=True)
class AudioFinishingMeasurement:
    """MEASUREMENT ONLY. No field on this dataclass is a target, a
    threshold, or a verdict -- every value is either a real number this
    module measured from real decoded/probed media, or an explicit
    `None`/`UNKNOWN` when that measurement was not obtainable, never a
    fabricated default."""

    media_path: str
    window_start_sec: float | None
    window_end_sec: float | None

    duration_sec: float | None
    sample_rate_hz: int | None
    channel_count: int | None
    channel_layout: str | None

    integrated_loudness_lufs: float | None
    loudness_range_lu: float | None
    true_peak_dbfs: float | None
    sample_peak_dbfs: float | None
    clipping_status: str

    silence_result: PostRenderQCResult | None

    measurement_status: str
    measurement_errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


def measure_audio(
    media_path: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    include_silence: bool = True,
    protected_pause_windows: Sequence[tuple[float, float]] = (),
) -> AudioFinishingMeasurement:
    """Measure real loudness/peak/clipping/stream/silence facts about
    `media_path` (or, if `start_sec`/`end_sec` are given, one real time
    window of it). Applies zero correction. Computes zero policy verdict.

    `measurement_status`:
    - COMPLETE: every measurable field was obtained.
    - PARTIAL: at least one field was obtained but at least one failed.
    - UNAVAILABLE: no audio stream / no field could be measured at all.
    - MEASUREMENT_ERROR: an unexpected exception occurred while probing
      (never raised to the caller -- reported instead, matching this
      codebase's `_run`-never-raises convention).
    """
    errors: list[str] = []
    try:
        duration_sec: float | None
        if start_sec is not None or end_sec is not None:
            duration_sec = None
            if start_sec is not None and end_sec is not None:
                duration_sec = max(0.0, end_sec - start_sec)
        else:
            try:
                duration_sec = probe_media(media_path).duration_sec
            except Exception as exc:  # pragma: no cover - defensive, matches _run-never-raises convention
                duration_sec = None
                errors.append(f"probe_media duration failed: {exc}")

        sample_rate_hz, channel_count, channel_layout, stream_errors = _probe_audio_stream_fields(media_path)
        errors.extend(stream_errors)

        if channel_count == 0 or "no audio stream present" in stream_errors:
            silence_result = None
            integrated, lra, true_peak, sample_peak = None, None, None, None
            clipping_status = CLIPPING_STATUS_UNKNOWN
            return AudioFinishingMeasurement(
                media_path=media_path, window_start_sec=start_sec, window_end_sec=end_sec,
                duration_sec=duration_sec, sample_rate_hz=sample_rate_hz,
                channel_count=channel_count, channel_layout=channel_layout,
                integrated_loudness_lufs=integrated, loudness_range_lu=lra,
                true_peak_dbfs=true_peak, sample_peak_dbfs=sample_peak,
                clipping_status=clipping_status, silence_result=silence_result,
                measurement_status=MEASUREMENT_STATUS_UNAVAILABLE,
                measurement_errors=tuple(errors),
                provenance={"tool": "ffmpeg/ffprobe", "filters": []},
            )

        integrated, lra, true_peak, loudness_errors = _measure_loudness(
            media_path, start_sec=start_sec, end_sec=end_sec,
        )
        errors.extend(loudness_errors)

        sample_peak, peak_errors = _measure_sample_peak(
            media_path, start_sec=start_sec, end_sec=end_sec,
        )
        errors.extend(peak_errors)

        clipping_status = _classify_clipping(sample_peak)

        silence_result: PostRenderQCResult | None = None
        if include_silence:
            try:
                silence_result = check_accidental_silence(
                    media_path, protected_pause_windows=protected_pause_windows,
                )
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"silence measurement failed: {exc}")

        measured_fields = [
            duration_sec, sample_rate_hz, channel_count, integrated, lra, true_peak, sample_peak,
        ]
        obtained = sum(1 for v in measured_fields if v is not None)
        if obtained == len(measured_fields):
            status = MEASUREMENT_STATUS_COMPLETE
        elif obtained == 0:
            status = MEASUREMENT_STATUS_UNAVAILABLE
        else:
            status = MEASUREMENT_STATUS_PARTIAL

        return AudioFinishingMeasurement(
            media_path=media_path, window_start_sec=start_sec, window_end_sec=end_sec,
            duration_sec=duration_sec, sample_rate_hz=sample_rate_hz,
            channel_count=channel_count, channel_layout=channel_layout,
            integrated_loudness_lufs=integrated, loudness_range_lu=lra,
            true_peak_dbfs=true_peak, sample_peak_dbfs=sample_peak,
            clipping_status=clipping_status, silence_result=silence_result,
            measurement_status=status,
            measurement_errors=tuple(errors),
            provenance={"tool": "ffmpeg/ffprobe", "filters": ["ebur128", "astats"]},
        )
    except Exception as exc:
        return AudioFinishingMeasurement(
            media_path=media_path, window_start_sec=start_sec, window_end_sec=end_sec,
            duration_sec=None, sample_rate_hz=None, channel_count=None, channel_layout=None,
            integrated_loudness_lufs=None, loudness_range_lu=None,
            true_peak_dbfs=None, sample_peak_dbfs=None,
            clipping_status=CLIPPING_STATUS_UNKNOWN, silence_result=None,
            measurement_status=MEASUREMENT_STATUS_MEASUREMENT_ERROR,
            measurement_errors=(f"unexpected exception: {exc}",),
            provenance={"tool": "ffmpeg/ffprobe", "filters": []},
        )


def attach_audio_finishing_diagnostics(
    qc_result: PostRenderQCResult, media_path: str, **measure_kwargs,
) -> dict:
    """Additive integration point (D-247 Stage 10): package a real
    `AudioFinishingMeasurement` alongside an existing
    `PostRenderQCResult` as a plain diagnostics dict. Never reads or
    mutates `qc_result.status` -- this exists so a caller can attach real
    loudness/peak/clipping numbers to a render's diagnostics without this
    module (or any caller of it) changing the PASS/FAIL verdict
    `run_post_render_media_qc` already computed. Whether/when a future
    Finishing policy layer ever consults these numbers to fail a render
    is explicitly out of scope here."""
    measurement = measure_audio(media_path, **measure_kwargs)
    return {
        "post_render_qc_status": qc_result.status,
        "audio_finishing_measurement": measurement,
    }
