"""Real ffmpeg/ffprobe-backed PostRenderWatchListenQC media checks -- D-028.

The canonical directive is explicit: "do not fake this with transcript-only
checks. Use actual rendered media signals." This module is that -- every
function here runs `ffmpeg`/`ffprobe` as a subprocess against a real media
file on disk and parses its actual output; nothing here reads a transcript
or a diagnostics dict as a substitute for looking at the decoded media.

## What IS real and built here

- **DECODE_EXPORT_INTEGRITY**: a full null-decode of the file
  (`ffmpeg -v error -xerror -i <file> -f null -`). A nonzero exit or any
  stderr output means the file does not decode cleanly end to end -- the
  most literal possible "does this exported file actually work" check.
- **LINGERING_ACCIDENTAL_SILENCE**: ffmpeg's own `silencedetect` audio
  filter, run against the real decoded audio. A silence interval longer
  than `max_allowed_silence_sec` that does not fall inside a caller-
  supplied `protected_pause_windows` range is flagged. Which windows are
  "protected" is an ordinary parameter, never a Video00-specific constant
  -- the caller (whatever upstream authority knows which pauses were
  editorially intentional) supplies them.
- **FROZEN_OR_REPEATED_FRAME**: ffmpeg's own `freezedetect` video filter --
  a real per-frame comparison against the actual decoded video, not a
  guess.
- **DEAD_BLACK_FRAME**: ffmpeg's own `blackdetect` video filter, same
  reasoning.
- **ABRUPT_AUDIO_DISCONTINUITY**: at each caller-supplied boundary
  timestamp (an edit point in the final rendered timeline), decode a short
  raw-PCM window centered on it via ffmpeg and compare the sample-to-sample
  amplitude jump at the boundary against the local signal's own RMS
  variation (numpy). A jump many times larger than the surrounding
  variation is the actual acoustic signature of a hard "click"/step
  discontinuity at a cut -- not inferred from text, measured from the
  waveform.

## What is honestly NOT built here, and why

`unsafe phoneme truncation`, `unnatural breath cut`: require ASR-phoneme
alignment against the rendered audio (word/phone-level timing), which is a
speech-recognition capability, not something ffmpeg/ffprobe alone can do.
`body/mic/camera reset debris`, `awkward post-line expression`, `obvious
face/body jump`: require computer-vision/pose/face estimation -- no
cv2/mediapipe/similar library is installed in this environment, and adding
one is a separate, larger decision than this cycle's scope. `framing
integrity` beyond a gross resolution/aspect-ratio mismatch (which IS
checkable via ffprobe stream metadata, but is not implemented here because
no fixture or real failure case motivates it yet -- adding an unmotivated
check risks a heuristic with no evidence behind it). Fine-grained `A/V sync
drift` (sub-frame): ffprobe can report each stream's start_time/duration,
enough to catch a gross whole-stream misalignment, but nothing finer without
audio/video cross-correlation, not implemented here for the same reason.
None of these are faked with a heuristic standing in for the real signal --
they are left as an honest, stated gap, exactly like D-024/D-025's
STORY_ORDER_BREAK-before-CAUSAL_ORDER_BREAK gap was until it was actually
built.

## Authority rule and the bounded physical repair loop

Every finding kind this module can emit is a physical/timing kind (see
`post_render_watch_listen_qc.is_physical_finding_kind` -- this module's own
tests assert every finding it ever produces satisfies that). Per the
canonical directive, PostRenderWatchListenQC may NEVER change semantic
membership: `run_bounded_physical_repair_loop` below only ever asks its
caller-supplied `render_attempt` function to re-render (a BoundaryEngine/
Renderer concern), never touches `CanonicalEditPlan` or `draft.selected`,
and refuses (raises) if ever handed a non-physical finding kind to "repair"
-- that would be a semantic mismatch, which this module has no authority to
touch and must instead route upstream (the caller's responsibility, not
this loop's). Bounded at `max_attempts`; exhausting them without a clean
pass reports `NEEDS_HUMAN_REVIEW`, never `PASS`.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .post_render_watch_listen_qc import (
    ABRUPT_AUDIO_DISCONTINUITY,
    DEAD_BLACK_FRAME,
    DECODE_EXPORT_INTEGRITY,
    FROZEN_OR_REPEATED_FRAME,
    LINGERING_ACCIDENTAL_SILENCE,
    PostRenderFinding,
    PostRenderQCResult,
    is_physical_finding_kind,
)

_FFMPEG = "ffmpeg"
_DEFAULT_TIMEOUT_SEC = 60.0


def _run(args: list[str], *, timeout_sec: float = _DEFAULT_TIMEOUT_SEC) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined stdout+stderr).
    Never raises on a nonzero exit -- callers decide what that means."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout_sec, check=False,
    )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


def probe_decode_integrity(media_path: str) -> PostRenderQCResult:
    """Real, literal "does this file actually decode end to end" check: a
    full null-decode of both streams. Any nonzero exit or stderr output at
    `-v error` means real corruption/truncation/desync, not a guess."""
    returncode, output = _run([
        _FFMPEG, "-hide_banner", "-v", "error", "-xerror",
        "-i", media_path, "-f", "null", "-",
    ])
    if returncode == 0 and not output.strip():
        return PostRenderQCResult(status="PASS", findings=())
    finding = PostRenderFinding(
        kind=DECODE_EXPORT_INTEGRITY,
        start=0.0, end=0.0,
        detail={"returncode": returncode, "ffmpeg_error": output.strip()[:2000]},
        routes_to="BoundaryEngine",
    )
    return PostRenderQCResult(status="FAIL", findings=(finding,))


_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def _detect_silence_intervals(
    media_path: str, *, noise_floor_db: float = -35.0, min_silence_sec: float = 0.3,
) -> list[tuple[float, float]]:
    _, output = _run([
        _FFMPEG, "-hide_banner", "-i", media_path,
        "-af", f"silencedetect=noise={noise_floor_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ])
    starts = [float(m) for m in _SILENCE_START_RE.findall(output)]
    ends = [float(m) for m in _SILENCE_END_RE.findall(output)]
    return list(zip(starts, ends))


def check_accidental_silence(
    media_path: str,
    *,
    protected_pause_windows: Sequence[tuple[float, float]] = (),
    max_allowed_silence_sec: float = 1.2,
    noise_floor_db: float = -35.0,
) -> PostRenderQCResult:
    """Flag a real, decoded silence interval longer than
    `max_allowed_silence_sec` that does not fall inside any caller-supplied
    protected/expected pause window. `protected_pause_windows` is an
    ordinary parameter (e.g. an editorially intentional dramatic pause the
    upstream draft already knows about) -- never a hardcoded constant."""
    findings: list[PostRenderFinding] = []
    for start, end in _detect_silence_intervals(
        media_path, noise_floor_db=noise_floor_db, min_silence_sec=max_allowed_silence_sec,
    ):
        duration = end - start
        if duration < max_allowed_silence_sec:
            continue
        protected = any(
            window_start <= start and end <= window_end
            for window_start, window_end in protected_pause_windows
        )
        if protected:
            continue
        findings.append(PostRenderFinding(
            kind=LINGERING_ACCIDENTAL_SILENCE,
            start=start, end=end,
            detail={"duration_sec": duration, "noise_floor_db": noise_floor_db},
            routes_to="BoundaryEngine",
        ))
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))


_FREEZE_START_RE = re.compile(r"freeze_start:\s*(-?[\d.]+)")
_FREEZE_END_RE = re.compile(r"freeze_end:\s*(-?[\d.]+)")


def check_frozen_frames(
    media_path: str, *, min_freeze_sec: float = 0.5, noise_threshold_db: float = -60.0,
) -> PostRenderQCResult:
    """ffmpeg's own `freezedetect` filter -- a real frame-to-frame video
    comparison, not a guess -- flags a stretch of visually-identical/
    near-identical frames (dead air, a stuck frame, an accidental repeat)."""
    _, output = _run([
        _FFMPEG, "-hide_banner", "-i", media_path,
        "-vf", f"freezedetect=n={noise_threshold_db}dB:d={min_freeze_sec}",
        "-f", "null", "-",
    ])
    starts = [float(m) for m in _FREEZE_START_RE.findall(output)]
    ends = [float(m) for m in _FREEZE_END_RE.findall(output)]
    findings = [
        PostRenderFinding(
            kind=FROZEN_OR_REPEATED_FRAME, start=start, end=end,
            detail={"duration_sec": end - start},
            routes_to="BoundaryEngine",
        )
        for start, end in zip(starts, ends)
    ]
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))


_BLACK_START_RE = re.compile(r"black_start:\s*(-?[\d.]+)")
_BLACK_END_RE = re.compile(r"black_end:\s*(-?[\d.]+)")


def check_dead_black_frames(
    media_path: str, *, min_black_sec: float = 0.3, picture_black_threshold: float = 0.98,
) -> PostRenderQCResult:
    """ffmpeg's own `blackdetect` filter -- flags a real stretch of
    (near-)black frames, e.g. a dropped/failed segment in the export."""
    _, output = _run([
        _FFMPEG, "-hide_banner", "-i", media_path,
        "-vf", f"blackdetect=d={min_black_sec}:pic_th={picture_black_threshold}",
        "-f", "null", "-",
    ])
    starts = [float(m) for m in _BLACK_START_RE.findall(output)]
    ends = [float(m) for m in _BLACK_END_RE.findall(output)]
    findings = [
        PostRenderFinding(
            kind=DEAD_BLACK_FRAME, start=start, end=end,
            detail={"duration_sec": end - start},
            routes_to="BoundaryEngine",
        )
        for start, end in zip(starts, ends)
    ]
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))


def _extract_pcm_window(
    media_path: str, *, center_sec: float, window_sec: float, sample_rate: int = 22_050,
):
    """Decode a short mono PCM window centered on `center_sec` via ffmpeg,
    returning it as an int16 numpy array. Real decoded samples, not a proxy.

    numpy is imported lazily here (matches the existing lazy-import pattern
    in human_gold_decision_map.py/human_gold_decision_map_v2.py) so this
    module carries no hard numpy import-time dependency for callers that
    never touch the audio-discontinuity check."""
    import numpy as np

    start = max(0.0, center_sec - window_sec)
    duration = window_sec * 2
    proc = subprocess.run(
        [
            _FFMPEG, "-hide_banner", "-v", "error",
            "-ss", f"{start:.6f}", "-i", media_path, "-t", f"{duration:.6f}",
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=_DEFAULT_TIMEOUT_SEC, check=False,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16)


def check_audio_discontinuity_at_boundaries(
    media_path: str,
    boundary_timestamps: Iterable[float],
    *,
    window_sec: float = 0.08,
    sample_rate: int = 22_050,
    jump_ratio_threshold: float = 6.0,
) -> PostRenderQCResult:
    """At each caller-supplied edit-point timestamp, decode the real audio
    immediately around it and compare the single largest sample-to-sample
    jump against the local signal's typical (median) jump -- the actual
    acoustic signature of a hard "click"/step discontinuity a bad cut
    leaves, not inferred from text. A boundary with too few real samples to
    judge (e.g. right at the very start of the file) is skipped, not
    guessed at."""
    import numpy as np

    findings: list[PostRenderFinding] = []
    for timestamp in boundary_timestamps:
        pcm = _extract_pcm_window(media_path, center_sec=timestamp, window_sec=window_sec, sample_rate=sample_rate)
        if pcm.size < 8:
            continue
        deltas = np.abs(np.diff(pcm.astype(np.int64)))
        if deltas.size < 4:
            continue
        peak = float(np.max(deltas))
        typical = float(np.median(deltas)) + 1.0  # +1 avoids a divide-by-zero on true silence
        if peak / typical >= jump_ratio_threshold and peak > 500:
            findings.append(PostRenderFinding(
                kind=ABRUPT_AUDIO_DISCONTINUITY,
                start=timestamp, end=timestamp,
                detail={"peak_sample_jump": peak, "typical_sample_jump": typical, "ratio": peak / typical},
                routes_to="BoundaryEngine",
            ))
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))


def run_post_render_media_qc(
    media_path: str,
    *,
    boundary_timestamps: Sequence[float] = (),
    protected_pause_windows: Sequence[tuple[float, float]] = (),
    max_allowed_silence_sec: float = 1.2,
    min_freeze_sec: float = 0.5,
    min_black_sec: float = 0.3,
) -> PostRenderQCResult:
    """Run every real media check against one rendered file and merge the
    findings. Decode integrity is checked first and short-circuits the rest
    (a file that does not decode cannot be meaningfully probed further by
    the other filters, which would themselves just fail confusingly)."""
    integrity = probe_decode_integrity(media_path)
    if integrity.status == "FAIL":
        return integrity

    results = [
        check_accidental_silence(
            media_path, protected_pause_windows=protected_pause_windows,
            max_allowed_silence_sec=max_allowed_silence_sec,
        ),
        check_frozen_frames(media_path, min_freeze_sec=min_freeze_sec),
        check_dead_black_frames(media_path, min_black_sec=min_black_sec),
    ]
    if boundary_timestamps:
        results.append(check_audio_discontinuity_at_boundaries(media_path, boundary_timestamps))

    findings = tuple(f for result in results for f in result.findings)
    assert all(is_physical_finding_kind(f.kind) for f in findings), (
        "run_post_render_media_qc must never emit a non-physical finding kind"
    )
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=findings)


@dataclass(frozen=True)
class PhysicalRepairAttempt:
    attempt_number: int
    finding_kinds: tuple[str, ...]
    repaired: bool


@dataclass(frozen=True)
class BoundedPhysicalRepairResult:
    status: str  # "PASS" | "NEEDS_HUMAN_REVIEW"
    final_result: PostRenderQCResult
    attempts: tuple[PhysicalRepairAttempt, ...]


DEFAULT_MAX_PHYSICAL_REPAIR_ATTEMPTS = 3


def run_bounded_physical_repair_loop(
    render_attempt: Callable[[int], str],
    *,
    max_attempts: int = DEFAULT_MAX_PHYSICAL_REPAIR_ATTEMPTS,
    qc_check: Callable[[str], PostRenderQCResult] = run_post_render_media_qc,
) -> BoundedPhysicalRepairResult:
    """Call `render_attempt(attempt_index)` (BoundaryEngine/Renderer's own
    concern -- this loop never edits `CanonicalEditPlan` or `draft.selected`
    itself) to obtain a rendered media path, QC it, and retry up to
    `max_attempts` if it still fails. Never claims PASS while any finding
    remains; exhausting attempts reports NEEDS_HUMAN_REVIEW.

    Raises `ValueError` immediately (does not silently swallow it) if
    `qc_check` ever returns a non-physical finding kind -- that would be a
    semantic mismatch this loop has no authority to "repair" by
    re-rendering; the canonical directive requires that to invalidate the
    candidate and route upstream instead, never be retried here."""
    attempts: list[PhysicalRepairAttempt] = []
    result = PostRenderQCResult(status="FAIL", findings=())
    for attempt_index in range(max_attempts):
        media_path = render_attempt(attempt_index)
        result = qc_check(media_path)
        for finding in result.findings:
            if not is_physical_finding_kind(finding.kind):
                raise ValueError(
                    f"run_bounded_physical_repair_loop received a non-physical finding "
                    f"kind ({finding.kind!r}) -- this is a semantic mismatch that must "
                    f"route upstream, never be retried as a physical repair"
                )
        attempts.append(PhysicalRepairAttempt(
            attempt_number=attempt_index + 1,
            finding_kinds=tuple(f.kind for f in result.findings),
            repaired=(result.status == "PASS"),
        ))
        if result.status == "PASS":
            return BoundedPhysicalRepairResult(status="PASS", final_result=result, attempts=tuple(attempts))
    return BoundedPhysicalRepairResult(status="NEEDS_HUMAN_REVIEW", final_result=result, attempts=tuple(attempts))
