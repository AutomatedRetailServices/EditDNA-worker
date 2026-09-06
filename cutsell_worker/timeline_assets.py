"""Source-derived timeline assets for the lightweight mobile editor.

Filmstrip thumbnails and waveform peaks are presentation metadata only. They never
change source identity, clip boundaries, transcript, or AI decisions.
"""
from __future__ import annotations

from array import array
from pathlib import Path
import subprocess


def _run(command: list[str], *, capture: bool = False) -> bytes:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError("timeline_asset_ffmpeg_failed")
    return bytes(completed.stdout or b"")


def filmstrip_times(duration_sec: float, *, max_frames: int = 24) -> tuple[float, ...]:
    duration = float(duration_sec)
    if duration <= 0:
        raise ValueError("duration must be positive")
    count = max(2, min(int(max_frames), max(2, int(duration // 1.5) + 1)))
    if count == 2:
        return (0.0, max(0.0, duration - 0.001))
    step = duration / (count - 1)
    return tuple(min(duration - 0.001, index * step) for index in range(count))


def generate_filmstrip(
    source_path: str,
    output_dir: str,
    *,
    duration_sec: float,
    max_frames: int = 24,
    width: int = 160,
) -> tuple[dict, ...]:
    if width < 64 or width > 640:
        raise ValueError("filmstrip width must be between 64 and 640")
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    assets = []
    for index, timestamp in enumerate(filmstrip_times(duration_sec, max_frames=max_frames)):
        path = target / f"frame-{index:03d}.jpg"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}", "-i", str(source),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(path),
        ])
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError("filmstrip_frame_missing")
        assets.append({"time": round(timestamp, 3), "path": str(path)})
    return tuple(assets)


def waveform_peaks(source_path: str, *, buckets: int = 256, sample_rate: int = 8000) -> tuple[float, ...]:
    """Return normalized mono amplitude peaks for drawing a mobile timeline waveform."""
    if not 32 <= int(buckets) <= 2048:
        raise ValueError("waveform buckets must be between 32 and 2048")
    if sample_rate < 4000 or sample_rate > 48000:
        raise ValueError("unsupported waveform sample rate")
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source_path)
    pcm = _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ], capture=True)
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return tuple(0.0 for _ in range(int(buckets)))
    bucket_count = int(buckets)
    chunk = max(1, (len(samples) + bucket_count - 1) // bucket_count)
    peaks = []
    for index in range(bucket_count):
        segment = samples[index * chunk : min(len(samples), (index + 1) * chunk)]
        peak = max((abs(int(value)) for value in segment), default=0) / 32768.0
        peaks.append(round(min(1.0, peak), 4))
    return tuple(peaks)
