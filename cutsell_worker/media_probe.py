"""FFprobe-backed media metadata for the clean CutSell worker."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MediaProbe:
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _fps(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / max(float(den), 1.0)
    return float(value)


_PROBE_CACHE: dict = {}
_PROBE_CACHE_MAX = 256


def probe_media(path: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> MediaProbe:
    """ffprobe facts about one file. The same unchanged file is probed once per process (the
    export used to probe each source again for every segment and every QC attempt)."""
    key = None
    if runner is subprocess.run:
        try:
            stat = os.stat(path)
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if key is not None and key in _PROBE_CACHE:
            return _PROBE_CACHE[key]
    result = _probe_media_uncached(path, runner)
    if key is not None:
        if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
            _PROBE_CACHE.clear()
        _PROBE_CACHE[key] = result
    return result


def _probe_media_uncached(path: str, runner: Callable[..., subprocess.CompletedProcess]) -> MediaProbe:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate",
        "-of", "json", path,
    ]
    completed = runner(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    return MediaProbe(
        duration_sec=max(0.0, duration),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=max(0.0, _fps(video.get("avg_frame_rate"))),
        has_audio=has_audio,
    )
