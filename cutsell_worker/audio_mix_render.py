"""Editor v5 final audio mix: voice-over, music and sound effects over the finished video.

Runs as ONE ffmpeg pass on the composed file (after speed and overlays), so the technical QC and
the delivery review hear the real final sound:

    base audio [-> cleanup] -> x video_volume [-> ducking envelope]
    each clip: atrim(source_in..source_out) [-> cleanup] -> volume -> adelay(start)
    amix(normalize=0, duration=first) -> alimiter          video stream copied (-c:v copy)

`duration=first` cuts every clip at the end of the video. A mix with no clips, video_volume 1 and
no cleanup is a no-op: `render_preview` does not run this pass at all for it.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Voice cleanup: broadband denoise, rumble cut, loudness to about -16 LUFS, back to 48 kHz.
CLEANUP_FILTERS: tuple[str, ...] = (
    "afftdn",
    "highpass=f=80",
    "loudnorm=I=-16:TP=-1.5:LRA=11",
    "aresample=48000",
)
#: Video sound level under a ducking clip ("Lower video sound", -12 dB) and the smooth ramp into /
#: out of it.
DUCK_DB = -12.0
DUCK_GAIN = round(10 ** (DUCK_DB / 20.0), 3)  # 0.251
DUCK_FADE_SEC = 0.12
LIMITER = "alimiter=limit=0.95:level=disabled:latency=1"
_AUDIO_FORMAT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


@dataclass(frozen=True)
class LocalAudioClip:
    """One audio-lane clip resolved to a local file. `start` is in final-timeline seconds."""
    path: str
    start: float
    source_in: float
    source_out: float
    volume: float = 1.0
    duck_video: bool = False
    clean: bool = False

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.source_out) - float(self.source_in))


@dataclass(frozen=True)
class AudioMix:
    video_volume: float = 1.0
    video_clean: bool = False
    clips: tuple[LocalAudioClip, ...] = ()

    @property
    def is_noop(self) -> bool:
        return not self.clips and not self.video_clean and float(self.video_volume) == 1.0


def _duck_intervals(clips: tuple[LocalAudioClip, ...]) -> list[tuple[float, float]]:
    spans = sorted((float(c.start), float(c.start) + c.duration_sec) for c in clips if c.duck_video and c.duration_sec > 0)
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 2 * DUCK_FADE_SEC:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duck_volume_filter(clips: tuple[LocalAudioClip, ...]) -> str | None:
    """A volume envelope on the video's own sound: DUCK_GAIN while a ducking clip plays, with a
    DUCK_FADE_SEC ramp on each side. None when no clip ducks."""
    intervals = _duck_intervals(clips)
    if not intervals:
        return None
    fade = DUCK_FADE_SEC
    terms = [
        f"clip(min((t-{start - fade:.3f})/{fade:.3f},({end + fade:.3f}-t)/{fade:.3f}),0,1)"
        for start, end in intervals
    ]
    envelope = f"min(1,{'+'.join(terms)})"
    return f"volume='1-{1.0 - DUCK_GAIN:.3f}*{envelope}':eval=frame"


def build_audio_mix_command(composed_path: str, output_path: str, mix: AudioMix, *,
                            duration_sec: float | None = None) -> list[str]:
    """The single ffmpeg pass described in the module docstring. `duration_sec` (the composed
    video's length) pads/trims the mixed audio to exactly that length, so cleanup filters and the
    mixer's own buffering can never make the sound longer or shorter than the picture."""
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(composed_path)]
    for clip in mix.clips:
        command += ["-i", str(clip.path)]

    base_chain = ["[0:a]" + _AUDIO_FORMAT]
    if mix.video_clean:
        base_chain += [*CLEANUP_FILTERS, _AUDIO_FORMAT]
    base_chain.append(f"volume={float(mix.video_volume):.3f}")
    duck = duck_volume_filter(mix.clips)
    if duck:
        base_chain.append(duck)
    filters = [",".join(base_chain) + "[mixbase]"]

    labels = ["[mixbase]"]
    for index, clip in enumerate(mix.clips, start=1):
        if clip.duration_sec <= 0:
            continue
        delay_ms = max(0, round(float(clip.start) * 1000.0))
        chain = [
            f"[{index}:a]atrim=start={float(clip.source_in):.3f}:end={float(clip.source_out):.3f}",
            "asetpts=PTS-STARTPTS",
            _AUDIO_FORMAT,
        ]
        if clip.clean:
            chain += [*CLEANUP_FILTERS, _AUDIO_FORMAT]
        chain += [f"volume={float(clip.volume):.3f}", f"adelay={delay_ms}|{delay_ms}"]
        filters.append(",".join(chain) + f"[mixclip{index}]")
        labels.append(f"[mixclip{index}]")

    finish = LIMITER
    if duration_sec is not None and float(duration_sec) > 0:
        finish += f",apad=whole_dur={float(duration_sec):.6f},atrim=duration={float(duration_sec):.6f}"
    if len(labels) > 1:
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0,"
                       f"{finish}[mixout]")
    else:
        filters.append(f"[mixbase]{finish}[mixout]")

    command += [
        "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[mixout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return command
