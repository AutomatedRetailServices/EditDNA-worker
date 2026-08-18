"""Build a source-safe render plan from an editable CutSell draft."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Tuple

from .contracts import DraftTimeline


@dataclass(frozen=True)
class RenderSegment:
    clip_id: str
    source_asset_id: str
    source_path: str
    start: float
    end: float
    audio_muted: bool = False
    audio_volume: float = 1.0
    caption_text: str = ""
    caption_preset: str = "classic"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)


def _can_coalesce(left: RenderSegment, right: RenderSegment, *, tolerance_sec: float = 0.05) -> bool:
    """Return True when a hard cut would be visually/media-equivalent to continuity.

    Two selected clips that touch in the same source should not be rendered as two
    separate files and concatenated again. That creates a redundant decoder/encoder
    boundary at the exact same source frame and can show up as a visible jump even
    though the creator never stopped. Keep separate segments only when playback or
    caption settings materially differ.
    """
    if left.source_asset_id != right.source_asset_id or left.source_path != right.source_path:
        return False
    if abs(float(right.start) - float(left.end)) > tolerance_sec:
        return False
    if left.audio_muted != right.audio_muted or abs(left.audio_volume - right.audio_volume) > 1e-6:
        return False
    if left.caption_preset != right.caption_preset:
        return False
    # Different active caption payloads still need independent timing in the current
    # render contract. Empty captions are safe to coalesce and are the common Clean Cut
    # preview path.
    if left.caption_text != right.caption_text and (left.caption_text or right.caption_text):
        return False
    return True


def _coalesce_contiguous_segments(segments: Tuple[RenderSegment, ...]) -> Tuple[RenderSegment, ...]:
    if len(segments) <= 1:
        return segments
    output: list[RenderSegment] = []
    for current in segments:
        if output and _can_coalesce(output[-1], current):
            previous = output[-1]
            output[-1] = replace(previous, end=max(previous.end, current.end))
            continue
        output.append(current)
    return tuple(output)


def build_render_plan(draft: DraftTimeline, local_paths: Mapping[str, str]) -> Tuple[RenderSegment, ...]:
    """Translate selected draft clips to concrete source-safe media segments."""
    output = []
    for clip in draft.selected:
        path = local_paths.get(clip.source_asset_id)
        if not path:
            raise ValueError(f"missing source path for selected clip {clip.clip_id}")
        if clip.end <= clip.start:
            raise ValueError(f"invalid selected clip boundary {clip.clip_id}")
        volume = float(clip.audio_volume)
        if volume < 0.0 or volume > 2.0:
            raise ValueError(f"invalid audio volume for selected clip {clip.clip_id}")
        output.append(RenderSegment(
            clip_id=clip.clip_id,
            source_asset_id=clip.source_asset_id,
            source_path=path,
            start=float(clip.start),
            end=float(clip.end),
            audio_muted=bool(clip.audio_muted),
            audio_volume=volume,
            caption_text=(str(clip.caption_text or "") if draft.captions_enabled else ""),
            caption_preset=str(draft.caption_preset or "classic"),
        ))
    if not output:
        raise ValueError("draft has no selected clips to render")
    return _coalesce_contiguous_segments(tuple(output))
