"""Build a source-safe render plan from an editable CutSell draft."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .contracts import DraftTimeline


@dataclass(frozen=True)
class RenderSegment:
    clip_id: str
    source_asset_id: str
    source_path: str
    start: float
    end: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)


def build_render_plan(draft: DraftTimeline, local_paths: Mapping[str, str]) -> Tuple[RenderSegment, ...]:
    """Translate selected draft clips to concrete source-safe media segments."""
    output = []
    for clip in draft.selected:
        path = local_paths.get(clip.source_asset_id)
        if not path:
            raise ValueError(f"missing source path for selected clip {clip.clip_id}")
        if clip.end <= clip.start:
            raise ValueError(f"invalid selected clip boundary {clip.clip_id}")
        output.append(RenderSegment(
            clip_id=clip.clip_id,
            source_asset_id=clip.source_asset_id,
            source_path=path,
            start=float(clip.start),
            end=float(clip.end),
        ))
    if not output:
        raise ValueError("draft has no selected clips to render")
    return tuple(output)
