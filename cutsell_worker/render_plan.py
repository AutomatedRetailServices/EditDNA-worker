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
    # D-036: physical-fragment provenance carried through from the DraftClip
    # this segment was built from -- see contracts.py's DraftClip fields and
    # effective_render_fragment_id/effective_parent_semantic_clip_id for the
    # full identity contract. This is the "PhysicalRenderPlan" representation
    # (already existed as RenderSegment/build_render_plan); CanonicalEditPlan
    # remains the semantic source of truth and is never rewritten into these
    # physical terms.
    render_fragment_id: str | None = None
    parent_semantic_clip_id: str | None = None
    fragment_index: int | None = None
    fragment_count: int | None = None
    boundary_reason: str | None = None
    # D-214 (Pacing V2 renderer/timeline contract, OFFLINE ONLY -- no live
    # caller sets these; `build_render_plan` below never assigns them, so
    # every existing production segment keeps `audio_start`/`audio_end`
    # `None` and therefore an audio window IDENTICAL to `start`/`end`,
    # exactly today's behavior). `start`/`end` remain the canonical VIDEO
    # source window, unchanged in meaning. `audio_start`/`audio_end`, when
    # explicitly set (test-only today), let a segment's own AUDIO source
    # window diverge from its video window: `audio_start < start` is a
    # leading audio window (this segment's own audio begins before its own
    # video -- the per-segment primitive under a J-cut, from the RIGHT
    # segment's point of view); `audio_end > end` is a trailing audio
    # window (this segment's own audio continues after its own video ends
    # -- the per-segment primitive under an L-cut, from the LEFT segment's
    # point of view). A small instance of either (or both) on the same
    # join is a micro audio overlap. This is a pure DATA contract -- no
    # mode label, no eligibility decision, no threshold lives here; the
    # renderer only ever realizes whatever window it is given (see
    # `render.py`'s own "no mode selection logic" contract).
    audio_start: float | None = None
    audio_end: float | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def effective_audio_start(self) -> float:
        """The AUDIO source window's start -- `audio_start` when explicitly
        set, else identical to the VIDEO window's own `start` (today's only
        behavior on every live-produced `RenderSegment`)."""
        return self.start if self.audio_start is None else float(self.audio_start)

    @property
    def effective_audio_end(self) -> float:
        """The AUDIO source window's end -- `audio_end` when explicitly
        set, else identical to the VIDEO window's own `end`."""
        return self.end if self.audio_end is None else float(self.audio_end)

    @property
    def has_independent_audio_window(self) -> bool:
        """True only when this segment's own audio window has been
        DELIBERATELY set to diverge from its video window -- never true for
        any segment `build_render_plan` produces today."""
        return self.audio_start is not None or self.audio_end is not None

    @property
    def audio_duration_sec(self) -> float:
        return max(0.0, self.effective_audio_end - self.effective_audio_start)


def _can_coalesce(left: RenderSegment, right: RenderSegment, *, tolerance_sec: float = 0.05) -> bool:
    """Return True when a hard cut would be visually/media-equivalent to continuity.

    Two selected clips that touch in the same source should not be rendered as two
    separate files and concatenated again. That creates a redundant decoder/encoder
    boundary at the exact same source frame and can show up as a visible jump even
    though the creator never stopped. Keep separate segments only when playback or
    caption settings materially differ.

    D-214: never coalesce across a join where either segment carries a
    DELIBERATE, non-default audio window (`has_independent_audio_window`).
    Such a window is a live Pacing V2 transition plan physically realized
    on this exact pair -- merging the two segments into one would silently
    erase that plan (D-213's own "coalesce interaction" forensic finding)
    rather than ever produce a wrong result; failing to merge is always the
    safe direction. No existing production segment ever sets this field, so
    this guard changes nothing about today's live coalescing behavior.
    """
    if left.has_independent_audio_window or right.has_independent_audio_window:
        return False
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
            render_fragment_id=getattr(clip, "render_fragment_id", None),
            parent_semantic_clip_id=getattr(clip, "parent_semantic_clip_id", None),
            fragment_index=getattr(clip, "fragment_index", None),
            fragment_count=getattr(clip, "fragment_count", None),
            boundary_reason=getattr(clip, "boundary_reason", None),
        ))
    if not output:
        raise ValueError("draft has no selected clips to render")
    return _coalesce_contiguous_segments(tuple(output))
