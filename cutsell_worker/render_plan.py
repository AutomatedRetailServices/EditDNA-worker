"""Build a source-safe render plan from an editable CutSell draft."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Mapping, Tuple

from .contracts import DraftTimeline

if TYPE_CHECKING:  # pragma: no cover -- type-only, avoids a new hard runtime
    # dependency from render_plan.py onto the D-262 executor module (this
    # file otherwise only imports .contracts).
    from .visual_finishing_executor import VisualTransformSpec


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
    # D-262 (Visual Finishing EXECUTOR, OFFLINE ONLY -- no live caller sets
    # this; `build_render_plan` below never assigns it, so every existing
    # production segment keeps `visual_transform` `None` and the renderer's
    # video chain is therefore byte-identical to today's output). When
    # explicitly set (test-only today), a `VisualTransformSpec` is a pure,
    # already-authorized (D-260 policy -> D-262 executor) pixel-space
    # scale+crop geometry instruction consumed by `render.py`'s own
    # filtergraph construction -- never invented by the renderer itself,
    # matching this file's existing `audio_start`/`audio_end` precedent
    # exactly.
    visual_transform: "VisualTransformSpec | None" = None
    # An upstream-authorized visual interval is not silent recording slack.
    # Renderer may trim after this source time, never through it.
    trailing_trim_floor: float | None = None
    # Simple engine only (draft.diagnostics["engine"] == "simple"): short timed caption cues
    # for this segment, as (start, end, text) in seconds RELATIVE to the segment's own start.
    # Empty for every other draft, which keeps the single whole-clip caption exactly as before.
    # Simple-engine clips run many seconds; one cue for the whole clip would put the entire
    # paragraph on screen at once.
    caption_cues: Tuple[Tuple[float, float, str], ...] = ()
    # Per-word timings of each cue above (same order, same length); only the Highlight caption
    # looks read them, to colour the word being spoken.
    caption_cue_words: Tuple[Tuple[Tuple[float, float, str], ...], ...] = ()
    # Caption typeface key (caption_render.CAPTION_FONTS); "" = the default face.
    caption_font: str = ""
    # Caption placement / size for the whole video; None = the default (caption_render).
    caption_x: float | None = None
    caption_y: float | None = None
    caption_scale: float | None = None

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
    # Freeze and silence QC need each explicit action's original clip identity
    # and source interval. Joining adjacent muted actions would erase that
    # mapping even though the decoded pixels look continuous.
    if any(seg.audio_muted and seg.trailing_trim_floor is not None
           and seg.trailing_trim_floor >= seg.end for seg in (left, right)):
        return False
    if left.has_independent_audio_window or right.has_independent_audio_window:
        return False
    if left.source_asset_id != right.source_asset_id or left.source_path != right.source_path:
        return False
    if abs(float(right.start) - float(left.end)) > tolerance_sec:
        return False
    if left.audio_muted != right.audio_muted or abs(left.audio_volume - right.audio_volume) > 1e-6:
        return False
    if (left.caption_preset, left.caption_font, left.caption_x, left.caption_y, left.caption_scale) != (
            right.caption_preset, right.caption_font, right.caption_x, right.caption_y, right.caption_scale):
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
            floors = [f for f in (previous.trailing_trim_floor, current.trailing_trim_floor) if f is not None]
            output[-1] = replace(previous, end=max(previous.end, current.end),
                                 trailing_trim_floor=max(floors) if floors else None)
            continue
        output.append(current)
    return tuple(output)


CAPTION_CUE_MAX_WORDS = 3
CAPTION_CUE_MAX_GAP_SEC = 0.45
CAPTION_CUE_TAIL_HOLD_SEC = 0.30


def timed_caption_word_groups(clip) -> Tuple[Tuple[float, float, Tuple[Tuple[float, float, str], ...]], ...]:
    """Groups of up to three spoken words with their own on-screen window, relative to the
    clip start: (cue_start, cue_end, ((word_start, word_end, word), ...)). A group closes at
    three words, at punctuation, or at a pause. Each cue stays up until the next one starts
    (short hold after the last word otherwise), so text does not flicker between words.
    Returns () when the clip has no word timings or its caption was edited by hand
    (caption_text no longer equals the spoken text): the caller then falls back to the
    single whole-clip caption."""
    words = [w for w in (clip.words or ()) if float(w.end) > float(w.start)]
    if not words:
        return ()
    if " ".join(str(clip.caption_text or "").split()) != " ".join(str(clip.text or "").split()):
        return ()
    start, end = float(clip.start), float(clip.end)
    groups: list[list] = []
    current: list = []
    for word in words:
        if float(word.end) <= start or float(word.start) >= end:
            continue
        if current and float(word.start) - float(current[-1].end) > CAPTION_CUE_MAX_GAP_SEC:
            groups.append(current); current = []
        current.append(word)
        if len(current) >= CAPTION_CUE_MAX_WORDS or str(word.text).rstrip().endswith((".", "?", "!", ",")):
            groups.append(current); current = []
    if current:
        groups.append(current)
    cues = []
    for index, group in enumerate(groups):
        cue_start = max(0.0, float(group[0].start) - start)
        last_end = float(group[-1].end) - start
        if index + 1 < len(groups):
            next_start = float(groups[index + 1][0].start) - start
            cue_end = min(next_start, last_end + CAPTION_CUE_TAIL_HOLD_SEC * 2)
        else:
            cue_end = last_end + CAPTION_CUE_TAIL_HOLD_SEC
        cue_end = min(cue_end, end - start)
        text = " ".join(str(word.text) for word in group)
        if cue_end - cue_start >= 0.05 and text.strip():
            cues.append((round(cue_start, 3), round(cue_end, 3), tuple(
                (round(max(0.0, float(word.start) - start), 3), round(float(word.end) - start, 3), str(word.text))
                for word in group
            )))
    return tuple(cues)


def timed_caption_cues(clip) -> Tuple[Tuple[float, float, str], ...]:
    """(cue_start, cue_end, text) for each group of `timed_caption_word_groups`."""
    return tuple(
        (cue_start, cue_end, " ".join(word for _s, _e, word in words))
        for cue_start, cue_end, words in timed_caption_word_groups(clip)
    )


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
        visual_floors = []
        if (draft.diagnostics or {}).get("editorial_engine_v2"):
            for row in draft.diagnostics.get("editorial_engine_v2_continuity_restoration", ()):
                if (row.get("left_clip_id") in (clip.clip_id, clip.parent_semantic_clip_id)
                        and row.get("source_asset_id") == clip.source_asset_id
                        and row.get("basis") == "selected_neighbors_inside_high_confidence_audience_demonstration"):
                    floor = float(row["gap_end"])
                    if math.isfinite(floor) and clip.start < floor <= clip.end + 1e-6:
                        visual_floors.append(min(floor, float(clip.end)))
        # A wordless, explicitly selected visual scene owns its full source
        # window. Silence trimming must not silently erase its last frames.
        if clip.audio_muted and not clip.words and not clip.text.strip():
            visual_floors.append(float(clip.end))
        word_groups = (
            timed_caption_word_groups(clip)
            if draft.captions_enabled and (draft.diagnostics or {}).get("engine") == "simple"
            else ()
        )
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
            trailing_trim_floor=max(visual_floors) if visual_floors else None,
            caption_cues=tuple((a, b, " ".join(w for _s, _e, w in ws)) for a, b, ws in word_groups),
            caption_cue_words=tuple(ws for _a, _b, ws in word_groups),
            caption_font=str(getattr(draft, "caption_font", "") or ""),
            caption_x=getattr(draft, "caption_x", None),
            caption_y=getattr(draft, "caption_y", None),
            caption_scale=getattr(draft, "caption_scale", None),
        ))
    if not output:
        raise ValueError("draft has no selected clips to render")
    return _coalesce_contiguous_segments(tuple(output))
