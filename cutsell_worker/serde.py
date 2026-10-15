"""JSON-safe serialization helpers for RQ/API boundaries."""
from __future__ import annotations

from dataclasses import asdict, fields
from enum import Enum
from typing import Any

from .contracts import (
    AudioClip,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    MediaOverlay,
    ProcessingRequest,
    SemanticRole,
    SourceAsset,
    TextOverlay,
    Word,
    ZoomKeyframe,
)

from .sfx_library import SFX_IDS  # noqa: E402
from .caption_render import CAPTION_FONTS, CAPTION_PRESETS, DEFAULT_CAPTION_FONT, caption_layout  # noqa: E402,F401


#: D-134 current effective default for the Overlap pacing permission, taken
#: directly from the pre-D-134 code, not guessed: `ProcessingRequest.
#: audio_overlap`'s own dataclass default (contracts.py) and the iOS
#: `audioOverlap` toggle's own initial state (NewCutView.swift) were both
#: already `False` before this task. Never change this without a Product
#: Owner decision -- see CUTSELL_DECISIONS.md D-134 Part 4.
_DIALOGUE_OVERLAP_DEFAULT = False

# D-134 diagnostic source labels -- see docs/CUTSELL_DECISIONS.md D-134 Part 5.
DIALOGUE_OVERLAP_SOURCE_CANONICAL = "canonical_request"
DIALOGUE_OVERLAP_SOURCE_LEGACY = "legacy_audio_overlap"
DIALOGUE_OVERLAP_SOURCE_DEFAULT = "default"


def _normalize_dialogue_overlap(payload: dict) -> tuple[bool, dict]:
    """D-134's ONE normalization point for the Overlap pacing permission.

    Precedence (CUTSELL_DECISIONS.md D-134 Part 2, rules 1-5):
      1. Both `dialogue_overlap_enabled` (canonical, D-129 naming) and
         `audio_overlap` (legacy iOS field) represent the SAME future
         pacing permission.
      2. If `dialogue_overlap_enabled` is explicitly present (not None):
         it wins, regardless of what `audio_overlap` carries.
      3. Else if legacy `audio_overlap` is explicitly present (not None):
         normalize its value into the canonical field.
      4. Else: preserve the current effective default (`False`, per Part 4).
      5. Downstream (`ProcessingRequest.dialogue_overlap_enabled`) receives
         this ONE canonical normalized value; `ProcessingRequest.
         audio_overlap` keeps carrying the raw legacy value unchanged, for
         backward introspection only -- no consumer reads either field yet.

    Returns `(dialogue_overlap_enabled, diagnostics)` where `diagnostics` is
    the compact, request-level (not per-BestTake-family) block from Part 5:
    `dialogue_overlap_enabled`, `dialogue_overlap_source`,
    `legacy_audio_overlap_present`, `canonical_dialogue_overlap_present`.
    """
    canonical_value = payload.get("dialogue_overlap_enabled")
    canonical_present = canonical_value is not None
    legacy_value = payload.get("audio_overlap")
    legacy_present = legacy_value is not None

    if canonical_present:
        enabled, source = bool(canonical_value), DIALOGUE_OVERLAP_SOURCE_CANONICAL
    elif legacy_present:
        enabled, source = bool(legacy_value), DIALOGUE_OVERLAP_SOURCE_LEGACY
    else:
        enabled, source = _DIALOGUE_OVERLAP_DEFAULT, DIALOGUE_OVERLAP_SOURCE_DEFAULT

    diagnostics = {
        "dialogue_overlap_enabled": enabled,
        "dialogue_overlap_source": source,
        "legacy_audio_overlap_present": legacy_present,
        "canonical_dialogue_overlap_present": canonical_present,
    }
    return enabled, diagnostics


def request_from_dict(payload: dict) -> ProcessingRequest:
    sources = tuple(
        SourceAsset(
            source_asset_id=str(item["source_asset_id"]),
            project_id=str(payload["project_id"]),
            user_id=str(payload["user_id"]),
            original_name=str(item.get("original_name") or "source"),
            source_order=int(item.get("source_order", index)),
            duration_sec=float(item.get("duration_sec") or 0.0),
            uri=str(item["uri"]),
            has_audio=bool(item.get("has_audio", True)),
            # D-134: `metadata` already carried arbitrary JSON-safe keys
            # before this task; it is now also the carrier for the optional,
            # descriptive-only iOS media-ingestion fields (D-133 reality) --
            # see docs/CUTSELL_DECISIONS.md D-134 Part 6. Unknown/absent
            # keys are simply not present in the dict; nothing is required.
            metadata=dict(item.get("metadata") or {}),
        )
        for index, item in enumerate(payload.get("sources") or ())
    )
    if not sources:
        raise ValueError("processing request requires at least one source")
    dialogue_overlap_enabled, overlap_diagnostics = _normalize_dialogue_overlap(payload)
    return ProcessingRequest(
        project_id=str(payload["project_id"]),
        user_id=str(payload["user_id"]),
        sources=sources,
        preferred_source_order=tuple(payload.get("preferred_source_order") or ()),
        audio_overlap=bool(payload.get("audio_overlap", False)),
        language_hint=(str(payload["language_hint"]) if payload.get("language_hint") else None),
        dialogue_overlap_enabled=dialogue_overlap_enabled,
        auto_zoom_enabled=bool(payload.get("auto_zoom_enabled", False)),
        overlap_diagnostics=overlap_diagnostics,
    )


def _word_from_dict(item: dict) -> Word:
    return Word(
        text=str(item.get("text") or ""),
        start=float(item["start"]),
        end=float(item["end"]),
        confidence=(float(item["confidence"]) if item.get("confidence") is not None else None),
    )


def _draft_clip_from_dict(item: dict, *, selected_default: bool) -> DraftClip:
    words = tuple(_word_from_dict(dict(word)) for word in item.get("words") or ())
    volume = float(item.get("audio_volume", 1.0))
    if volume < 0.0 or volume > 2.0:
        raise ValueError("audio_volume must be between 0.0 and 2.0")
    return DraftClip(
        clip_id=str(item["clip_id"]),
        source_asset_id=str(item["source_asset_id"]),
        source_order=int(item.get("source_order", 0)),
        start=float(item["start"]),
        end=float(item["end"]),
        text=str(item.get("text") or ""),
        caption_text=str(item.get("caption_text") if item.get("caption_text") is not None else item.get("text") or ""),
        words=words,
        semantic_role=SemanticRole(str(item.get("semantic_role") or SemanticRole.OTHER.value)),
        take_group_id=(str(item["take_group_id"]) if item.get("take_group_id") else None),
        selected=bool(item.get("selected", selected_default)),
        audio_muted=bool(item.get("audio_muted", False)),
        audio_volume=volume,
        audio_pre_roll=_roll(item.get("audio_pre_roll")),
        audio_post_roll=_roll(item.get("audio_post_roll")),
        mirror=bool(item.get("mirror", False)),
        manual_end=bool(item.get("manual_end", False)),
        speed=_clip_speed(item.get("speed", 1.0)),
    )


#: Editor v5 clip speeds the export supports (pitch-preserving atempo, frame-exact setpts).
CLIP_SPEEDS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _clip_speed(value) -> float:
    try:
        number = float(1.0 if value is None else value)
    except (TypeError, ValueError):
        raise ValueError("speed must be one of 0.5, 1, 1.5, 2, 3") from None
    for allowed in CLIP_SPEEDS:
        if abs(number - allowed) < 1e-9:
            return allowed
    raise ValueError("speed must be one of 0.5, 1, 1.5, 2, 3")


def timeline_duration_sec(clips) -> float:
    """Length of the final timeline: each selected clip lasts (end - start) / speed."""
    return sum(max(0.0, clip.end - clip.start) / float(getattr(clip, "speed", 1.0) or 1.0) for clip in clips)


#: Longest audio roll a clip may carry on one side of a join (a join overlaps pre + post <= 0.12 s).
MAX_AUDIO_ROLL_SEC = 0.06
ZOOM_SCALE_RANGE = (1.0, 1.25)
ZOOM_DURATION_RANGE = (0.4, 4.0)


def _roll(value) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        raise ValueError("audio roll must be a number") from None
    return round(max(0.0, min(MAX_AUDIO_ROLL_SEC, number)), 3)


def _zoom_keyframe_from_dict(item: dict) -> ZoomKeyframe:
    try:
        source_time = float(item["source_time"])
        duration = float(item.get("duration", 1.5))
        scale = float(item.get("scale", 1.08))
    except (KeyError, TypeError, ValueError):
        raise ValueError("zoom keyframe needs source_time, duration and scale") from None
    if source_time < 0:
        raise ValueError("zoom keyframe source_time must not be negative")
    if not ZOOM_DURATION_RANGE[0] <= duration <= ZOOM_DURATION_RANGE[1]:
        raise ValueError("zoom keyframe duration out of range")
    if not ZOOM_SCALE_RANGE[0] <= scale <= ZOOM_SCALE_RANGE[1]:
        raise ValueError("zoom keyframe scale out of range")
    return ZoomKeyframe(
        keyframe_id=str(item.get("keyframe_id") or f"z{int(source_time * 1000)}"),
        source_asset_id=str(item.get("source_asset_id") or ""),
        source_time=source_time, duration=duration, scale=scale,
        kind=str(item.get("kind") or "key")[:32], origin=str(item.get("origin") or "ai")[:16],
    )


def _text_overlay_from_dict(item: dict) -> TextOverlay:
    text = str(item.get("text") or "").strip()
    if not text or len(text) > 500:
        raise ValueError("text overlay text must contain 1 to 500 characters")
    start = float(item["start"]); end = float(item["end"])
    x = float(item.get("x", 0.5)); y = float(item.get("y", 0.2)); scale = float(item.get("scale", 1.0))
    if start < 0 or end <= start:
        raise ValueError("text overlay end must be after start")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("text overlay position must be normalized 0 to 1")
    if not 0.5 <= scale <= 3.0:
        raise ValueError("text overlay scale must be between 0.5 and 3.0")
    return TextOverlay(str(item["overlay_id"]), text, start, end, x, y, scale)


def _media_overlay_from_dict(item: dict) -> MediaOverlay:
    kind = str(item.get("kind") or "")
    if kind not in {"photo", "video"}:
        raise ValueError("media overlay kind must be photo or video")
    uri = str(item.get("uri") or "")
    if not uri:
        raise ValueError("media overlay uri is required")
    start = float(item["start"]); end = float(item["end"])
    x = float(item.get("x", 0.5)); y = float(item.get("y", 0.5)); width = float(item.get("width", 0.4))
    source_start = float(item.get("source_start", 0.0))
    source_end = float(item["source_end"]) if item.get("source_end") is not None else None
    if start < 0 or end <= start:
        raise ValueError("media overlay end must be after start")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("media overlay position must be normalized 0 to 1")
    if not 0.1 <= width <= 1.0:
        raise ValueError("media overlay width must be between 0.1 and 1.0")
    if source_start < 0 or (source_end is not None and source_end <= source_start):
        raise ValueError("media overlay source trim is invalid")
    if kind == "photo" and (source_start != 0.0 or source_end is not None):
        raise ValueError("photo overlay cannot use source trim")
    try:
        layer = int(item.get("layer", 0) or 0)
        rotation = float(item.get("rotation", 0.0) or 0.0)
    except (TypeError, ValueError):
        raise ValueError("media overlay layer/rotation must be numbers") from None
    if isinstance(item.get("layer"), float) and not float(item["layer"]).is_integer():
        raise ValueError("media overlay layer must be a whole number")
    if not 0 <= layer <= MAX_OVERLAY_LAYER:
        raise ValueError(f"media overlay layer must be between 0 and {MAX_OVERLAY_LAYER}")
    layout = str(item.get("layout") or "float")
    if layout not in OVERLAY_LAYOUTS:
        raise ValueError("media overlay layout must be one of: " + ", ".join(OVERLAY_LAYOUTS))
    edge = str(item.get("edge") or "sharp")
    if edge not in OVERLAY_EDGES:
        raise ValueError("media overlay edge must be one of: " + ", ".join(OVERLAY_EDGES))
    if not -360.0 <= rotation <= 360.0:
        raise ValueError("media overlay rotation must be between -360 and 360 degrees")
    return MediaOverlay(
        overlay_id=str(item["overlay_id"]), kind=kind, uri=uri, start=start, end=end,
        x=x, y=y, width=width, source_start=source_start, source_end=source_end,
        mute_audio=bool(item.get("mute_audio", True)),
        layer=layer, layout=layout, edge=edge, fill=bool(item.get("fill", False)), rotation=rotation,
    )


MAX_OVERLAY_LAYER = 50
OVERLAY_LAYOUTS = ("float", "side_by_side", "stacked", "left", "right", "top", "bottom", "full")
OVERLAY_EDGES = ("sharp", "soft", "extra_soft")
AUDIO_CLIP_KINDS = ("voice", "music", "sfx")
MAX_AUDIO_CLIPS = 100
#: An audio clip may start up to this far past the end of the timeline (rounding slack).
AUDIO_CLIP_START_SLACK_SEC = 0.05


def _audio_clip_from_dict(item: dict) -> AudioClip:
    audio_id = str(item.get("audio_id") or "").strip()
    if not audio_id or len(audio_id) > 200:
        raise ValueError("audio clip audio_id is required")
    kind = str(item.get("kind") or "")
    if kind not in AUDIO_CLIP_KINDS:
        raise ValueError("audio clip kind must be voice, music or sfx")
    asset_id = str(item.get("asset_id") or "").strip() or None
    sfx_id = str(item.get("sfx_id") or "").strip() or None
    if kind in ("voice", "music"):
        if not asset_id or len(asset_id) > 200:
            raise ValueError(f"{kind} audio clip requires asset_id")
        sfx_id = None
    else:
        if sfx_id not in SFX_IDS:
            raise ValueError("sfx audio clip requires sfx_id, one of: " + ", ".join(sorted(SFX_IDS)))
        asset_id = None
    try:
        start = float(item["start"])
        source_in = float(item.get("source_in", 0.0) or 0.0)
        source_out = float(item["source_out"])
        volume = float(item.get("volume", 1.0) if item.get("volume") is not None else 1.0)
    except (KeyError, TypeError, ValueError):
        raise ValueError("audio clip needs numeric start, source_in and source_out") from None
    if start < 0 or source_in < 0:
        raise ValueError("audio clip start and source_in must not be negative")
    if source_out <= source_in:
        raise ValueError("audio clip source_out must be after source_in")
    if not 0.0 <= volume <= 2.0:
        raise ValueError("audio clip volume must be between 0.0 and 2.0")
    return AudioClip(
        audio_id=audio_id, kind=kind, start=start, source_in=source_in, source_out=source_out,
        asset_id=asset_id, sfx_id=sfx_id, volume=volume,
        duck_video=bool(item.get("duck_video", False)), clean=bool(item.get("clean", False)),
    )


def draft_from_dict(payload: dict) -> DraftTimeline:
    if not isinstance(payload, dict):
        raise ValueError("draft must be an object")
    selected = tuple(_draft_clip_from_dict(dict(item), selected_default=True) for item in payload.get("selected") or ())
    alternates = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("alternates") or ())
    discarded = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("discarded") or ())
    if not selected:
        raise ValueError("draft requires at least one selected clip")
    preset = str(payload.get("caption_preset") or "classic")
    if preset not in CAPTION_PRESETS:
        raise ValueError("caption_preset must be one of: " + ", ".join(sorted(CAPTION_PRESETS)))
    caption_font = str(payload.get("caption_font") or DEFAULT_CAPTION_FONT)
    if caption_font not in CAPTION_FONTS:
        raise ValueError("caption_font must be one of: " + ", ".join(sorted(CAPTION_FONTS)))
    caption_x, caption_y, caption_scale = caption_layout(
        payload.get("caption_x"), payload.get("caption_y"), payload.get("caption_scale"))
    try:
        rotation = int(float(payload.get("rotation") or 0)) % 360
    except (TypeError, ValueError):
        raise ValueError("rotation must be 0, 90, 180 or 270") from None
    if rotation not in VIDEO_ROTATIONS:
        raise ValueError("rotation must be 0, 90, 180 or 270")
    text_overlays = tuple(_text_overlay_from_dict(dict(item)) for item in payload.get("text_overlays") or ())
    media_overlays = tuple(_media_overlay_from_dict(dict(item)) for item in payload.get("media_overlays") or ())
    # Editor v5: a clip played at speed S lasts (end - start) / S on the final timeline.
    total_duration = timeline_duration_sec(selected)
    if any(overlay.end > total_duration + 1e-6 for overlay in text_overlays):
        raise ValueError("text overlay exceeds draft timeline duration")
    if any(overlay.end > total_duration + 1e-6 for overlay in media_overlays):
        raise ValueError("media overlay exceeds draft timeline duration")
    raw_audio = payload.get("audio_clips") or ()
    if not isinstance(raw_audio, (list, tuple)):
        raise ValueError("audio_clips must be a list")
    if len(raw_audio) > MAX_AUDIO_CLIPS:
        raise ValueError(f"at most {MAX_AUDIO_CLIPS} audio clips are allowed")
    audio_clips = tuple(_audio_clip_from_dict(dict(item)) for item in raw_audio)
    if len({clip.audio_id for clip in audio_clips}) != len(audio_clips):
        raise ValueError("audio clip audio_id values must be unique")
    # A clip that runs past the end is allowed (the export cuts it there); one that starts after
    # the end would never be heard.
    if any(clip.start > total_duration + AUDIO_CLIP_START_SLACK_SEC for clip in audio_clips):
        raise ValueError("audio clip starts after the end of the draft timeline")
    video_volume = _gain(payload.get("video_volume"), "video_volume")
    return DraftTimeline(
        schema_version=str(payload.get("schema_version") or "cutsell.v1"),
        project_id=str(payload["project_id"]),
        strategy=EditStrategy(str(payload.get("strategy") or EditStrategy.MIXED.value)),
        selected=selected,
        alternates=alternates,
        discarded=discarded,
        diagnostics=dict(payload.get("diagnostics") or {}),
        captions_enabled=bool(payload.get("captions_enabled", True)),
        caption_preset=preset,
        caption_font=caption_font,
        caption_x=caption_x,
        caption_y=caption_y,
        caption_scale=caption_scale,
        text_overlays=text_overlays,
        media_overlays=media_overlays,
        mirror=bool(payload.get("mirror", False)),
        rotation=rotation,
        audio_overlap_enabled=bool(payload.get("audio_overlap_enabled", False)),
        auto_zoom_enabled=bool(payload.get("auto_zoom_enabled", False)),
        zoom_keyframes=tuple(_zoom_keyframe_from_dict(dict(item)) for item in payload.get("zoom_keyframes") or ()),
        user_media_source_ids=tuple(
            str(item.get("source_asset_id")) for item in payload.get("broll_sources") or ()
            if isinstance(item, dict) and item.get("source_asset_id")
        ),
        audio_clips=audio_clips,
        video_volume=video_volume,
        video_clean_voice=bool(payload.get("video_clean_voice", False)),
    )


def _gain(value, name: str) -> float:
    if value is None:
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number between 0.0 and 2.0") from None
    if not 0.0 <= number <= 2.0:
        raise ValueError(f"{name} must be between 0.0 and 2.0")
    return number


VIDEO_ROTATIONS = (0, 90, 180, 270)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        # One level at a time (nested dataclasses come back through here), so editor-only
        # settings can be left out while they are at their defaults: cut results and stored
        # drafts stay exactly as before for every draft that does not use them.
        data = {item.name: getattr(value, item.name) for item in fields(value)}
        if isinstance(value, DraftTimeline):
            for key in ("mirror", "rotation", "audio_overlap_enabled", "auto_zoom_enabled", "zoom_keyframes",
                        "user_media_source_ids", "audio_clips", "video_clean_voice"):
                if not data.get(key):
                    data.pop(key, None)
            if data.get("video_volume") == 1.0:
                data.pop("video_volume", None)
        elif isinstance(value, DraftClip):
            for key in ("audio_pre_roll", "audio_post_roll", "mirror", "manual_end"):
                if not data.get(key):
                    data.pop(key, None)
            if data.get("speed") == 1.0:
                data.pop("speed", None)
        elif isinstance(value, MediaOverlay):
            for key, default in (("layer", 0), ("layout", "float"), ("edge", "sharp"), ("fill", False),
                                 ("rotation", 0.0)):
                if data.get(key) == default:
                    data.pop(key, None)
        elif isinstance(value, AudioClip):
            for key in ("asset_id", "sfx_id"):
                if data.get(key) is None:
                    data.pop(key, None)
        return _json_safe(data)
    return value


def result_to_dict(result) -> dict:
    return _json_safe(result)
