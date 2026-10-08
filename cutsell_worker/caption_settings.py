"""Pure Draft-level caption display settings for the mobile editor."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .caption_render import CAPTION_FONTS, CAPTION_PRESETS, caption_layout  # noqa: F401  (re-exported)


def patch_caption_settings(
    draft: Mapping[str, Any],
    *,
    enabled: bool | None = None,
    preset: str | None = None,
    font: str | None = None,
    x: float | None = None,
    y: float | None = None,
    scale: float | None = None,
) -> dict[str, Any]:
    if all(value is None for value in (enabled, preset, font, x, y, scale)):
        raise ValueError("caption settings require enabled, preset, font, x, y and/or scale")
    out = deepcopy(dict(draft))
    if not isinstance(out.get("selected"), list):
        raise ValueError("draft requires selected list")
    if enabled is not None:
        out["captions_enabled"] = bool(enabled)
    if preset is not None:
        resolved = str(preset)
        if resolved not in CAPTION_PRESETS:
            raise ValueError("caption preset must be one of: " + ", ".join(sorted(CAPTION_PRESETS)))
        out["caption_preset"] = resolved
    if font is not None:
        chosen = str(font)
        if chosen not in CAPTION_FONTS:
            raise ValueError("caption font must be one of: " + ", ".join(sorted(CAPTION_FONTS)))
        out["caption_font"] = chosen
    if x is not None or y is not None or scale is not None:
        new_x, new_y, new_scale = caption_layout(x, y, scale)   # validates what was sent
        if x is not None:
            out["caption_x"] = new_x
        if y is not None:
            out["caption_y"] = new_y
        if scale is not None:
            out["caption_scale"] = new_scale
    return out
