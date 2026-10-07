"""Pure Draft-level caption display settings for the mobile editor."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .caption_render import CAPTION_FONTS, CAPTION_PRESETS  # noqa: F401  (re-exported)


def patch_caption_settings(
    draft: Mapping[str, Any],
    *,
    enabled: bool | None = None,
    preset: str | None = None,
    font: str | None = None,
) -> dict[str, Any]:
    if enabled is None and preset is None and font is None:
        raise ValueError("caption settings require enabled, preset and/or font")
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
    return out
