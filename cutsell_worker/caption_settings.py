"""Pure Draft-level caption display settings for the mobile editor."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

CAPTION_PRESETS = {"classic", "clean"}


def patch_caption_settings(
    draft: Mapping[str, Any],
    *,
    enabled: bool | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    if enabled is None and preset is None:
        raise ValueError("caption settings require enabled and/or preset")
    out = deepcopy(dict(draft))
    if not isinstance(out.get("selected"), list):
        raise ValueError("draft requires selected list")
    if enabled is not None:
        out["captions_enabled"] = bool(enabled)
    if preset is not None:
        resolved = str(preset)
        if resolved not in CAPTION_PRESETS:
            raise ValueError("caption preset must be classic or clean")
        out["caption_preset"] = resolved
    return out
