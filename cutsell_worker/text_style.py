"""Text look chosen in the editor's Fonts / Style sheet (PO 30 Oct 2026), for the captions
(`draft.caption_style`) and for each text overlay (`overlay.style`, `overlay.font`).

The iPhone draws it (preview and on-phone export, the same SwiftUI view); the server only checks
that what is stored is sane -- known keys, colours as #RRGGBB, numbers inside their ranges -- so a
draft can never carry junk. Every key is optional: a missing key keeps the look's default.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
COLOR_KEYS = ("color", "stroke_color", "bg_color", "shadow_color", "highlight_color")
NUMBER_RANGES = {
    "opacity": (0.0, 1.0),
    "stroke_width": (0.0, 12.0),
    "bg_opacity": (0.0, 1.0),
    "bg_radius": (0.0, 40.0),
    "shadow_opacity": (0.0, 1.0),
    "shadow_blur": (0.0, 20.0),
    "shadow_distance": (0.0, 20.0),
    "letter_spacing": (-5.0, 30.0),
    "line_spacing": (-20.0, 60.0),
}
CHOICES = {"align": ("left", "center", "right"), "preset": (
    "plain", "outline", "thick", "shadow", "dark_box", "light_box", "pop", "neon", "soft_box", "glow")}
BOOL_KEYS = ("uppercase", "bold", "italic")


def validate_text_style(value: Any, name: str = "style") -> dict:
    """The style as stored (a plain dict), or ValueError. None / {} = the default look."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    out: dict = {}
    for key, item in value.items():
        if key in COLOR_KEYS:
            if item is None:
                continue
            if not isinstance(item, str) or not _HEX.match(item):
                raise ValueError(f"{name}.{key} must be a colour like #FF0000")
            out[key] = item.upper()
        elif key in NUMBER_RANGES:
            if isinstance(item, bool):
                raise ValueError(f"{name}.{key} must be a number")
            try:
                number = float(item)
            except (TypeError, ValueError):
                raise ValueError(f"{name}.{key} must be a number") from None
            low, high = NUMBER_RANGES[key]
            if number != number or not low <= number <= high:
                raise ValueError(f"{name}.{key} must be between {low} and {high}")
            out[key] = number
        elif key in CHOICES:
            if item not in CHOICES[key]:
                raise ValueError(f"{name}.{key} must be one of: " + ", ".join(CHOICES[key]))
            out[key] = item
        elif key in BOOL_KEYS:
            out[key] = bool(item)
        else:
            raise ValueError(f"{name} has an unknown setting: {str(key)[:40]}")
    return out
