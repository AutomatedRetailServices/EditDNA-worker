"""Burned-in caption look for short timed cues (Editor v2 design, Figma "Captions · On").

The design draws one short phrase (up to three words) in the lower third of a 9:16 frame:
bold, white with a soft dark shadow, centred at ~76% of the frame height.
Four looks are offered -- Classic, Highlight, Box, Yellow -- plus the two Box variants.
Highlight colours ONLY the word being spoken; its colour is the creator's choice.

Everything here is pure text generation: it returns the body of an .ass subtitle file. The
script resolution is fixed at 1080x1920, so sizes below are output pixels for the standard
export and scale proportionally for any other frame size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

PLAY_RES_X = 1080
PLAY_RES_Y = 1920

# Figma draws SF Pro Bold 18 px in a 412 px tall preview (18 / 412 * 1920 = 84 px). SF Pro
# cannot be shipped on a server, so the creator picks one of these freely licensed faces
# (files and licence texts in cutsell_worker/fonts/). Sizes are tuned per face so that every
# choice reads at about the same visual size; `bold` asks the renderer for the bold cut of
# a family and stays off for faces that only exist in one heavy weight.
#   key -> (family name inside the file, size in px at 1080x1920, bold flag)
CAPTION_FONTS = {
    "montserrat": ("Montserrat ExtraBold", 94, 0),
    "poppins": ("Poppins", 96, -1),
    "roboto": ("Roboto", 86, -1),
    "oswald": ("Oswald", 118, -1),
    "anton": ("Anton", 116, 0),
    "luckiest_guy": ("Luckiest Guy", 80, 0),
    "bebas_neue": ("Bebas Neue", 112, 0),
    "inter": ("Inter", 84, -1),
    "bangers": ("Bangers", 124, 0),
}
DEFAULT_CAPTION_FONT = "montserrat"
FONTS_DIR = Path(__file__).resolve().parent / "fonts"
SIDE_MARGIN = 70

# Where the caption sits and how big it is, for the WHOLE video (one setting per draft, the
# way mobile editors do it): the creator drags it and pinches it on the preview.
#   x, y   centre of the caption as a fraction of the frame width / height (0 = left / top)
#   scale  multiplier on the face's own size
DEFAULT_CAPTION_X = 0.5
DEFAULT_CAPTION_Y = 0.758      # Figma: caption centre at 75.8% of the frame height
DEFAULT_CAPTION_SCALE = 1.0
CAPTION_SCALE_RANGE = (0.5, 2.0)


def caption_layout(x: object = None, y: object = None, scale: object = None) -> Tuple[float, float, float]:
    """Validated (x, y, scale). None means "the default". Raises ValueError on anything that
    is not a finite number inside 0..1 (position) or the allowed scale range."""
    def number(value, default, low, high, name):
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number")
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
        if out != out or out in (float("inf"), float("-inf")) or not (low <= out <= high):
            raise ValueError(f"{name} must be between {low} and {high}")
        return out
    return (
        number(x, DEFAULT_CAPTION_X, 0.0, 1.0, "caption_x"),
        number(y, DEFAULT_CAPTION_Y, 0.0, 1.0, "caption_y"),
        number(scale, DEFAULT_CAPTION_SCALE, CAPTION_SCALE_RANGE[0], CAPTION_SCALE_RANGE[1], "caption_scale"),
    )


# Rough width of one character as a fraction of the font size, per face. Only used to keep a
# caption inside the frame; it does not have to be exact.
_CHAR_WIDTH = {
    "montserrat": 0.68, "poppins": 0.64, "roboto": 0.58, "oswald": 0.47, "anton": 0.45,
    "luckiest_guy": 0.66, "bebas_neue": 0.42, "inter": 0.62, "bangers": 0.44,
}
_EDGE_PAD = 36          # px kept free between the caption and the frame edge
_MIN_COLUMN = 520       # px: narrowest column a caption may be wrapped into


def _placement(x: float, y: float, scale: float, font: str) -> Tuple[int, int, int, int]:
    """(centre_x, centre_y, side_margin, font_size) in script pixels.

    The caption is drawn centred on the point the creator chose, inside a column that is
    symmetric around that point and never crosses a frame edge; long phrases wrap inside the
    column. Near an edge the centre is pulled in just enough for the column to keep a usable
    width, and the vertical centre leaves room for two lines."""
    key = str(font or "") if str(font or "") in CAPTION_FONTS else DEFAULT_CAPTION_FONT
    size = max(1, int(round(CAPTION_FONTS[key][1] * scale)))
    half_min = _MIN_COLUMN // 2 + _EDGE_PAD
    cx = int(round(min(max(float(x) * PLAY_RES_X, half_min), PLAY_RES_X - half_min)))
    half_column = min(cx, PLAY_RES_X - cx) - _EDGE_PAD
    margin = max(0, (PLAY_RES_X - 2 * half_column) // 2)
    half_height = int(size * 1.3) + _EDGE_PAD
    cy = int(round(min(max(float(y) * PLAY_RES_Y, half_height), PLAY_RES_Y - half_height)))
    return cx, cy, margin, size


def _fit_size(text: str, size: int, column: int, font: str) -> int:
    """Font size for one cue: the chosen size, or smaller when its longest word alone would
    be wider than the column (a single word cannot wrap)."""
    key = str(font or "") if str(font or "") in _CHAR_WIDTH else DEFAULT_CAPTION_FONT
    longest = max((len(word) for word in text.split()), default=0)
    if longest == 0:
        return size
    widest = longest * size * _CHAR_WIDTH[key]
    return size if widest <= column else max(12, int(size * column / widest))


WHITE = "FFFFFF"
INK = "0B1020"             # Figma dark used for the box and for words on a white box
YELLOW = "FFD60A"
HIGHLIGHT_COLOURS = {
    "green": "39FF78",     # Figma Highlight chip
    "red": "FF453A",
    "blue": "1AA6FF",      # Figma accent blue
}
DEFAULT_HIGHLIGHT = "green"

# preset -> (look, highlight colour name or None)
_PRESET_LOOKS = {
    "classic": ("plain_white", None),
    "yellow": ("plain_yellow", None),
    "highlight": ("highlight", DEFAULT_HIGHLIGHT),
    "highlight_green": ("highlight", "green"),
    "highlight_red": ("highlight", "red"),
    "highlight_blue": ("highlight", "blue"),
    "box": ("box_dark", None),          # black box, white words
    "box_light": ("box_light", None),   # white box, black words
    "clean": ("box_dark", None),        # pre-v2 name, kept so saved drafts still export
}

CAPTION_PRESETS = frozenset(_PRESET_LOOKS)


def _ass_colour(rgb_hex: str, alpha: int = 0) -> str:
    """RRGGBB -> ASS &HAABBGGRR (alpha 00 = opaque)."""
    r, g, b = rgb_hex[0:2], rgb_hex[2:4], rgb_hex[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_time(seconds: float) -> str:
    centis = max(0, int(round(float(seconds) * 100)))
    hours, rest = divmod(centis, 360_000)
    minutes, rest = divmod(rest, 6_000)
    secs, cs = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def clean_caption_text(raw: object, limit: int = 120) -> str:
    """One line of plain text. Braces and backslashes would be read by the renderer as
    styling commands, and a line break would open a new subtitle line, so none survive."""
    text = str(raw or "").replace("\x00", "")
    for forbidden in ("{", "}", "\\"):
        text = text.replace(forbidden, "")
    return " ".join(text.split())[:limit]


def _style_line(look: str, font: str, scale: float = 1.0, margin: int = SIDE_MARGIN) -> str:
    # Name, Fontname, Fontsize, Primary, Secondary, Outline, Back, Bold, Italic, Underline,
    # StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment,
    # MarginL, MarginR, MarginV, Encoding
    if look == "box_dark":
        primary, outline, back = _ass_colour(WHITE), _ass_colour(INK), _ass_colour(INK)
        border_style, outline_px, shadow_px = 3, 16, 0
    elif look == "box_light":
        primary, outline, back = _ass_colour(INK), _ass_colour(WHITE), _ass_colour(WHITE)
        border_style, outline_px, shadow_px = 3, 16, 0
    else:
        primary = _ass_colour(YELLOW if look == "plain_yellow" else WHITE)
        # Figma: shadow 0 1 3 rgba(0,0,0,.85), no stroke. A thin soft edge is kept as well so
        # white words stay readable over a bright or busy video.
        outline, back = _ass_colour("000000", 0x40), _ass_colour("000000", 0x26)
        border_style, outline_px, shadow_px = 1, 3, 3
    family, size, bold = CAPTION_FONTS.get(str(font or ""), CAPTION_FONTS[DEFAULT_CAPTION_FONT])
    size = max(1, int(round(size * scale)))
    outline_px = max(1, int(round(outline_px * scale)))
    shadow_px = int(round(shadow_px * scale))
    return (
        f"Style: Caption,{family},{size},{primary},{primary},{outline},{back},"
        f"{bold},0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},5,"
        f"{margin},{margin},0,1"
    )


def _header(look: str, font: str, scale: float = 1.0, margin: int = SIDE_MARGIN) -> str:
    return "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(look, font, scale, margin),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])


def _event(start: float, end: float, body: str, look: str, centre: Tuple[int, int], size: int | None = None) -> str:
    blur = "" if look.startswith("box") else "\\blur1.2"
    fit = f"\\fs{size}" if size else ""
    return (
        f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
        f"{{\\an5\\pos({centre[0]},{centre[1]}){blur}{fit}}}{body}"
    )


def build_caption_ass(
    cues: Sequence[Tuple[float, float, str]],
    cue_words: Sequence[Sequence[Tuple[float, float, str]]],
    *,
    preset: str,
    duration_sec: float,
    font: str = "",
    x: object = None,
    y: object = None,
    scale: object = None,
) -> str:
    """Body of the .ass file for one rendered segment, or "" when nothing is drawable.

    `cues` are (start, end, text) relative to the segment start. `cue_words[i]` holds the
    (start, end, word) timings of cue i and is only needed for the Highlight looks; when it
    is missing or does not match, that cue is drawn without a highlighted word.
    Cues are clamped to the segment's real duration and never overlap."""
    look, highlight_name = _PRESET_LOOKS.get(str(preset or "classic"), _PRESET_LOOKS["classic"])
    pos_x, pos_y, size_scale = caption_layout(x, y, scale)
    centre_x, centre_y, margin, base_size = _placement(pos_x, pos_y, size_scale, font)
    centre = (centre_x, centre_y)
    column = PLAY_RES_X - 2 * margin
    duration = float(duration_sec)
    events: list[str] = []
    previous_end = 0.0
    for index, (raw_start, raw_end, raw_text) in enumerate(cues or ()):
        text = clean_caption_text(raw_text)
        start = max(float(raw_start), previous_end)
        end = min(float(raw_end), duration)
        if not text or end - start < 0.05:
            continue
        previous_end = end
        fitted = _fit_size(text, base_size, column, font)
        fit = fitted if fitted != base_size else None
        words = _usable_words(cue_words[index] if index < len(cue_words or ()) else (), text)
        if look != "highlight" or not words:
            events.append(_event(start, end, text, look, centre, fit))
            continue
        active = _ass_colour(HIGHLIGHT_COLOURS[highlight_name or DEFAULT_HIGHLIGHT])
        base = _ass_colour(WHITE)
        cursor = start
        for position, (word_start, _word_end, _word) in enumerate(words):
            step_start = cursor if position == 0 else max(cursor, min(float(word_start), end))
            if position + 1 < len(words):
                step_end = max(step_start, min(float(words[position + 1][0]), end))
            else:
                step_end = end
            if step_end - step_start < 0.02:
                continue
            parts = [
                (f"{{\\1c{active}}}{word}{{\\1c{base}}}" if i == position else word)
                for i, (_s, _e, word) in enumerate(words)
            ]
            events.append(_event(step_start, step_end, " ".join(parts), look, centre, fit))
            cursor = step_end
    if not events:
        return ""
    return _header(look, font, size_scale, margin) + "\n" + "\n".join(events) + "\n"


def _usable_words(words: Iterable[Tuple[float, float, str]], cue_text: str):
    cleaned = [(float(s), float(e), clean_caption_text(w, 60)) for s, e, w in (words or ())]
    cleaned = [row for row in cleaned if row[2]]
    if not cleaned or " ".join(row[2] for row in cleaned) != cue_text:
        return []
    return cleaned
