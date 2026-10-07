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
CENTER_X = PLAY_RES_X // 2
CENTER_Y = 1456            # caption centre at 75.8% of the frame height
SIDE_MARGIN = 70

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


def _style_line(look: str, font: str) -> str:
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
    return (
        f"Style: Caption,{family},{size},{primary},{primary},{outline},{back},"
        f"{bold},0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},5,"
        f"{SIDE_MARGIN},{SIDE_MARGIN},0,1"
    )


def _header(look: str, font: str) -> str:
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
        _style_line(look, font),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])


def _event(start: float, end: float, body: str, look: str) -> str:
    blur = "" if look.startswith("box") else "\\blur1.2"
    return (
        f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
        f"{{\\an5\\pos({CENTER_X},{CENTER_Y}){blur}}}{body}"
    )


def build_caption_ass(
    cues: Sequence[Tuple[float, float, str]],
    cue_words: Sequence[Sequence[Tuple[float, float, str]]],
    *,
    preset: str,
    duration_sec: float,
    font: str = "",
) -> str:
    """Body of the .ass file for one rendered segment, or "" when nothing is drawable.

    `cues` are (start, end, text) relative to the segment start. `cue_words[i]` holds the
    (start, end, word) timings of cue i and is only needed for the Highlight looks; when it
    is missing or does not match, that cue is drawn without a highlighted word.
    Cues are clamped to the segment's real duration and never overlap."""
    look, highlight_name = _PRESET_LOOKS.get(str(preset or "classic"), _PRESET_LOOKS["classic"])
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
        words = _usable_words(cue_words[index] if index < len(cue_words or ()) else (), text)
        if look != "highlight" or not words:
            events.append(_event(start, end, text, look))
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
            events.append(_event(step_start, step_end, " ".join(parts), look))
            cursor = step_end
    if not events:
        return ""
    return _header(look, font) + "\n" + "\n".join(events) + "\n"


def _usable_words(words: Iterable[Tuple[float, float, str]], cue_text: str):
    cleaned = [(float(s), float(e), clean_caption_text(w, 60)) for s, e, w in (words or ())]
    cleaned = [row for row in cleaned if row[2]]
    if not cleaned or " ".join(row[2] for row in cleaned) != cue_text:
        return []
    return cleaned
