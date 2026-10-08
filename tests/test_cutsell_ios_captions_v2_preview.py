"""Editor v2 captions on iPhone -- the preview must follow the server's rules.

The app draws captions over the video while editing. To show what the export will
really burn in, `CaptionPreviewRules.swift` is a port of the server's caption code.
These tests read the Swift source (no Swift toolchain here, the convention in this
repo) and check that every number it carries is still the server's number, so the
two cannot drift apart silently. The real build check is `cutsell-ios-ci.yml`.
"""
import re
from pathlib import Path

import pytest

from cutsell_worker import caption_render, render_plan

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile/ios"
RULES = IOS / "CutSell/CaptionPreviewRules.swift"
OVERLAY = IOS / "CutSell/CaptionOverlayView.swift"
PLAYBACK = IOS / "CutSell/DraftPlaybackView.swift"
VIEW_MODEL = IOS / "CutSell/DraftEditorViewModel.swift"
PROJECT = IOS / "project.yml"

SPEC = re.compile(
    r'CaptionFontSpec\(key: "(?P<key>\w+)", displayName: "(?P<name>[^"]+)", '
    r'fileName: "(?P<file>[^"]+)", fileExtension: "(?P<ext>\w+)",\s*'
    r'postScriptName: "(?P<ps>[^"]+)", renderSize: (?P<size>[\d.]+), charWidth: (?P<cw>[\d.]+),\s*'
    r"unitsPerEm: (?P<upem>\d+), ascent: (?P<asc>\d+), descent: (?P<desc>\d+), lineGap: (?P<gap>\d+), "
    r"winAscent: (?P<wasc>\d+), winDescent: (?P<wdesc>\d+)\)"
)


def _swift_fonts():
    return {m["key"]: m.groupdict() for m in SPEC.finditer(RULES.read_text())}


def _number(source: str, name: str) -> float:
    match = re.search(rf"static let {name}(?:: \w+)? = (0x[0-9A-Fa-f]+|[\d.]+)", source)
    assert match, f"{name} not found"
    raw = match.group(1)
    return float(int(raw, 16)) if raw.startswith("0x") else float(raw)


# ---------------------------------------------------------------------------
# Phrase grouping: at most three words, same pauses, same hold.
# ---------------------------------------------------------------------------

def test_phrase_rules_equal_the_servers():
    source = RULES.read_text()
    assert _number(source, "cueMaxWords") == render_plan.CAPTION_CUE_MAX_WORDS == 3
    assert _number(source, "cueMaxGapSec") == render_plan.CAPTION_CUE_MAX_GAP_SEC
    assert _number(source, "cueTailHoldSec") == render_plan.CAPTION_CUE_TAIL_HOLD_SEC
    # A phrase also closes at punctuation, exactly the server's four marks.
    assert 'last == "." || last == "?" || last == "!" || last == ","' in source
    # Timed phrases exist only for the simple engine; hand-edited text falls back
    # to one caption for the whole clip, as on the server.
    assert 'static let timedEngine = "simple"' in source
    assert "collapseWhitespace(captionText(clip)) == collapseWhitespace(spokenText(clip))" in source


# ---------------------------------------------------------------------------
# The nine typefaces: same list, same sizes, the server's own files.
# ---------------------------------------------------------------------------

def test_the_app_offers_exactly_the_servers_nine_fonts():
    fonts = _swift_fonts()
    assert set(fonts) == set(caption_render.CAPTION_FONTS)
    assert len(fonts) == 9
    assert f'static let defaultKey = "{caption_render.DEFAULT_CAPTION_FONT}"' in RULES.read_text()


def test_font_sizes_and_widths_equal_the_servers():
    for key, spec in _swift_fonts().items():
        _family, size, _bold = caption_render.CAPTION_FONTS[key]
        assert float(spec["size"]) == size, key
        assert float(spec["cw"]) == caption_render._CHAR_WIDTH[key], key


def test_font_files_are_the_servers_and_are_bundled_into_the_app():
    for key, spec in _swift_fonts().items():
        assert (caption_render.FONTS_DIR / f'{spec["file"]}.{spec["ext"]}').is_file(), key
    project = PROJECT.read_text()
    assert "- path: ../../cutsell_worker/fonts" in project
    assert "buildPhase: resources" in project
    # Loaded at start from the bundle, from the one catalog.
    overlay = OVERLAY.read_text()
    assert "CTFontManagerRegisterFontsForURL" in overlay
    assert "for spec in CaptionFontCatalog.all" in overlay
    assert "CaptionFontLoader.registerIfNeeded()" in PLAYBACK.read_text()


def test_font_metrics_are_the_ones_inside_the_files():
    ttlib = pytest.importorskip("fontTools.ttLib")
    for key, spec in _swift_fonts().items():
        font = ttlib.TTFont(str(caption_render.FONTS_DIR / f'{spec["file"]}.{spec["ext"]}'))
        assert font["name"].getDebugName(6) == spec["ps"], key
        assert font["head"].unitsPerEm == int(spec["upem"]), key
        assert font["hhea"].ascent == int(spec["asc"]), key
        assert -font["hhea"].descent == int(spec["desc"]), key
        assert font["hhea"].lineGap == int(spec["gap"]), key
        assert font["OS/2"].usWinAscent == int(spec["wasc"]), key
        assert font["OS/2"].usWinDescent == int(spec["wdesc"]), key


# ---------------------------------------------------------------------------
# Placement, size and colours.
# ---------------------------------------------------------------------------

def test_layout_numbers_equal_the_servers():
    source = RULES.read_text()
    assert _number(source, "frameWidth") == caption_render.PLAY_RES_X
    assert _number(source, "frameHeight") == caption_render.PLAY_RES_Y
    assert _number(source, "defaultX") == caption_render.DEFAULT_CAPTION_X
    assert _number(source, "defaultY") == caption_render.DEFAULT_CAPTION_Y
    assert _number(source, "defaultScale") == caption_render.DEFAULT_CAPTION_SCALE
    assert (_number(source, "minimumScale"), _number(source, "maximumScale")) == caption_render.CAPTION_SCALE_RANGE
    assert _number(source, "edgePad") == caption_render._EDGE_PAD
    assert _number(source, "minimumColumn") == caption_render._MIN_COLUMN


def test_placement_matches_the_server_for_a_grid_of_positions():
    """The Swift formula, evaluated here, must give the server's own placement."""
    source = RULES.read_text()
    edge, column_min = _number(source, "edgePad"), _number(source, "minimumColumn")
    width, height = _number(source, "frameWidth"), _number(source, "frameHeight")
    assert "let halfMinimum = (minimumColumn / 2).rounded(.down) + edgePad" in source
    assert "let halfColumn = min(centerX, frameWidth - centerX) - edgePad" in source
    assert "let halfHeight = (size * 1.3).rounded(.down) + edgePad" in source
    for key, spec in _swift_fonts().items():
        for x in (0.0, 0.2, 0.5, 0.83, 1.0):
            for y in (0.0, 0.3, 0.758, 1.0):
                for scale in (0.5, 1.0, 1.6, 2.0):
                    size = max(1, round(float(spec["size"]) * scale))
                    half_min = column_min // 2 + edge
                    cx = round(min(max(x * width, half_min), width - half_min))
                    half_column = min(cx, width - cx) - edge
                    margin = max(0, (width - 2 * half_column) // 2)
                    half_height = int(size * 1.3) + edge
                    cy = round(min(max(y * height, half_height), height - half_height))
                    assert (cx, cy, margin, size) == caption_render._placement(x, y, scale, key)
                    # Never outside the frame.
                    assert edge <= cx - half_column and cx + half_column <= width - edge
                    assert 0 <= cy - half_height + edge and cy + half_height - edge <= height


def test_colours_and_looks_equal_the_servers():
    source = RULES.read_text()
    assert int(_number(source, "white")) == int(caption_render.WHITE, 16)
    assert int(_number(source, "ink")) == int(caption_render.INK, 16)
    assert int(_number(source, "yellow")) == int(caption_render.YELLOW, 16)
    for name in ("green", "red", "blue"):
        assert int(_number(source, f"highlight{name.capitalize()}")) == int(caption_render.HIGHLIGHT_COLOURS[name], 16)
    # Every preset the server accepts is drawn on purpose ("classic" is the default branch).
    for preset in caption_render.CAPTION_PRESETS - {"classic"}:
        assert f'"{preset}"' in source, preset


# ---------------------------------------------------------------------------
# Wiring: drawn over the video, from the draft the server returned.
# ---------------------------------------------------------------------------

def test_overlay_is_drawn_over_the_preview_only_when_captions_are_on():
    source = PLAYBACK.read_text()
    assert "if model.captionsEnabled {" in source
    assert "CaptionOverlayView(" in source
    for argument in ("preset: model.captionPreset", "fontKey: model.captionFont",
                     "x: model.captionX", "y: model.captionY", "scale: model.captionScale"):
        assert argument in source


def test_preview_reads_caption_settings_from_the_real_draft():
    source = VIEW_MODEL.read_text()
    for key in ("caption_font", "caption_x", "caption_y", "caption_scale"):
        assert f'captionValue("{key}")' in source
    # A choice just made shows at once; otherwise the draft decides.
    assert "captionOverrides[key] ?? snapshot?.draft[key]" in source
    assert 'snapshot?.draft["diagnostics"]?["engine"]?.stringValue' in source


def test_preview_rules_never_touch_the_network_or_the_draft():
    for path in (RULES, OVERLAY):
        source = path.read_text()
        assert "APIClient" not in source
        assert "api.request(" not in source
        assert "URLSession" not in source
