"""Fonts / Style sheet (PO 30 Oct 2026): the caption look and each text's look are stored with the draft."""
import pytest

from cutsell_worker.serde import draft_from_dict
from cutsell_worker.text_style import validate_text_style


def _draft(**extra):
    base = {"project_id": "p1", "selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0,
                                              "text": "hola", "selected": True}]}
    base.update(extra)
    return base


def test_a_full_style_is_accepted_and_normalised():
    style = {"preset": "outline", "color": "#ff2d55", "opacity": 0.9, "stroke_color": "#000000", "stroke_width": 4,
             "bg_color": "#FFFFFF", "bg_opacity": 0.0, "bg_radius": 8, "shadow_color": "#000000", "shadow_opacity": 0.6,
             "shadow_blur": 4, "shadow_distance": 3, "letter_spacing": 2, "line_spacing": 6, "align": "center",
             "uppercase": True, "highlight_color": "#39FF78"}
    out = validate_text_style(style)
    assert out["color"] == "#FF2D55" and out["stroke_width"] == 4.0 and out["uppercase"] is True
    draft_from_dict(_draft(caption_style=style, caption_font="pacifico"))


@pytest.mark.parametrize("bad", [{"color": "red"}, {"opacity": 2}, {"stroke_width": -1}, {"align": "justify"},
                                 {"unknown": 1}, {"opacity": True}, "not an object"])
def test_junk_is_refused(bad):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(caption_style=bad))


def test_a_text_overlay_keeps_its_own_font_and_style():
    text = {"overlay_id": "t1", "text": "SALE", "start": 0.0, "end": 2.0, "font": "bungee",
            "style": {"preset": "dark_box", "bg_color": "#000000", "bg_opacity": 0.8}}
    draft_from_dict(_draft(text_overlays=[text]))
    with pytest.raises(ValueError):
        draft_from_dict(_draft(text_overlays=[dict(text, font="comic-sans")]))
    with pytest.raises(ValueError):
        draft_from_dict(_draft(text_overlays=[dict(text, style={"color": "#12"})]))


def test_drafts_without_a_style_are_unchanged():
    draft_from_dict(_draft())
