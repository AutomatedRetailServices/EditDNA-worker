"""Focused English/Spanish V1 regressions consolidated from PR #21."""
import importlib.util
import sys
import types

import pytest

for name, attrs in (("requests", {}), ("boto3", {}), ("clip", {}), ("faster_whisper", {"WhisperModel": object})):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

from worker import pipeline
from worker.diagnostics import sanitize_source_identifier
from worker.text_normalization import normalized_text


def clip(cid, start, end, text, *, source=0, words=None):
    item = pipeline.make_base_clip(cid, start, end, text, words=words)
    item.update(source_index=source, source_local=f"take-{source}.mp4", source_start=start, source_end=end)
    item["meta"].update(source_index=source, source_local=f"take-{source}.mp4")
    return item


@pytest.mark.parametrize("text,keep", [
    ("But wait, there's more.", True), ("This cream is, um, lightweight.", True),
    ("Wait, let me do that again.", False), ("Thanks", False),
    ("Thank you for choosing our moisturizer.", True),
])
def test_filler_vs_valid_narration(text, keep):
    item = clip("x", 0, 1, text); pipeline.tag_clips_heuristic([item])
    assert item["meta"]["keep"] is keep


@pytest.mark.parametrize("text,is_meta", [
    ("Camera rolling", True), ("Take two, start over", True),
    ("Cámara grabando", True), ("Toma dos", True),
    ("Take two capsules daily with water.", False),
    ("Toma dos cápsulas al día.", False), ("Rolling this serum onto your skin feels cool.", False),
])
def test_restart_slate_vs_dosage(text, is_meta):
    assert (pipeline.production_meta_rule(text) is not None) is is_meta


@pytest.mark.parametrize("text,expected", [
    ("Buy now", "CTA"), ("You can shop the collection now", "CTA"),
    ("I went to the shop yesterday", "OTHER"), ("The buyer loved it", "OTHER"),
    ("Compra ahora", "CTA"), ("Haz clic en el enlace", "CTA"),
])
def test_cta_action_context_and_whole_tokens(text, expected):
    assert pipeline.classify_slot(text) == expected


@pytest.mark.parametrize("text", [
    "Get it now", "Grab this set", "Order one today", "Pick one from the collection",
    "Compra uno ahora", "Pide este set", "Consíguelo en el enlace", "Llévate uno hoy",
])
def test_imperative_singular_product_objects_are_cta(text):
    assert pipeline.classify_slot(text) == "CTA"


@pytest.mark.parametrize("text", [
    "I grabbed one yesterday", "I ordered one for my sister",
    "Pick one example from the list", "Get one point for each answer",
    "Get it right", "Get this done", "Grab it carefully",
    "Compré uno ayer", "Pedí uno para mi hermana", "Elige un ejemplo", "Toma uno de la lista",
])
def test_singular_object_words_without_commercial_imperative_are_not_cta(text):
    assert pipeline.classify_slot(text) != "CTA"


@pytest.mark.parametrize("text", [
    "start again", "redo", "do it again", "try again", "restart", "start over",
    "one more time", "let's redo that", "empieza de nuevo", "hazlo otra vez",
    "repite", "vamos otra vez", "una vez más", "reinicia", "volvamos a empezar",
])
def test_standalone_restart_commands_are_production(text):
    assert pipeline.command_intent(text).final_intent == "production"
    item = clip("restart", 0, 1, text)
    pipeline.tag_clips_heuristic([item])
    assert item["meta"]["keep"] is False
    assert pipeline.build_composer([item])["used_clip_ids"] == []


@pytest.mark.parametrize("text", [
    "Camera rolling, start again", "Recording this take, let's redo that",
    "Cámara grabando, hazlo otra vez", "Estamos grabando, volvamos a empezar",
])
def test_recording_context_restart_commands_are_production(text):
    assert pipeline.command_intent(text).final_intent == "production"


@pytest.mark.parametrize("text", [
    "Start over with clearer skin", "Try again tomorrow", "Reinicia tu rutina",
    "Hazlo otra vez cuando se seque",
])
def test_restart_like_narration_is_not_production(text):
    assert pipeline.command_intent(text).final_intent != "production"
    assert pipeline.looks_like_filler(text) is False


@pytest.mark.parametrize("text", [
    "Get it right", "Get this done", "Grab it carefully", "Pick one example from the list",
    "Get one point for every answer", "Haz esto primero", "Elige un ejemplo", "Toma uno de la lista",
])
def test_noncommercial_commands_default_safely(text):
    assert pipeline.command_intent(text).final_intent in {"noncommercial_command", "ambiguous", "narration"}
    assert pipeline.classify_slot(text) == "OTHER"


@pytest.mark.parametrize("text,expected", [
    ("Honestly, it helps you feel confident", "BENEFITS"),
    ("In my experience, it reduced dryness", "BENEFITS"),
    ("Honestamente, te ayuda a sentirte segura", "BENEFITS"),
    ("En mi experiencia, redujo la sequedad", "BENEFITS"),
    ("I've tried it and it comes with two brushes", "FEATURES"),
    ("He probado el producto y viene con dos brochas", "FEATURES"),
    ("I used it for a week and five customers confirmed the result", "PROOF"),
    ("Lo usé una semana y cinco clientes confirmaron el resultado", "PROOF"),
    ("Honestly, I discovered it while traveling", "STORY"),
    ("Déjame contarte cómo lo descubrí durante un viaje", "STORY"),
])
def test_personal_framing_yields_to_stronger_commercial_evidence(text, expected):
    assert pipeline.classify_slot(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("It's a problem keeping my skin hydrated.", "PROBLEM"),
    ("This is a product that helps you feel confident.", "BENEFITS"),
    ("It's a five stars measurable result.", "PROOF"),
    ("Es un problema mantener mi piel hidratada.", "PROBLEM"),
])
def test_explicit_evidence_precedes_generic_prefix(text, expected):
    assert pipeline.classify_slot(text) == expected


@pytest.mark.parametrize("text", [
    "You get red, blue, and green shades.", "Viene con tres tonos diferentes.",
    "The set includes two glosses and three shades.",
])
def test_feature_enumeration(text):
    assert pipeline.classify_slot(text) == "FEATURES"


def test_boundary_repair_and_microfragment_rejection():
    repaired = clip("repair", 1, 1, "Two useful words", words=[
        {"start": 1.0, "end": 1.2, "word": "Two "}, {"start": 1.2, "end": 1.5, "word": "words"},
    ])
    repaired["words"] = [
        {"start": 1.0, "end": 1.2, "word": "Two "}, {"start": 1.2, "end": 1.5, "word": "words"},
    ]
    bad = clip("bad", 2, 2.01, "Impossible fragment")
    diagnostics = []
    result = pipeline.validate_clip_boundaries([repaired, bad], discarded_diagnostics=diagnostics)
    assert [(x["start"], x["end"]) for x in result] == [(1.0, 1.5)]
    assert repaired["source_start"] == 1.0 and repaired["source_end"] == 1.5
    assert diagnostics[0]["reason"] == "discarded_invalid_microfragment"


def test_duplicate_suppression_is_adjacent_and_source_aware():
    a = clip("a", 0, 1, "Buy now.")
    residual = clip("b", 1.05, 1.15, "Buy now.")
    later = clip("c", 5, 6, "Buy now.")
    other_source = clip("d", 6, 7, "Buy now.", source=1)
    diagnostics = []
    result = pipeline.validate_clip_boundaries([a, residual, later, other_source], discarded_diagnostics=diagnostics)
    assert [x["id"] for x in result] == ["a", "c", "d"]
    assert diagnostics[0]["reason"] == "discarded_duplicate_residual_text"


def test_valid_incomplete_merge_but_never_cross_take():
    first = clip("a", 0, 1, "This formula is designed for")
    second = clip("b", 1, 2, "people with dry skin.")
    assert len(pipeline.merge_incomplete_phrases([first, second])) == 1
    # Cross-take clips are validated independently and retained.
    same = [clip("c", 3, 4, "A complete line.", source=0), clip("d", 4, 5, "A complete line.", source=1)]
    assert len(pipeline.validate_clip_boundaries(same)) == 2


def prepared(items):
    pipeline.tag_clips_heuristic(items)
    for item in items:
        item["semantic_score"] = item["score"] = .9
        item["meta"]["semantic_score"] = item["meta"]["score"] = .9
    return items


@pytest.mark.parametrize("texts", [
    ["I use this every morning.", "It helps you feel fresh.", "Buy now."],
    ["It comes in three shades.", "I get so many compliments.", "It includes a brush."],
    ["I get so many compliments.", "Buy now."],
])
def test_flexible_funnel_preserves_order_repeated_and_missing_slots(texts):
    items = prepared([clip(str(i), i * 2, i * 2 + 1, text) for i, text in enumerate(texts)])
    composer = pipeline.build_composer(items)
    selected = composer["used_clip_ids"]
    assert selected == sorted(selected, key=lambda cid: int(cid))


def test_composer_keeps_only_selected_cta_without_reordering():
    items = prepared([clip("0", 0, 1, "Buy now."), clip("1", 2, 3, "It helps you feel fresh."), clip("2", 4, 5, "Click the link below.")])
    items[0]["semantic_score"] = items[0]["score"] = .6
    items[0]["meta"]["semantic_score"] = items[0]["meta"]["score"] = .6
    result = pipeline.build_composer(items)["used_clip_ids"]
    assert "0" not in result and result == sorted(result, key=int)


@pytest.mark.parametrize("text", ["Buy now.", "Compra ahora.", "Suave y ligero."])
def test_short_valid_transcript_has_safe_composer_fallback(text):
    item = clip("short", 0, 1, text)
    pipeline.tag_clips_heuristic([item])
    item["visual_score"] = item["meta"]["visual_score"] = .8
    assert pipeline.build_composer([item])["used_clip_ids"] == ["short"]
    assert item["meta"]["composer_fallback"] == "keepable_short_transcript"


def test_short_transcript_fallback_preserves_hard_rejections():
    poor_visual = clip("poor", 0, 1, "Buy now.")
    poor_visual["visual_score"] = poor_visual["meta"]["visual_score"] = .2
    rejected = clip("rejected", 2, 3, "Compra ahora.")
    rejected["meta"]["keep"] = False
    slate = clip("slate", 4, 5, "Camera rolling")
    pipeline.tag_clips_heuristic([slate])
    assert pipeline.build_composer([poor_visual, rejected, slate])["used_clip_ids"] == []


def test_threshold_fallback_keeps_selected_cta_and_source_order():
    items = [clip("0", 0, 1, "Buy now."), clip("1", 2, 3, "Soft and light."), clip("2", 4, 5, "Click the link below.")]
    for item in items:
        pipeline.tag_clips_heuristic([item])
        item["visual_score"] = item["meta"]["visual_score"] = .8
    items[0]["semantic_score"] = items[0]["meta"]["semantic_score"] = .1
    items[2]["semantic_score"] = items[2]["meta"]["semantic_score"] = .2
    assert pipeline.build_composer(items)["used_clip_ids"] == ["1", "2"]


@pytest.mark.parametrize("left,right", [("don't", "don’t"), ("I'M", "I’m"), ("NIÑA", "niña")])
def test_english_spanish_unicode_normalization(left, right):
    assert normalized_text(left).tokens == normalized_text(right).tokens


@pytest.mark.parametrize("source,expected", [
    ("/tmp/private/session/video.mp4", "video.mp4"),
    ("s3://secret-bucket/uploads/customer/video.mp4?token=x", "uploads/customer/video.mp4"),
    ("../private/video.mp4", "video.mp4"),
])
def test_diagnostic_source_sanitation(source, expected):
    assert sanitize_source_identifier(source) == expected


# The original five benchmark examples from PR #21.
def test_original_five_benchmark_examples():
    examples = {
        "I found the perfect gift for our lip gloss girlies.": "HOOK",
        "You get six lip glosses, but wait, there's so much more to these.": "FEATURES",
        "stocking, the Santa hat, a Christmas tree, and let's not forget, a snowman.": "FEATURES",
        "These are so cute, they are all lip glosses.": "BENEFITS",
        "So if you know anyone who loves lip glosses or you yourself want them, grab some up. You can grab them up in a set of one, two, or three.": "CTA",
    }
    assert {text: pipeline.classify_slot(text) for text in examples} == examples
