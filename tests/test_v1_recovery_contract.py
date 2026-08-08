from worker.commercial_fallback import classify_slot_rule, filler_rule
from worker.text_normalization import normalized_text, semantic_content_measure


def test_uncertain_sales_copy_fails_open_instead_of_becoming_filler():
    slot, rule = classify_slot_rule("This serum feels amazing on my skin")
    assert slot == "OTHER"
    assert rule == "unclassified_product_context"
    assert filler_rule("This serum feels amazing on my skin") is None


def test_wait_inside_sales_copy_is_not_blanket_filler():
    assert filler_rule("But wait, there's more") is None


def test_explicit_restart_is_production_meta():
    assert filler_rule("Start again") == "production_meta_phrase"
    assert filler_rule("Empieza de nuevo") == "production_meta_phrase"


def test_obvious_cta_can_be_hinted_without_phrase_brain():
    slot, rule = classify_slot_rule("Tap the link")
    assert (slot, rule) == ("CTA", "explicit_cta")


def test_latin_spanish_normalization_preserves_semantic_units():
    n = normalized_text("¡Piel más suave, mañana!")
    assert "piel" in n.tokens
    assert "más" in n.tokens
    assert semantic_content_measure(n.text).effective_semantic_units >= 4


def test_spanish_v1_fallback_uses_broad_intent_cues():
    examples = {
        "¿Tienes la piel seca?": "HOOK",
        "Estoy cansada de productos que no funcionan": "PROBLEM",
        "Esto tiene una placa de cerámica": "FEATURES",
        "Te ayuda a ahorrar tiempo": "BENEFITS",
        "Recibo cumplidos todo el tiempo": "PROOF",
        "La primera vez que lo encontré estaba viajando": "STORY",
        "Añade al carrito": "CTA",
    }
    for text, expected in examples.items():
        assert classify_slot_rule(text)[0] == expected
