"""Small conservative English/Spanish fallback for V1.

This module is deliberately not the commercial brain. Semantic V2 owns commercial
meaning when available. These rules only provide safe production-meta detection,
legacy-compatible slot hints, and fail-open behavior for provider fallback.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

from worker.text_normalization import NormalizedText, normalized_text


def _normalized_words(text: str) -> List[str]:
    return list(normalized_text(text).tokens)


def has_phrase(n: NormalizedText, phrase: str) -> bool:
    p = normalized_text(phrase).compact
    return bool(p) and f" {p} " in f" {n.compact} "


def starts_phrase(n: NormalizedText, phrase: str) -> bool:
    p = normalized_text(phrase).compact
    return bool(p) and (n.compact == p or n.compact.startswith(p + " "))


def has_any_phrase(n: NormalizedText, phrases: Sequence[str]) -> bool:
    return any(has_phrase(n, p) for p in phrases)


_PRODUCTION_EXACT = {
    "start again", "start over", "restart", "redo", "redo that", "try again",
    "do that again", "one more time", "cut that", "camera rolling", "rolling",
    "empieza de nuevo", "empecemos de nuevo", "reinicia", "repite", "otra vez",
    "hazlo otra vez", "una vez más", "corta", "corten", "cámara grabando",
    "estamos grabando",
}
_PRODUCTION_PREFIXES = (
    "let me start again", "let me redo", "wait let me", "hold on let me",
    "no restart", "no redo", "espera déjame", "déjame repetir",
)
_STANDALONE_FILLER = {
    "uh", "um", "hmm", "is that good", "is that funny", "am i saying it right",
    "why can't i remember", "that one good",
}


def is_camera_rolling_slate(text: str) -> bool:
    n = normalized_text(text)
    return n.compact in {"camera rolling", "camera is rolling", "rolling", "cámara grabando", "estamos grabando"}


def _strip_token_prefixes(tokens: tuple[str, ...], prefixes: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    for prefix in prefixes:
        if tokens[:len(prefix)] == prefix:
            return tokens[len(prefix):]
    return tokens


def is_clear_recording_direction(n: NormalizedText) -> bool:
    if not n.compact:
        return False
    return n.compact in _PRODUCTION_EXACT or any(starts_phrase(n, p) for p in _PRODUCTION_PREFIXES)


def is_compound_take_slate(n: NormalizedText) -> bool:
    tokens = n.tokens
    if not tokens:
        return False
    if tokens[0] in {"okay", "so"}:
        tokens = tokens[1:]
    if len(tokens) < 2 or tokens[0] not in {"take", "toma"}:
        return False
    return tokens[1] in {"2", "3", "two", "three", "dos", "tres"}


def production_meta_rule(text: str, include_camera_rolling: bool = True) -> Optional[str]:
    n = normalized_text(text)
    if not n.compact:
        return None
    if is_clear_recording_direction(n) or is_compound_take_slate(n):
        return "production_meta_phrase"
    if include_camera_rolling and is_camera_rolling_slate(text):
        return "production_meta_phrase"
    return None


def filler_rule(text: str) -> Optional[str]:
    n = normalized_text(text)
    if not n.compact:
        return "empty"
    if production_meta_rule(text):
        return "production_meta_phrase"
    if n.compact in _STANDALONE_FILLER:
        return "standalone_meta_token"
    return None


def looks_like_filler(text: str) -> bool:
    return filler_rule(text) is not None


def looks_like_dependent_tail(text: str) -> bool:
    n = normalized_text(text)
    if not n.tokens or len(n.tokens) > 4:
        return False
    return n.tokens[0] in {"and", "so", "but", "because", "y", "pero", "porque"} or n.compact.endswith((" as well", " too", " either"))


@dataclass(frozen=True)
class CTAActionFrame:
    action: str = ""
    target: str = ""
    explicit: bool = False


def _action_target(text: str) -> str:
    return normalized_text(text).compact


def _starts_explicit_cta(text: str) -> bool:
    n = normalized_text(text)
    return any(starts_phrase(n, p) for p in ("shop now", "get yours", "add to cart", "order now", "tap the link", "click the link", "compra ahora", "ordena ahora", "toca el enlace"))


def _cta_clauses(text: str) -> List[str]:
    return [text]


def _strip_cta_discourse(text: str) -> str:
    return text.strip()


def _is_reported_or_narrated_action(text: str) -> bool:
    return False


def cta_action_frames(text: str) -> List[CTAActionFrame]:
    return [CTAActionFrame(action="cta", target=_action_target(text), explicit=True)] if _starts_explicit_cta(text) else []


def is_explicit_link_cta(text: str) -> bool:
    n = normalized_text(text)
    return has_any_phrase(n, ("link below", "link in bio", "click the link", "tap the link", "enlace abajo", "enlace en mi bio"))


def cta_action_rule(text: str) -> Optional[str]:
    if _starts_explicit_cta(text) or is_explicit_link_cta(text):
        return "explicit_viewer_cta"
    return None


@dataclass(frozen=True)
class CommandIntent:
    production_rule: Optional[str]
    cta_rule: Optional[str]
    final_intent: str


def command_intent(text: str, include_camera_rolling: bool = True) -> CommandIntent:
    production = production_meta_rule(text, include_camera_rolling=include_camera_rolling)
    cta = cta_action_rule(text)
    if production:
        return CommandIntent(production, cta, "production")
    if cta:
        return CommandIntent(None, cta, "cta")
    return CommandIntent(None, None, "none")


def has_cta_action_context(text: str) -> bool:
    return cta_action_rule(text) is not None


def has_product_enumeration_evidence(text: str) -> bool:
    n = normalized_text(text)
    return any(token in n.tokens for token in ("comes", "includes", "contains", "features", "incluye", "contiene"))


def classify_slot_rule(text: str) -> tuple[str, str]:
    """Conservative fallback hint only; uncertain speech remains OTHER and keepable."""
    n = normalized_text(text)
    if not n.compact:
        return "OTHER", "empty"
    if cta_action_rule(text):
        return "CTA", "explicit_cta"
    if n.compact.startswith(("stop scrolling", "have you", "are you", "if you", "si tú", "si tu")):
        return "HOOK", "obvious_hook"
    if has_product_enumeration_evidence(text):
        return "FEATURES", "obvious_product_enumeration"
    return "OTHER", "unclassified_product_context"


def classify_slot(text: str) -> str:
    return classify_slot_rule(text)[0]
