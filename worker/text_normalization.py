"""Text normalization primitives for the English/Spanish V1 pipeline.

This module deliberately does not infer script families or tokenize unsegmented
writing systems.  It preserves Latin letters (including Spanish diacritics),
numbers, product names, and common typographic punctuation.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import List


APOSTROPHE_TRANSLATION = str.maketrans({
    "’": "'", "‘": "'", "‛": "'", "ʼ": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
})


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    text: str
    tokens: tuple[str, ...]
    compact: str


@dataclass(frozen=True)
class SemanticContentMeasure:
    normalized_tokens: tuple[str, ...]
    token_count: int
    whitespace_token_count: int
    alphanumeric_count: int
    measurement_strategy: str = "space_delimited"
    effective_semantic_units: int = 0
    scoring_rule: str = "space_delimited"
    repetition_noise_adjusted: bool = False


def normalize_match_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").translate(APOSTROPHE_TRANSLATION)
    return re.sub(r"\s+", " ", value.casefold()).strip()


def unicode_word_tokens(text: str) -> tuple[str, ...]:
    """Tokenize Latin-script speech while retaining accents and apostrophes."""
    tokens: List[str] = []
    current: List[str] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        is_latin = category[0] == "N" or (
            category[0] in {"L", "M"} and
            (category[0] == "M" or "LATIN" in unicodedata.name(char, ""))
        )
        if is_latin:
            current.append(char)
            continue
        next_word = index + 1 < len(text) and (
            unicodedata.category(text[index + 1])[0] == "N" or
            "LATIN" in unicodedata.name(text[index + 1], "")
        )
        if char == "'" and current and next_word:
            current.append(char)
        elif current:
            tokens.append("".join(current)); current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def normalized_text(text: str) -> NormalizedText:
    normalized = normalize_match_text(text)
    tokens = unicode_word_tokens(normalized)
    return NormalizedText(text or "", normalized, tokens, " ".join(tokens))


def semantic_content_measure(text: str) -> SemanticContentMeasure:
    n = normalized_text(text)
    repeated = max((len(list(group)) for _, group in __import__("itertools").groupby(n.tokens)), default=0)
    units = len(n.tokens)
    if repeated >= 4:
        units = min(units, max(2, units - repeated + 2))
    return SemanticContentMeasure(
        n.tokens, len(n.tokens), len(n.tokens),
        sum(char.isalnum() for char in n.text),
        effective_semantic_units=units,
        repetition_noise_adjusted=repeated >= 4,
    )


def semantic_content_score(text: str) -> float:
    units = semantic_content_measure(text).effective_semantic_units
    return 0.0 if not units else min(0.95, 0.4 + 0.03 * units)


def comparison_units(text: str) -> tuple[str, ...]:
    return normalized_text(text).tokens
