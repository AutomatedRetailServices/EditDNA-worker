"""Text normalization primitives for the English/Spanish V1 pipeline.

This module deliberately stays Latin-focused. It preserves Spanish diacritics,
numbers, product names, and common typographic punctuation without broad script
family detection or universal tokenization.
"""
from dataclasses import dataclass
import re
import unicodedata
from typing import List, Tuple

APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"'})
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿÑñ]+(?:'[0-9A-Za-zÀ-ÖØ-öø-ÿÑñ]+)?", re.UNICODE)


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    text: str
    compact: str
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class SemanticContentMeasure:
    normalized_tokens: Tuple[str, ...]
    token_count: int
    whitespace_token_count: int
    alphanumeric_count: int
    measurement_strategy: str
    effective_semantic_units: int
    scoring_rule: str
    repetition_noise_adjusted: bool


def normalize_match_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or "")).translate(APOSTROPHE_TRANSLATION)
    return " ".join(value.casefold().strip().split())


def unicode_word_tokens(text: str) -> List[str]:
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(unicodedata.normalize("NFC", str(text or "")))]


def normalized_text(text: str) -> NormalizedText:
    raw = str(text or "")
    norm = unicodedata.normalize("NFC", raw).translate(APOSTROPHE_TRANSLATION)
    compact = " ".join(norm.casefold().strip().split())
    return NormalizedText(raw=raw, text=norm, compact=compact, tokens=tuple(unicode_word_tokens(norm)))


def comparison_units(text: str) -> List[str]:
    return unicode_word_tokens(text)


def semantic_content_measure(text: str) -> SemanticContentMeasure:
    tokens = tuple(unicode_word_tokens(text))
    whitespace = len(str(text or "").split())
    alnum = sum(ch.isalnum() for ch in str(text or ""))
    unique = len(set(tokens))
    repeated = bool(tokens) and unique < max(1, len(tokens) // 2)
    effective = unique if repeated else len(tokens)
    return SemanticContentMeasure(
        normalized_tokens=tokens,
        token_count=len(tokens),
        whitespace_token_count=whitespace,
        alphanumeric_count=alnum,
        measurement_strategy="latin_word_tokens",
        effective_semantic_units=effective,
        scoring_rule="token_count_capped",
        repetition_noise_adjusted=repeated,
    )


def semantic_content_score(text: str) -> float:
    units = semantic_content_measure(text).effective_semantic_units
    if units <= 0:
        return 0.0
    return min(0.95, 0.4 + 0.03 * units)
