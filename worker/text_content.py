import math
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
    token_count: int
    alphanumeric_count: int
    predominantly_unsegmented: bool
    effective_semantic_units: int
    scoring_rule: str


def normalize_match_text(text: str) -> str:
    normalized = (text or "").translate(APOSTROPHE_TRANSLATION).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def unicode_word_tokens(text: str) -> tuple[str, ...]:
    """Preserve Unicode letters/numbers and marks plus internal apostrophes."""
    tokens: List[str] = []
    current: List[str] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(char)
            continue
        next_is_word = (
            index + 1 < len(text)
            and unicodedata.category(text[index + 1])[0] in {"L", "N"}
        )
        if char == "'" and current and next_is_word:
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def normalized_text(text: str) -> NormalizedText:
    normalized = normalize_match_text(text)
    tokens = unicode_word_tokens(normalized)
    return NormalizedText(text or "", normalized, tokens, " ".join(tokens))


def semantic_content_measure(text: str) -> SemanticContentMeasure:
    """Measure meaningful content without treating punctuation as speech."""
    normalized = normalized_text(text)
    letters = [char for char in normalized.text if unicodedata.category(char)[0] == "L"]
    alphanumeric_count = sum(
        unicodedata.category(char)[0] in {"L", "N"} for char in normalized.text
    )
    non_latin_letters = sum(
        "LATIN" not in unicodedata.name(char, "") for char in letters
    )
    non_latin_ratio = non_latin_letters / max(1, len(letters))
    predominantly_unsegmented = bool(
        alphanumeric_count >= 4
        and len(normalized.tokens) <= 2
        and non_latin_letters >= 3
        and non_latin_ratio >= 0.25
    )
    if not normalized.tokens:
        effective_units, rule = 0, "empty_or_symbols"
    elif predominantly_unsegmented:
        effective_units = min(20, alphanumeric_count)
        rule = "unsegmented_unicode_characters"
    elif non_latin_letters >= 3 and non_latin_ratio >= 0.15:
        effective_units = min(
            20, max(len(normalized.tokens), math.ceil(alphanumeric_count / 2))
        )
        rule = "unicode_tokens_with_character_support"
    else:
        effective_units, rule = len(normalized.tokens), "normalized_token_count"
    return SemanticContentMeasure(
        token_count=len(normalized.tokens),
        alphanumeric_count=alphanumeric_count,
        predominantly_unsegmented=predominantly_unsegmented,
        effective_semantic_units=effective_units,
        scoring_rule=rule,
    )


def semantic_content_score(text: str) -> float:
    measure = semantic_content_measure(text)
    if measure.effective_semantic_units <= 0:
        return 0.0
    return min(0.95, 0.4 + 0.03 * measure.effective_semantic_units)


def comparison_units(text: str) -> tuple[str, ...]:
    """Return stable units for overlap and length tie-breaking."""
    normalized = normalized_text(text)
    measure = semantic_content_measure(text)
    if measure.predominantly_unsegmented:
        return tuple(
            char for char in normalized.text
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    return normalized.tokens
