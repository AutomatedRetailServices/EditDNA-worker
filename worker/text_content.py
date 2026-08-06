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
    normalized_tokens: tuple[str, ...]
    token_count: int
    whitespace_token_count: int
    alphanumeric_count: int
    predominantly_unsegmented: bool
    measurement_strategy: str
    effective_semantic_units: int
    scoring_rule: str
    repetition_noise_adjusted: bool


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


UNSEGMENTED_SCRIPT_PREFIXES = (
    "CJK", "IDEOGRAPHIC", "HIRAGANA", "KATAKANA", "THAI", "LAO",
    "KHMER", "MYANMAR",
)


def _script_family(char: str) -> str:
    category = unicodedata.category(char)[0]
    if category == "N":
        return "NUMBER"
    if category != "L":
        return ""
    name = unicodedata.name(char, "")
    for family in (
        "LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI",
        *UNSEGMENTED_SCRIPT_PREFIXES,
    ):
        if family in name:
            return family
    return name.split(" ", 1)[0] if name else "LETTER"


def _is_unsegmented_char(char: str) -> bool:
    family = _script_family(char)
    return family.startswith(UNSEGMENTED_SCRIPT_PREFIXES)


def _mixed_effective_units(tokens: tuple[str, ...]) -> int:
    units = 0
    for token in tokens:
        run_type = None
        run_length = 0
        for char in token:
            if unicodedata.category(char)[0] not in {"L", "N"}:
                continue
            current_type = "unsegmented" if _is_unsegmented_char(char) else "segmented"
            if run_type is not None and current_type != run_type:
                units += math.ceil(run_length / 1.5) if run_type == "unsegmented" else 1
                run_length = 0
            run_type = current_type
            run_length += 1
        if run_length:
            units += math.ceil(run_length / 1.5) if run_type == "unsegmented" else 1
    return units


def _long_repetition_run(tokens: tuple[str, ...]) -> int:
    longest = current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def semantic_content_measure(text: str) -> SemanticContentMeasure:
    """Measure meaningful content without treating punctuation as speech."""
    normalized = normalized_text(text)
    alphanumeric_count = sum(
        unicodedata.category(char)[0] in {"L", "N"} for char in normalized.text
    )
    whitespace_token_count = sum(
        bool(unicode_word_tokens(chunk)) for chunk in normalized.text.split()
    )
    families = {
        _script_family(char) for char in normalized.text
        if _script_family(char)
    }
    has_unsegmented = any(
        family.startswith(UNSEGMENTED_SCRIPT_PREFIXES) for family in families
    )
    has_segmented = any(
        not family.startswith(UNSEGMENTED_SCRIPT_PREFIXES) and family != "NUMBER"
        for family in families
    )
    has_multiple_families = len(families) > 1
    repeated_symbol_noise = bool(re.search(r"([^\w\s])\1{3,}", normalized.text))
    repetition_run = _long_repetition_run(normalized.tokens)
    repetition_adjusted = repeated_symbol_noise or repetition_run >= 4
    if not normalized.tokens:
        strategy, effective_units = "empty/noise", 0
    elif has_unsegmented and has_segmented:
        strategy = "mixed"
        effective_units = min(20, _mixed_effective_units(normalized.tokens))
    elif has_unsegmented:
        strategy = "unsegmented"
        effective_units = min(20, math.ceil(alphanumeric_count / 1.5))
    elif has_multiple_families:
        strategy = "mixed"
        effective_units = len(normalized.tokens)
    else:
        strategy = "space_delimited"
        effective_units = len(normalized.tokens)
    if repetition_run >= 4:
        effective_units = min(effective_units, max(2, len(normalized.tokens) - repetition_run + 2))
    rule = strategy.replace("/", "_")
    return SemanticContentMeasure(
        normalized_tokens=normalized.tokens,
        token_count=len(normalized.tokens),
        whitespace_token_count=whitespace_token_count,
        alphanumeric_count=alphanumeric_count,
        predominantly_unsegmented=strategy == "unsegmented",
        measurement_strategy=strategy,
        effective_semantic_units=effective_units,
        scoring_rule=rule,
        repetition_noise_adjusted=repetition_adjusted,
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
    if measure.measurement_strategy == "unsegmented":
        return tuple(
            f"u:{char}" for char in normalized.text
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    if measure.measurement_strategy == "mixed" and any(
        _is_unsegmented_char(char) for char in normalized.text
    ):
        units: List[str] = []
        for token in normalized.tokens:
            segmented_run = ""
            for char in token:
                if unicodedata.category(char)[0] == "M":
                    continue
                if _is_unsegmented_char(char):
                    if segmented_run:
                        units.append(f"s:{segmented_run}")
                        segmented_run = ""
                    units.append(f"u:{char}")
                else:
                    segmented_run += char
            if segmented_run:
                units.append(f"s:{segmented_run}")
        return tuple(units)
    return normalized.tokens
