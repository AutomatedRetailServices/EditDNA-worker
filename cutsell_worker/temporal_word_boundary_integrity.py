"""Speech-safe temporal boundary policy for end trims.

A visual/performance event can begin during the final spoken word. The existing end snap
moved backward to the previous word boundary, which can delete a valid final word even
though the rule is only meant to remove post-delivery reset material. For an end trim,
if the raw boundary falls inside a spoken word, the safe direction is forward to that
word's end. This preserves the word and still avoids a mid-syllable cut.
"""
from __future__ import annotations


def _preserve_current_word_end(words, start: float, candidate: float) -> tuple[float, bool]:
    ordered = tuple(sorted(words, key=lambda word: (float(word.start), float(word.end))))
    for word in ordered:
        word_start = float(word.start)
        word_end = float(word.end)
        if word_start < candidate < word_end and word_end > start:
            return word_end, True
    return candidate, False


def install_temporal_word_boundary_integrity() -> None:
    from . import temporal_editing

    original = temporal_editing._snapped_end
    if getattr(original, "_cutsell_preserve_current_word_end", False):
        return

    def snapped_end_preserving_spoken_word(words, start, candidate):
        boundary, snapped = _preserve_current_word_end(words, start, candidate)
        if snapped:
            return boundary, True
        return original(words, start, candidate)

    snapped_end_preserving_spoken_word._cutsell_preserve_current_word_end = True
    temporal_editing._snapped_end = snapped_end_preserving_spoken_word
