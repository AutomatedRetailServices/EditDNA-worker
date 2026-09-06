"""Absolute speech guard for recording-process dead-air cuts.

Human boundary polish may identify convincing acoustic/visual recording-process gaps,
but no such interval is allowed to remove source-aligned spoken words. This guard keeps
word envelopes as physical authority and fails open whenever a proposed dead-air cut
intersects speech. It contains no benchmark-specific timestamps, phrases, or clip IDs.
"""
from __future__ import annotations


def _intersects_word(clip, cut_start: float, cut_end: float) -> bool:
    """Return True when any aligned word overlaps the proposed removed interval."""
    start = float(cut_start)
    end = float(cut_end)
    if end <= start:
        return True
    for word in tuple(getattr(clip, "words", ()) or ()):
        word_start = float(getattr(word, "start", 0.0))
        word_end = float(getattr(word, "end", word_start))
        # Require a real temporal overlap, while tolerating tiny floating-point noise.
        if word_end > start + 1e-4 and word_start < end - 1e-4:
            return True
    return False


def install_speech_safe_dead_air_guard() -> None:
    from . import human_boundary_polish_v3 as polish_v3

    original = polish_v3._remove_interval
    if getattr(original, "_cutsell_speech_safe_dead_air_guard", False):
        return

    def protected(clip, cut_start: float, cut_end: float):
        # Acoustic silence and visual reset evidence can be wrong around quiet words,
        # breathy endings, or ASR timing edges. Spoken-word envelopes are absolute:
        # if a proposed dead-air removal intersects speech, keep the source unchanged.
        if _intersects_word(clip, cut_start, cut_end):
            return (clip,)
        return original(clip, cut_start, cut_end)

    protected._cutsell_speech_safe_dead_air_guard = True
    polish_v3._remove_interval = protected
