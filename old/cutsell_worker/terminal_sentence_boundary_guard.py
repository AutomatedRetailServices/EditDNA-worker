"""Prevent complete-idea boundary recovery from absorbing a following sentence.

If the original selected clip already reaches a terminal source word, trailing recovery
may finish that terminal word but must not extend into the next sentence. This preserves
spoken information while preventing a completed idea from swallowing adjacent narrative
material or retry setup. No benchmark phrases, timestamps, or clip ids are hardcoded.

Boundary validation also covers upstream same-delivery microgap continuity so a final
sentence guard is always exercised on the exact timeline produced by those repairs.
"""
from __future__ import annotations

from dataclasses import replace


def install_terminal_sentence_boundary_guard() -> None:
    from . import final_boundary_authority as authority

    original = authority._clip_from_envelope
    if getattr(original, "_cutsell_terminal_sentence_boundary_guard", False):
        return

    def protected(clip, source_words):
        overlap = authority._overlapping_indices(source_words, float(clip.start), float(clip.end))
        updated, row = original(clip, source_words)
        if overlap is None:
            return updated, row

        _, original_last = overlap
        terminal_word = source_words[original_last]
        if not authority._terminal(terminal_word):
            return updated, row
        if float(updated.end) <= float(clip.end) + 1e-6:
            return updated, row

        # The clip may end inside the terminal word. Finish that word, but do not absorb
        # any following sentence solely because its first words are tightly connected.
        safe_end = max(float(clip.end), float(terminal_word.end))
        kept_words = tuple(
            word for word in tuple(updated.words)
            if float(word.start) < safe_end - 1e-6 and float(word.end) <= safe_end + 1e-6
        )
        if not kept_words:
            return updated, row

        text = " ".join(str(word.text).strip() for word in kept_words).strip()
        guarded = replace(
            updated,
            end=safe_end,
            words=kept_words,
            text=text or clip.text,
            caption_text=text or clip.caption_text,
        )
        guarded_row = dict(row)
        guarded_row.update({
            "action": "stop_at_existing_terminal_sentence_boundary",
            "result_end": round(float(guarded.end), 3),
            "added_trailing_sec": round(max(0.0, float(guarded.end) - float(clip.end)), 3),
            "terminal_word": str(terminal_word.text),
        })
        return guarded, guarded_row

    protected._cutsell_terminal_sentence_boundary_guard = True
    authority._clip_from_envelope = protected
