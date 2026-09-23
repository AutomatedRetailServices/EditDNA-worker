"""Speech-safe post-Best-Take boundary repair for retry-setup end trims.

The baseline temporal editor intentionally rewinds an end boundary to the previous word
when a generic harmful event starts mid-word. That contract is correct for ordinary bad
performance. Round 7 exposed a narrower case after Best Take: a ``retry_setup`` event can
begin during the final *valid* word of an already-complete selected delivery. Rewinding
there deletes the whole final word (for example ``ejercicio``) even though the reset is
post-delivery evidence.

Only when ``preserve_clip_id=True`` (the post-Best-Take pass), the take is already a
complete idea, and a retry_setup starts inside its final word do we restore that final
word boundary. All earlier/generic temporal behavior remains unchanged.
"""
from __future__ import annotations

from dataclasses import replace


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
    from . import pipeline

    original = temporal_editing.refine_takes_with_temporal_context
    if getattr(original, "_cutsell_post_best_take_final_word", False):
        return

    def refine_with_post_best_take_final_word(
        takes,
        context,
        *,
        edge_tolerance_sec=0.30,
        minimum_keep_sec=0.30,
        preserve_clip_id=False,
    ):
        take_tuple = tuple(takes)
        refined, diagnostics = original(
            take_tuple,
            context,
            edge_tolerance_sec=edge_tolerance_sec,
            minimum_keep_sec=minimum_keep_sec,
            preserve_clip_id=preserve_clip_id,
        )
        if not preserve_clip_id:
            return refined, diagnostics

        repaired = list(refined)
        repaired_diagnostics = [dict(item) for item in diagnostics]

        for index, (source_take, child, diag) in enumerate(zip(take_tuple, refined, repaired_diagnostics)):
            if not source_take.complete_idea or not source_take.words:
                continue
            trim_events = [
                item for item in (diag.get("applied") or ())
                if item.get("action") == "trim_end"
                and str(item.get("kind") or "").strip().lower().replace("-", "_").replace(" ", "_") == "retry_setup"
            ]
            if not trim_events:
                continue

            ordered_words = tuple(sorted(source_take.words, key=lambda word: (float(word.start), float(word.end))))
            last_word = ordered_words[-1]
            last_start = float(last_word.start)
            last_end = float(last_word.end)
            if last_end > float(source_take.end) + 0.08:
                continue

            event_start = min(float(item.get("start") or 0.0) for item in trim_events)
            if not (last_start < event_start < last_end):
                continue
            if float(child.end) >= last_end - 1e-6:
                continue

            words = temporal_editing._trim_words(source_take.words, float(child.start), last_end)
            text = " ".join(word.text for word in words).strip() if words else source_take.text
            repaired[index] = replace(
                child,
                end=last_end,
                text=text,
                words=words,
                signals=temporal_editing._signals_for_trim(child.signals, float(child.start), last_end),
            )
            diag["result_end"] = last_end
            diag.setdefault("word_boundary_snaps", []).append({
                "action": "preserve_final_word_after_retry_setup",
                "raw_boundary": event_start,
                "safe_boundary": last_end,
                "word": str(last_word.text or ""),
            })

        return tuple(repaired), tuple(repaired_diagnostics)

    refine_with_post_best_take_final_word._cutsell_post_best_take_final_word = True
    temporal_editing.refine_takes_with_temporal_context = refine_with_post_best_take_final_word
    # pipeline imported the function directly, so update its bound reference as well.
    pipeline.refine_takes_with_temporal_context = refine_with_post_best_take_final_word
