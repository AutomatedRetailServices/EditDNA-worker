"""Conservative V2 pre-Freeze execution of explicit recording-tail proposals."""
from dataclasses import replace
import math
import re


def trim_recording_tail(clip, decision, diagnostics, *, selected_clips=()):
    row = {"clip_id": clip.clip_id, "source_asset_id": clip.source_asset_id,
           "action": "preserve", "reason": "insufficient_evidence"}
    count = decision.trailing_recording_word_count
    confidence = (decision.confidence if decision.trailing_recording_confidence is None
                  else decision.trailing_recording_confidence)
    row.update(proposed_word_count=count, selection_confidence=decision.confidence,
               trailing_recording_confidence=confidence, kind=decision.trailing_recording_kind)
    words = tuple(clip.words)
    if (decision.action != "select" or not math.isfinite(confidence) or confidence < .97
            or type(count) is not int or not 1 <= count <= 8 or len(words) - count < 3):
        return clip, row
    tokenize = lambda text: re.findall(r"\w+", text.casefold())
    if tokenize(clip.text) != tokenize(" ".join(w.text for w in words)):
        return clip, {**row, "reason": "transcript_word_mismatch"}
    if any(not math.isfinite(w.start) or not math.isfinite(w.end) or w.end <= w.start for w in words):
        return clip, {**row, "reason": "invalid_alignment"}
    if any(a.end > b.start for a, b in zip(words, words[1:])):
        return clip, {**row, "reason": "overlapping_alignment"}
    head, tail = words[:-count], words[-count:]
    tail_tokens = tokenize(" ".join(w.text for w in tail))
    if any(any(ch.isdigit() for ch in token) for token in tail_tokens) or set(tail_tokens) & {
        "no", "not", "never", "nunca", "without", "sin",
    }:
        return clip, {**row, "reason": "protected_tail_fact"}
    if tail[-1].end - tail[0].start > 3 or not clip.start < head[-1].end < clip.end:
        return clip, row
    pauses = []
    for source in (diagnostics.get("attempt_reconstruction") or {}).get("positioned_performance_evidence", ()):
        if source.get("source_asset_id") != clip.source_asset_id:
            continue
        for event in source.get("positioned_events", ()):
            if (event.get("kind") != "audio_silence_interval"
                    or event.get("evidence_source") != "audio_silence" or event.get("confidence") != 1.0):
                continue
            try:
                start, end = float(event["start"]), float(event["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if math.isfinite(start) and math.isfinite(end) and head[-1].start <= start < end <= tail[0].end and end - start >= 1.2:
                pauses.append((start, end))
    abandoned = decision.trailing_recording_kind == "abandoned_restart"
    if abandoned:
        from .unified_selection_reasoner import _content_tokens
        import unicodedata
        native = set(diagnostics.get("v2_native_selection_modalities", ())) >= {"audio", "video", "source_words"}
        replacement = next((c for c in selected_clips if c.clip_id == decision.trailing_replacement_clip_id
                            and c.clip_id != clip.clip_id and c.source_asset_id == clip.source_asset_id
                            and c.start >= clip.end), None)
        if not native or replacement is None:
            return clip, {**row, "reason": "missing_native_av_or_selected_replacement"}
        rejected, covered = _content_tokens(" ".join(w.text for w in tail)), _content_tokens(replacement.text)
        missing = rejected - covered
        # A model-observed aborted syllable is corroborated only by the same
        # selected completed attempt. Accent folding alone never authorizes it.
        final = tail_tokens[-1] if tail_tokens else ""
        unaccented = not any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", final))
        partial = (missing == {final} and unaccented and len(final) >= 3
                   and any(t.startswith(final) and len(t) >= len(final) + 2 for t in covered))
        if len(rejected & covered) < 2 or (missing and not partial):
            return clip, {**row, "reason": "uncovered_suffix_content"}
    if not abandoned and not pauses:
        return clip, {**row, "reason": "missing_independent_pause"}
    text = " ".join(w.text for w in head)
    return replace(clip, end=head[-1].end, text=text, caption_text=text, words=head), {
        **row, "action": "trim", "reason": ("native_av_abandoned_restart_with_selected_coverage" if abandoned
                                               else "explicit_recording_proposal_and_measured_pause"),
        "allowed_end": head[-1].end, "original_end": clip.end,
        "removed_words": [w.text for w in tail], "measured_pauses": pauses,
    }
