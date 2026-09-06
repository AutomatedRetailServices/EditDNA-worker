"""Conservative internal retry cleanup after final Best-Take selection.

A single selected DraftClip can still contain two spoken attempts when ASR/candidate
segmentation merged an aborted delivery and the clean retake. Pre-selection Best Take
cannot compare those attempts because they are inside the same logical clip.

This pass removes spoken material only when a later contiguous repeated opening proves a
retry and the later delivery recovers the earlier attempt's audience-facing content.
Numeric facts and negations must survive. Visual/recording-process reset evidence is also
required near the later restart. Ambiguity fails open.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unicodedata
from typing import Iterable

from .contracts import DraftClip

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "que", "se", "so",
    "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_NEGATIONS = frozenset({"no", "not", "never", "nunca", "ni", "sin", "without"})
_AUTHORITATIVE_RETRY = frozenset({
    "retry_setup", "false_start", "wrong_take", "searching_for_words",
    "breaking_character", "unintentional_dead_air",
})
_PHYSICAL = frozenset({"hand_motion_reset_candidate", "body_reset_candidate"})
_BREAKS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canon(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    found = re.findall(r"[a-z0-9]+(?:[-–][0-9]+)?%?", text)
    return found[0] if found else ""


def _content(tokens) -> set[str]:
    return {token for token in tokens if len(token) >= 3 and token not in _STOP}


def _critical(tokens) -> set[str]:
    return {
        token for token in tokens
        if any(ch.isdigit() for ch in token) or token in _NEGATIONS
    }


def _events_for_source(diagnostics: dict, source_asset_id: str) -> tuple[dict, ...]:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(event for event in (source.get("events") or ()) if isinstance(event, dict))
    return ()


def _has_retry_evidence(events, restart_time: float, *, radius_sec: float = 0.90) -> tuple[bool, dict]:
    nearby = [
        event for event in events
        if float(event.get("end") or 0.0) >= restart_time - radius_sec
        and float(event.get("start") or 0.0) <= restart_time + radius_sec
    ]
    authoritative = [
        event for event in nearby
        if _kind(event.get("kind")) in _AUTHORITATIVE_RETRY
        and float(event.get("confidence") or 0.0) >= 0.78
    ]
    if authoritative:
        return True, {
            "evidence_type": "authoritative_retry_event",
            "event_kind": _kind(authoritative[0].get("kind")),
            "event_confidence": round(float(authoritative[0].get("confidence") or 0.0), 4),
        }

    physical = [
        event for event in nearby
        if _kind(event.get("kind")) in _PHYSICAL
        and float(event.get("confidence") or 0.0) >= 0.90
    ]
    breaks = [
        event for event in nearby
        if _kind(event.get("kind")) in _BREAKS
        and float(event.get("confidence") or 0.0) >= (
            0.72 if _kind(event.get("kind")) == "facial_expression_shift_candidate" else 0.80
        )
    ]
    hand_count = sum(1 for event in physical if _kind(event.get("kind")) == "hand_motion_reset_candidate")
    if len(physical) >= 2 and breaks and hand_count >= 1:
        return True, {
            "evidence_type": "multimodal_retry_reset",
            "physical_event_count": len(physical),
            "break_event_count": len(breaks),
        }
    return False, {}


def _child_id(clip: DraftClip, side: str, start: float, end: float) -> str:
    digest = hashlib.sha256(
        f"{clip.clip_id}|post-selection-internal-retake|{side}|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{clip.clip_id}__psirt{side}{digest}"


def _text(words) -> str:
    return " ".join(str(word.text or "").strip() for word in words).strip()


def _find_repeated_opening(words, *, min_size: int = 5, max_size: int = 8, minimum_attempt_words: int = 4):
    tokens = tuple(_canon(getattr(word, "text", "")) for word in words)
    for size in range(min(max_size, len(tokens) // 2), min_size - 1, -1):
        first_seen: dict[tuple[str, ...], int] = {}
        for second in range(0, len(tokens) - size + 1):
            phrase = tokens[second:second + size]
            if any(not token for token in phrase):
                continue
            if len(_content(phrase)) < 3:
                continue
            first = first_seen.get(phrase)
            if first is None:
                first_seen[phrase] = second
                continue
            if second - first < minimum_attempt_words:
                continue
            # Need enough later delivery to plausibly be the clean take.
            if len(tokens) - second < size + 3:
                continue
            return first, second, size, phrase
    return None


def trim_selected_internal_retakes(
    selected: Iterable[DraftClip],
    diagnostics: dict,
    *,
    minimum_attempt_content_coverage: float = 0.72,
    minimum_shared_content: int = 4,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    output: list[DraftClip] = []
    audit: list[dict] = []

    for clip in selected:
        words = tuple(sorted(clip.words, key=lambda word: (float(word.start), float(word.end))))
        if len(words) < 13:
            output.append(clip)
            continue
        found = _find_repeated_opening(words)
        if found is None:
            output.append(clip)
            continue
        first, second, size, phrase = found

        restart_time = float(words[second].start)
        ok, evidence = _has_retry_evidence(_events_for_source(diagnostics, clip.source_asset_id), restart_time)
        if not ok:
            output.append(clip)
            continue

        attempt_words = words[first:second]
        later_words = words[second:]
        attempt_tokens = tuple(_canon(word.text) for word in attempt_words)
        later_tokens = tuple(_canon(word.text) for word in later_words)
        attempt_content = _content(attempt_tokens)
        later_content = _content(later_tokens)
        shared = len(attempt_content & later_content)
        coverage = shared / max(1, len(attempt_content))
        if shared < minimum_shared_content or coverage < minimum_attempt_content_coverage:
            output.append(clip)
            continue
        if not _critical(attempt_tokens).issubset(_critical(later_tokens)):
            output.append(clip)
            continue

        # Keep any unique prefix before the failed attempt, then jump to the later clean retake.
        prefix_words = words[:first]
        pieces: list[DraftClip] = []
        if prefix_words:
            prefix_text = _text(prefix_words)
            if prefix_text:
                pieces.append(replace(
                    clip,
                    clip_id=_child_id(clip, "p", float(clip.start), float(prefix_words[-1].end)),
                    end=float(prefix_words[-1].end),
                    text=prefix_text,
                    caption_text=prefix_text,
                    words=prefix_words,
                ))
        later_text = _text(later_words)
        if not later_text:
            output.append(clip)
            continue
        pieces.append(replace(
            clip,
            clip_id=_child_id(clip, "r", float(later_words[0].start), float(clip.end)),
            start=float(later_words[0].start),
            text=later_text,
            caption_text=later_text,
            words=later_words,
        ))
        output.extend(pieces)
        removed_start = float(words[first].start)
        removed_end = float(words[second].start)
        audit.append({
            "authority": "post_selection_internal_retake_trim",
            "parent_clip_id": clip.clip_id,
            "reason": "earlier_internal_attempt_covered_by_later_clean_retake",
            "removed_start": round(removed_start, 3),
            "removed_end": round(removed_end, 3),
            "removed_sec": round(removed_end - removed_start, 3),
            "repeated_phrase": " ".join(phrase),
            "repeat_width": size,
            "shared_content_tokens": shared,
            "attempt_content_coverage": round(coverage, 4),
            **evidence,
        })

    output.sort(key=lambda item: (item.source_order, float(item.start), float(item.end), item.clip_id))
    return tuple(output), tuple(audit)


def install_post_selection_internal_retake_trim() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_internal_retake_trim", False):
        return

    def build_with_post_selection_internal_retake_trim(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, audit = trim_selected_internal_retakes(draft.selected, diagnostics)
        if not audit:
            return result
        diagnostics["post_selection_internal_retake_trim"] = list(audit)
        repaired = replace(draft, selected=selected, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_post_selection_internal_retake_trim._cutsell_post_selection_internal_retake_trim = True
    pipeline.build_flow_b_draft = build_with_post_selection_internal_retake_trim
