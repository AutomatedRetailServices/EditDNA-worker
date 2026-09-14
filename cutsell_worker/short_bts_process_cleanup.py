"""Remove tiny explicit recording-process BTS fragments at final draft level.

This guard is intentionally narrow: Hybrid must already classify the fragment as BTS,
the clip must be <=2 seconds, and the text itself must explicitly describe the act of
trying/retrying speech (for example ``trying to say``). It does not remove ordinary story
language such as ``trying to stay in character``.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from .contracts import DraftClip

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_PROCESS_PATTERNS = (
    re.compile(r"\btrying to say\b", re.IGNORECASE),
    re.compile(r"\btrying to remember (?:what|how)\b", re.IGNORECASE),
    re.compile(r"\bwhat was i saying\b", re.IGNORECASE),
    re.compile(r"\blet me say that again\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am) trying to say\b", re.IGNORECASE),
)


def _bts_confidence(diagnostics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for item in chunk.get("decisions") or ():
            if not isinstance(item, dict):
                continue
            if str(item.get("label") or "").strip().lower() != "bts":
                continue
            cid = str(item.get("clip_id") or "")
            if cid:
                out[cid] = max(out.get(cid, 0.0), float(item.get("confidence") or 0.0))
    return out


def _explicit_speech_process(text: str) -> bool:
    value = " ".join(str(text or "").split())
    return any(pattern.search(value) for pattern in _PROCESS_PATTERNS)


def suppress_short_explicit_bts_process_fragments(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    minimum_bts_confidence: float = 0.70,
    maximum_duration_sec: float = 2.0,
    maximum_tokens: int = 7,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    selected_list = list(selected)
    discarded_list = list(discarded)
    bts = _bts_confidence(diagnostics)
    removed: set[str] = set()
    audit: list[dict] = []

    for clip in selected_list:
        confidence = bts.get(clip.clip_id, 0.0)
        if confidence < minimum_bts_confidence:
            continue
        if float(clip.end) - float(clip.start) > maximum_duration_sec:
            continue
        if len(_TOKEN_RE.findall(str(clip.text or ""))) > maximum_tokens:
            continue
        if not _explicit_speech_process(clip.text):
            continue
        removed.add(clip.clip_id)
        audit.append({
            "reason": "short_explicit_recording_process_bts",
            "removed_clip_id": clip.clip_id,
            "bts_confidence": round(confidence, 4),
            "start": round(float(clip.start), 3),
            "end": round(float(clip.end), 3),
            "text": clip.text,
        })

    if not removed:
        return tuple(selected_list), tuple(discarded_list), ()
    moved = [replace(clip, selected=False) for clip in selected_list if clip.clip_id in removed]
    existing = {clip.clip_id for clip in discarded_list}
    return (
        tuple(clip for clip in selected_list if clip.clip_id not in removed),
        tuple(discarded_list + [clip for clip in moved if clip.clip_id not in existing]),
        tuple(audit),
    )


def install_short_bts_process_cleanup() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_short_bts_process_cleanup", False):
        return

    def build_with_short_bts_process_cleanup(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, discarded, audit = suppress_short_explicit_bts_process_fragments(
            draft.selected, draft.discarded, diagnostics
        )
        if not audit:
            return result
        diagnostics["short_bts_process_cleanup"] = list(audit)
        repaired = replace(draft, selected=selected, discarded=discarded, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_short_bts_process_cleanup._cutsell_short_bts_process_cleanup = True
    pipeline.build_flow_b_draft = build_with_short_bts_process_cleanup
