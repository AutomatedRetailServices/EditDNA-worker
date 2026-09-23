"""Deterministic authority for complete retry pairs separated by an incomplete reset.

When a creator records a complete delivery, immediately starts the same idea again but
abandons it, then records another complete delivery of that same idea, the incomplete
middle attempt is strong physical evidence of a retry sequence. In that narrow pattern,
the later complete delivery owns the message if it preserves the earlier numeric facts.

This authority runs after Hybrid cleanup and before Best Take. It never hardcodes source
phrases, clip ids, or benchmark timestamps. Ambiguous cases fail open unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
import re
import unicodedata

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "as", "at", "by", "con", "de", "del", "el", "en", "for",
    "from", "in", "la", "las", "lo", "los", "of", "on", "or", "para", "por", "que",
    "the", "to", "un", "una", "with", "y",
})


def _numbers(text: str) -> frozenset[str]:
    return frozenset(m.group(0).replace(",", ".") for m in _NUMBER_RE.finditer(str(text or "")))


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(str(text or ""))
        if len(token) >= 3 and token.casefold() not in _STOP
    }


def _semantic_overlap(left_text: str, right_text: str) -> float:
    left = _tokens(left_text)
    right = _tokens(right_text)
    if len(left) < 2 or len(right) < 2:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _normalized(text: str) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "").casefold())
    raw = "".join(c for c in raw if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", raw).split())


def _sequence_identity(left_text: str, right_text: str) -> float:
    left, right = _normalized(left_text), _normalized(right_text)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def install_incomplete_bridge_retry_authority() -> None:
    from . import hybrid_session_cleanup as cleanup

    original = cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_incomplete_bridge_retry_authority", False):
        return

    def protected(takes, context, editorial_judge, **kwargs):
        source_takes = tuple(sorted(tuple(takes), key=lambda t: (t.source_order, float(t.start), float(t.end))))
        result = original(source_takes, context, editorial_judge, **kwargs)
        kept_by_id = {t.clip_id: t for t in result.kept}
        deleted_by_id = {t.clip_id: t for t in result.deleted}
        restored_ids: set[str] = set()
        superseded_ids: set[str] = set()
        rows: list[dict] = []

        for i in range(len(source_takes) - 2):
            early, bridge, late = source_takes[i : i + 3]
            if early.source_asset_id != bridge.source_asset_id or early.source_asset_id != late.source_asset_id:
                continue
            if not bool(early.complete_idea) or bool(bridge.complete_idea) or not bool(late.complete_idea):
                continue
            if early.clip_id not in kept_by_id or bridge.clip_id not in deleted_by_id or late.clip_id not in deleted_by_id:
                continue
            gap1 = float(bridge.start) - float(early.end)
            gap2 = float(late.start) - float(bridge.end)
            if gap1 < 0.0 or gap1 > 8.0 or gap2 < 0.0 or gap2 > 8.0:
                continue

            early_numbers = _numbers(early.text)
            if early_numbers and not early_numbers.issubset(_numbers(late.text)):
                continue

            overlap = _semantic_overlap(early.text, late.text)
            seq = _sequence_identity(early.text, late.text)
            # Require strong delivery identity from two complementary text signals.
            if overlap < 0.60 or seq < 0.42:
                continue

            # The bridge must visibly belong to the earlier attempted restart: at least
            # one meaningful token or recognizable sequence prefix should connect it.
            bridge_overlap = _semantic_overlap(bridge.text, early.text)
            bridge_seq = _sequence_identity(bridge.text, early.text)
            if bridge_overlap < 0.40 and bridge_seq < 0.42:
                continue

            superseded_ids.add(early.clip_id)
            restored_ids.add(late.clip_id)
            rows.append({
                "authority": "incomplete_bridge_retry_authority",
                "early_clip_id": early.clip_id,
                "bridge_clip_id": bridge.clip_id,
                "late_clip_id": late.clip_id,
                "semantic_overlap": round(overlap, 3),
                "sequence_identity": round(seq, 3),
                "bridge_overlap": round(bridge_overlap, 3),
            })

        if not rows:
            return result

        kept = [t for t in result.kept if t.clip_id not in superseded_ids]
        kept.extend(deleted_by_id[cid] for cid in restored_ids)
        kept.sort(key=lambda t: (t.source_order, float(t.start), float(t.end)))
        deleted = [t for t in result.deleted if t.clip_id not in restored_ids]
        deleted.extend(kept_by_id[cid] for cid in superseded_ids)
        deleted.sort(key=lambda t: (t.source_order, float(t.start), float(t.end)))

        decisions = list(result.semantic_decisions)
        decision_map = {cid: (label, conf) for cid, label, conf in decisions}
        for row in rows:
            early_id = row["early_clip_id"]
            late_id = row["late_clip_id"]
            decision_map[early_id] = ("alternate", max(0.90, float(decision_map.get(early_id, ("", 0.0))[1])))
            decision_map[late_id] = ("winner", max(0.95, float(decision_map.get(late_id, ("", 0.0))[1])))
        semantic_decisions = tuple((cid, label, conf) for cid, (label, conf) in decision_map.items())

        diagnostics = tuple(result.diagnostics) + tuple(rows)
        return replace(
            result,
            kept=tuple(kept),
            deleted=tuple(deleted),
            diagnostics=diagnostics,
            semantic_decisions=semantic_decisions,
        )

    protected._cutsell_incomplete_bridge_retry_authority = True
    cleanup.apply_hybrid_session_cleanup = protected
