"""Final Selection guard for conflicted but contextually redundant retry bridges.

This module owns semantic membership only. It never changes clip boundaries.
A short selected clip may move to Alternates/SWAP when Hybrid strongly disagrees
(keep/winner vs alternate) but the neighboring selected deliveries independently
prove that its semantic content is already covered. Ambiguity and unique critical
facts fail open.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})
_DISCOURSE = frozenset({
    "ahi", "alli", "aca", "aqui", "entonces", "luego", "despues", "cuando", "donde",
    "fue", "era", "eran", "estaba", "estaban", "esta", "estan", "haber", "habia",
    "hacer", "hace", "hacia", "hice", "hizo", "hicieron", "mandar", "mando", "mandaron",
    "tener", "tengo", "tenia", "tuve", "problema", "problemas", "otro", "otros", "otra", "otras",
    "there", "here", "then", "after", "when", "where", "was", "were", "did", "do", "does",
    "made", "make", "had", "have", "has", "problem", "problems", "thing", "things", "other",
})
_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without", "ni"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _concept(token: str) -> str:
    value = _canon(token)
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 5 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _thematic(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _concept(raw)
        if len(token) >= 3 and token not in _STOP and token not in _DISCOURSE:
            out.add(token)
    return out


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _hybrid_votes(diagnostics: dict) -> dict[str, list[tuple[str, float]]]:
    votes: dict[str, list[tuple[str, float]]] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for row in chunk.get("decisions") or ():
            if not isinstance(row, dict) or not row.get("clip_id") or not row.get("label"):
                continue
            try:
                confidence = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                continue
            votes.setdefault(str(row["clip_id"]), []).append((str(row["label"]), confidence))
    return votes


def _strongest(votes, clip_id: str, labels: set[str]) -> float:
    return max(
        (confidence for label, confidence in votes.get(str(clip_id), ()) if label in labels),
        default=0.0,
    )


def conflicted_redundant_bridge_ids(selected, diagnostics: dict):
    """Return selected clip ids that should become Alternates/SWAP.

    Requirements are deliberately independent and conservative:
    - Hybrid contains a strong editorial conflict for the middle take;
    - the take is short and physically between neighboring selected deliveries;
    - both neighbors have meaningful semantic evidence;
    - at least 80% of the middle take's thematic content is covered across neighbors,
      with overlap on both sides;
    - no numeric or negated fact exists only in the middle take.

    A strong keep/winner vote only defeats the conflict when it has a meaningful
    confidence margin over the alternate vote. Near-tied overlapping Hybrid windows
    remain a true conflict and are resolved by the independent neighbor-coverage proof.
    """
    ordered = tuple(sorted(
        selected,
        key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id),
    ))
    votes = _hybrid_votes(diagnostics)
    move: set[str] = set()
    audit: list[dict] = []

    for index in range(1, len(ordered) - 1):
        left, middle, right = ordered[index - 1], ordered[index], ordered[index + 1]
        if not (left.source_asset_id == middle.source_asset_id == right.source_asset_id):
            continue

        alternate_strength = _strongest(votes, middle.clip_id, {"alternate"})
        keep_strength = _strongest(votes, middle.clip_id, {"winner", "keep"})
        if alternate_strength < 0.80 or keep_strength < 0.80:
            continue
        # Preserve a genuinely decisive strong keep. A near tie (for example 0.90
        # winner vs 0.88 alternate from overlapping windows) is still a conflict.
        keep_margin = keep_strength - alternate_strength
        if keep_strength >= 0.90 and keep_margin >= 0.05:
            continue

        duration = max(0.0, float(middle.end) - float(middle.start))
        left_gap = float(middle.start) - float(left.end)
        right_gap = float(right.start) - float(middle.end)
        if duration <= 0.0 or duration > 5.0:
            continue
        if left_gap < 0.0 or left_gap > 5.0 or right_gap < 0.0 or right_gap > 10.0:
            continue

        left_strength = _strongest(votes, left.clip_id, {"winner", "keep"})
        right_strength = max(
            _strongest(votes, right.clip_id, {"winner", "keep"}),
            _strongest(votes, right.clip_id, {"alternate", "failed"}),
        )
        if left_strength < 0.90 or right_strength < 0.75:
            continue

        middle_content = _thematic(middle.text)
        left_content = _thematic(left.text)
        right_content = _thematic(right.text)
        if len(middle_content) < 2:
            continue
        left_shared = middle_content & left_content
        right_shared = middle_content & right_content
        union_shared = middle_content & (left_content | right_content)
        coverage = len(union_shared) / max(1, len(middle_content))
        if len(left_shared) < 1 or len(right_shared) < 1 or coverage < 0.80:
            continue
        if not _critical(middle.text).issubset(_critical(left.text + " " + right.text)):
            continue

        move.add(middle.clip_id)
        audit.append({
            "clip_id": middle.clip_id,
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "reason": "conflicted_redundant_bridge_moved_to_swap",
            "alternate_confidence": round(alternate_strength, 4),
            "keep_confidence": round(keep_strength, 4),
            "keep_margin": round(keep_margin, 4),
            "left_strength": round(left_strength, 4),
            "right_strength": round(right_strength, 4),
            "thematic_union_coverage": round(coverage, 4),
            "left_shared_thematic_tokens": len(left_shared),
            "right_shared_thematic_tokens": len(right_shared),
            "duration_sec": round(duration, 3),
            "left_gap_sec": round(left_gap, 3),
            "right_gap_sec": round(right_gap, 3),
        })

    return move, audit


def apply_selection_conflicted_bridge_guard(draft):
    """Move proven conflicted redundant bridges from Selected to Alternates/SWAP."""
    diagnostics = dict(draft.diagnostics or {})
    move_ids, audit = conflicted_redundant_bridge_ids(draft.selected, diagnostics)
    if not move_ids:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    selected = tuple(clip for clip in draft.selected if clip.clip_id not in move_ids)
    alternates = list(draft.alternates)
    existing = {clip.clip_id for clip in alternates}
    for clip_id in sorted(move_ids):
        clip = selected_by_id.get(clip_id)
        if clip is not None and clip_id not in existing:
            alternates.append(replace(clip, selected=False))
    alternates.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    diagnostics["selection_conflicted_bridge_guard"] = list(audit)
    return replace(
        draft,
        selected=selected,
        alternates=tuple(alternates),
        diagnostics=diagnostics,
    )
