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
_DANGLING_TERMINALS = frozenset({
    "a", "al", "and", "because", "con", "de", "del", "for", "from", "if", "in",
    "of", "or", "para", "pero", "por", "porque", "que", "si", "so", "the", "to",
    "with", "y",
})
_ASSERTION_FRAMING = frozenset({
    "afirmo", "afirma", "avala", "avalado", "ciencia", "cientifica", "cientifico",
    "cientificamente", "comprobado", "convencida", "convencido", "creo", "dice",
    "evidence", "evidencia", "proven", "science", "scientific", "think", "believe",
})
_DETERMINISTIC_RETRY_KINDS = frozenset({
    "same_opening_restart",
    "same_opening_abandoned_start",
    "incomplete_attempt_completed_by_retry",
    "multimodal_corroborated_retry",
})


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


def _substantive(text: str) -> set[str]:
    """Content tokens with assertion boilerplate removed."""
    return {token for token in _thematic(text) if token not in _ASSERTION_FRAMING}


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_canon(token) for token in _TOKEN_RE.findall(str(text or "")))


def _is_contiguous_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index:index + width] == needle for index in range(len(haystack) - width + 1))


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


def _attempt_completeness(diagnostics: dict) -> dict[str, bool]:
    rows = (diagnostics.get("attempt_reconstruction") or {}).get("attempts") or ()
    return {
        str(row.get("clip_id")): bool(row.get("complete_idea"))
        for row in rows
        if isinstance(row, dict) and row.get("clip_id") and row.get("complete_idea") is not None
    }


def _deterministic_retry_rows(diagnostics: dict):
    equivalence = diagnostics.get("semantic_idea_equivalence") or {}
    for row in equivalence.get("merges") or ():
        if isinstance(row, dict) and str(row.get("accepted_by") or "") in _DETERMINISTIC_RETRY_KINDS:
            yield row


def deterministic_retry_resolution(selected, alternates, discarded, diagnostics: dict):
    """Apply already-proven retry relations to final membership.

    A deterministic relation proves that the clips compete.  Completeness or
    stronger positive audience-facing evidence may then settle the winner; a
    failed peer is never resurrected merely because it is later.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    all_by_id = {clip.clip_id: clip for clip in (*selected, *alternates, *discarded)}
    votes = _hybrid_votes(diagnostics)
    complete = _attempt_completeness(diagnostics)
    move: set[str] = set()
    add: set[str] = set()
    audit: list[dict] = []
    for row in _deterministic_retry_rows(diagnostics):
        left_id = str(row.get("left_clip_id") or "")
        right_id = str(row.get("right_clip_id") or "")
        if left_id not in all_by_id or right_id not in all_by_id:
            continue
        left_selected = left_id in selected_by_id
        right_selected = right_id in selected_by_id
        if not (left_selected or right_selected):
            continue

        left, right = all_by_id[left_id], all_by_id[right_id]
        if left.source_asset_id != right.source_asset_id:
            continue

        winner_id = loser_id = ""
        if complete.get(left_id) is False and complete.get(right_id) is True:
            winner_id, loser_id = right_id, left_id
        elif complete.get(right_id) is False and complete.get(left_id) is True:
            winner_id, loser_id = left_id, right_id
        elif left_selected != right_selected:
            current_id, peer_id = (left_id, right_id) if left_selected else (right_id, left_id)
            current_positive = _strongest(votes, current_id, {"winner", "keep"})
            peer_positive = _strongest(votes, peer_id, {"winner", "keep"})
            if peer_positive >= 0.90 and peer_positive - current_positive >= 0.05 - 1e-9:
                winner_id, loser_id = peer_id, current_id
        if not winner_id or loser_id not in selected_by_id:
            continue

        winner, loser = all_by_id[winner_id], all_by_id[loser_id]
        winner_positive = _strongest(votes, winner_id, {"winner", "keep"})
        winner_negative = _strongest(votes, winner_id, {"alternate", "failed"})
        if winner_id not in selected_by_id and (winner_positive < 0.80 or winner_negative >= 0.80):
            continue
        if not _critical(loser.text).issubset(_critical(winner.text)):
            continue
        move.add(loser_id)
        if winner_id not in selected_by_id:
            add.add(winner_id)
        audit.append({
            "clip_id": loser_id,
            "winner_clip_id": winner_id,
            "reason": "deterministic_retry_final_membership_resolution",
            "accepted_by": str(row.get("accepted_by") or ""),
            "loser_complete_idea": complete.get(loser_id),
            "winner_complete_idea": complete.get(winner_id),
            "loser_positive_confidence": round(_strongest(votes, loser_id, {"winner", "keep"}), 4),
            "winner_positive_confidence": round(winner_positive, 4),
        })
    return move, add, audit


def contained_proxy_duplicate_ids(selected, alternates, discarded, diagnostics: dict):
    """Propagate a confirmed duplicate through its enclosing source interval.

    A segmentation pass can select only the tail of a discarded monolith.  If
    that monolith is already confirmed equivalent to another selected winner,
    the contained tail cannot be rendered as a separate audience-facing idea.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    all_unselected = tuple((*alternates, *discarded))
    move: set[str] = set()
    audit: list[dict] = []
    equivalence = diagnostics.get("semantic_idea_equivalence") or {}
    for row in equivalence.get("merges") or ():
        if not isinstance(row, dict):
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < 0.85:
            continue
        left_id, right_id = str(row.get("left_clip_id") or ""), str(row.get("right_clip_id") or "")
        for proxy_id, winner_id in ((left_id, right_id), (right_id, left_id)):
            winner = selected_by_id.get(winner_id)
            proxy = next((clip for clip in all_unselected if clip.clip_id == proxy_id), None)
            if winner is None or proxy is None or winner.source_asset_id != proxy.source_asset_id:
                continue
            for inner in selected:
                if inner.clip_id == winner_id or inner.source_asset_id != proxy.source_asset_id:
                    continue
                temporal_containment = (
                    float(proxy.start) <= float(inner.start) + 1e-3
                    and float(proxy.end) >= float(inner.end) - 1e-3
                )
                lexical_containment = _is_contiguous_subsequence(_tokens(inner.text), _tokens(proxy.text))
                if not (temporal_containment or lexical_containment):
                    continue
                if not _critical(inner.text).issubset(_critical(winner.text)):
                    continue
                move.add(inner.clip_id)
                audit.append({
                    "clip_id": inner.clip_id,
                    "proxy_clip_id": proxy_id,
                    "winner_clip_id": winner_id,
                    "reason": "contained_fragment_of_confirmed_duplicate",
                    "equivalence_confidence": round(confidence, 4),
                })
    return move, audit


def missing_continuation_bridge_ids(selected, alternates, discarded, diagnostics: dict):
    """Restore an omitted positive bridge that closes a selected continuation."""
    ordered = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    complete = _attempt_completeness(diagnostics)
    votes = _hybrid_votes(diagnostics)
    candidates = tuple((*alternates, *discarded))
    add: set[str] = set()
    audit: list[dict] = []
    for left, right in zip(ordered, ordered[1:]):
        if left.source_asset_id != right.source_asset_id:
            continue
        for bridge in candidates:
            if bridge.source_asset_id != left.source_asset_id or complete.get(bridge.clip_id) is not False:
                continue
            if not (float(left.end) <= float(bridge.start) + 1e-3 and float(bridge.end) <= float(right.start) + 1e-3):
                continue
            left_gap = float(bridge.start) - float(left.end)
            right_gap = float(right.start) - float(bridge.end)
            if left_gap < -1e-3 or left_gap > 0.8 or right_gap < -1e-3 or right_gap > 2.0:
                continue
            tokens = _tokens(bridge.text)
            if not tokens or tokens[-1] not in _DANGLING_TERMINALS:
                continue
            positive = _strongest(votes, bridge.clip_id, {"winner", "keep"})
            negative = _strongest(votes, bridge.clip_id, {"alternate", "failed"})
            if positive < 0.80 or positive + 1e-9 < negative:
                continue
            add.add(bridge.clip_id)
            audit.append({
                "clip_id": bridge.clip_id,
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "reason": "missing_positive_continuation_bridge_restored",
                "left_gap_sec": round(left_gap, 3),
                "right_gap_sec": round(right_gap, 3),
                "terminal_token": tokens[-1],
                "positive_confidence": round(positive, 4),
                "negative_confidence": round(negative, 4),
            })
    return add, audit


def redundant_continuation_chain_ids(selected, diagnostics: dict):
    """Remove a later continuation chain whose critical claim is already covered."""
    selected_by_id = {clip.clip_id: clip for clip in selected}
    ordered = tuple(sorted(selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)))
    index_by_id = {clip.clip_id: index for index, clip in enumerate(ordered)}
    move: set[str] = set()
    audit: list[dict] = []
    equivalence = diagnostics.get("semantic_idea_equivalence") or {}
    for chain in equivalence.get("continuation_merges") or ():
        if not isinstance(chain, dict) or str(chain.get("accepted_by") or "") != "sentence_continuation":
            continue
        ids = [str(chain.get("left_clip_id") or ""), str(chain.get("right_clip_id") or "")]
        if any(clip_id not in selected_by_id for clip_id in ids):
            continue
        chain_clips = [selected_by_id[clip_id] for clip_id in ids]
        first_index = min(index_by_id[clip_id] for clip_id in ids)
        if first_index <= 0 or len({clip.source_asset_id for clip in chain_clips}) != 1:
            continue
        prior = [
            clip for clip in ordered[max(0, first_index - 4):first_index]
            if clip.source_asset_id == chain_clips[0].source_asset_id
        ]
        if not prior:
            continue
        later_text = " ".join(clip.text for clip in chain_clips)
        prior_text = " ".join(clip.text for clip in prior)
        later_content = _substantive(later_text)
        prior_content = _substantive(prior_text)
        if len(later_content) < 2 or not _critical(later_text):
            continue
        if not _critical(later_text).issubset(_critical(prior_text)):
            continue
        coverage = len(later_content & prior_content) / max(1, len(later_content))
        if coverage < 0.80:
            continue
        move.update(ids)
        audit.append({
            "clip_ids": ids,
            "reason": "later_continuation_chain_repeats_nearby_critical_claim",
            "prior_clip_ids": [clip.clip_id for clip in prior],
            "substantive_coverage": round(coverage, 4),
            "critical_markers": sorted(_critical(later_text)),
        })
    return move, audit


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


def confirmed_selected_duplicate_ids(selected, diagnostics: dict):
    """Resolve direct, already-confirmed equivalence between final winners.

    Group cohesion may conservatively keep two families separate because a
    third member conflicts with one side.  Once the actual winners are known,
    a direct high-confidence equivalence verdict between those two winners is
    still valid evidence.  Move only the candidate that Hybrid independently
    called alternate/failed when the other has a strong keep/winner verdict.
    Numeric and negated facts remain fail-open unless the kept winner carries
    the same critical markers.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    votes = _hybrid_votes(diagnostics)
    move: set[str] = set()
    audit: list[dict] = []
    equivalence = diagnostics.get("semantic_idea_equivalence") or {}
    for row in equivalence.get("merges") or ():
        if not isinstance(row, dict):
            continue
        left_id = str(row.get("left_clip_id") or "")
        right_id = str(row.get("right_clip_id") or "")
        if left_id not in selected_by_id or right_id not in selected_by_id:
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < 0.85:
            continue

        left_positive = _strongest(votes, left_id, {"winner", "keep"})
        right_positive = _strongest(votes, right_id, {"winner", "keep"})
        left_negative = _strongest(votes, left_id, {"alternate", "failed"})
        right_negative = _strongest(votes, right_id, {"alternate", "failed"})
        loser_id = winner_id = ""
        if right_positive >= 0.90 and left_negative >= 0.80 and left_positive < 0.90:
            loser_id, winner_id = left_id, right_id
        elif left_positive >= 0.90 and right_negative >= 0.80 and right_positive < 0.90:
            loser_id, winner_id = right_id, left_id
        if not loser_id or loser_id in move:
            continue
        loser = selected_by_id[loser_id]
        winner = selected_by_id[winner_id]
        if not _critical(loser.text).issubset(_critical(winner.text)):
            continue
        move.add(loser_id)
        audit.append({
            "clip_id": loser_id,
            "winner_clip_id": winner_id,
            "reason": "direct_equivalence_confirmed_final_winner",
            "equivalence_confidence": round(confidence, 4),
            "winner_positive_confidence": round(
                _strongest(votes, winner_id, {"winner", "keep"}), 4
            ),
            "loser_negative_confidence": round(
                _strongest(votes, loser_id, {"alternate", "failed"}), 4
            ),
        })
    return move, audit


def terminally_incomplete_selected_ids(selected, diagnostics: dict):
    """Discard tiny attempt fragments proven incomplete upstream."""
    selected_ids = {clip.clip_id for clip in selected}
    move: set[str] = set()
    audit: list[dict] = []
    reconstruction = diagnostics.get("attempt_reconstruction") or {}
    for row in reconstruction.get("attempts") or ():
        if not isinstance(row, dict):
            continue
        clip_id = str(row.get("clip_id") or "")
        if clip_id not in selected_ids or row.get("complete_idea") is not False:
            continue
        try:
            duration = float(row.get("duration_sec") or 0.0)
        except (TypeError, ValueError):
            continue
        clip = next((item for item in selected if item.clip_id == clip_id), None)
        if clip is None:
            continue
        token_count = len(_TOKEN_RE.findall(str(clip.text or "")))
        terminally_open = str(clip.text or "").rstrip().endswith(("...", "…"))
        if duration > 2.0 or token_count > 4 or not terminally_open:
            continue
        move.add(clip_id)
        audit.append({
            "clip_id": clip_id,
            "reason": "short_terminally_incomplete_attempt",
            "duration_sec": round(duration, 3),
            "token_count": token_count,
        })
    return move, audit


def apply_selection_conflicted_bridge_guard(draft):
    """Move proven conflicted redundant bridges from Selected to Alternates/SWAP."""
    diagnostics = dict(draft.diagnostics or {})
    bridge_ids, bridge_audit = conflicted_redundant_bridge_ids(draft.selected, diagnostics)
    duplicate_ids, duplicate_audit = confirmed_selected_duplicate_ids(draft.selected, diagnostics)
    incomplete_ids, incomplete_audit = terminally_incomplete_selected_ids(draft.selected, diagnostics)
    retry_ids, retry_add_ids, retry_audit = deterministic_retry_resolution(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    proxy_ids, proxy_audit = contained_proxy_duplicate_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    continuation_add_ids, continuation_add_audit = missing_continuation_bridge_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )

    all_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.alternates, *draft.discarded)}
    provisional = tuple(
        (*draft.selected, *(all_by_id[clip_id] for clip_id in continuation_add_ids if clip_id in all_by_id))
    )
    chain_ids, chain_audit = redundant_continuation_chain_ids(provisional, diagnostics)

    move_ids = bridge_ids | duplicate_ids | incomplete_ids | retry_ids | proxy_ids | chain_ids
    add_ids = (retry_add_ids | continuation_add_ids) - move_ids
    audit = (
        bridge_audit + duplicate_audit + incomplete_audit + retry_audit
        + proxy_audit + continuation_add_audit + chain_audit
    )
    if not move_ids and not add_ids:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    selected = [clip for clip in draft.selected if clip.clip_id not in move_ids]
    for clip_id in sorted(add_ids):
        clip = all_by_id.get(clip_id)
        if clip is not None and clip_id not in {item.clip_id for item in selected}:
            selected.append(replace(clip, selected=True))
    selected.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    alternates = [clip for clip in draft.alternates if clip.clip_id not in add_ids]
    existing = {clip.clip_id for clip in alternates}
    for clip_id in sorted(move_ids):
        clip = selected_by_id.get(clip_id)
        if clip is not None and clip_id not in existing:
            alternates.append(replace(clip, selected=False))
    alternates.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    diagnostics["selection_conflicted_bridge_guard"] = list(audit)
    return replace(
        draft,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=tuple(clip for clip in draft.discarded if clip.clip_id not in add_ids),
        diagnostics=diagnostics,
    )
