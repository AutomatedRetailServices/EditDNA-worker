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
_DEPENDENT_OPENERS = frozenset({
    "cuando", "whereas", "while",
})
_ASSERTION_FRAMING = frozenset({
    "afirmo", "afirma", "avala", "avalado", "ciencia", "cientifica", "cientifico",
    "cientificamente", "comprobado", "convencida", "convencido", "creo", "dice",
    "am", "estoy", "evidence", "evidencia", "proven", "science", "scientific", "think", "believe",
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


def _audience_support(diagnostics: dict, clip_id: str) -> float:
    """Strongest Hybrid observation that classified the clip as audience speech."""
    best = 0.0
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        for row in chunk.get("decisions") or ():
            if str(row.get("clip_id") or "") != clip_id:
                continue
            if str(row.get("content_role") or "") != "audience":
                pass
            else:
                try:
                    best = max(best, float(row.get("confidence") or 0.0))
                except (TypeError, ValueError):
                    pass
            audiovisual = row.get("audiovisual") or {}
            for observation in audiovisual.get("observations") or ():
                if not isinstance(observation, dict) or str(observation.get("role") or "") != "audience":
                    continue
                try:
                    best = max(best, float(observation.get("confidence") or 0.0))
                except (TypeError, ValueError):
                    continue
    return best


def _clean_cut_role_support(diagnostics: dict, clip_id: str, roles: set[str]) -> float:
    """Strongest already-recorded AV role support from the clean-cut judge."""
    best = 0.0
    for row in diagnostics.get("clean_cut_judge") or ():
        if not isinstance(row, dict) or str(row.get("clip_id") or "") != clip_id:
            continue
        audiovisual = row.get("audiovisual") or {}
        for observation in audiovisual.get("observations") or ():
            if str(observation.get("role") or "") not in roles:
                continue
            try:
                best = max(best, float(observation.get("confidence") or 0.0))
            except (TypeError, ValueError):
                continue
    return best


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


def _take_judge_usability(diagnostics: dict) -> tuple[dict[str, str], dict[str, dict]]:
    """Return the strongest existing Best-Take usability evidence by clip.

    The final guard does not invent a new performance judgment.  It only
    consumes the terminal judgment already recorded by the take judge so a
    later per-idea resolver cannot accidentally promote a candidate that the
    real delivery pass found unusable.
    """
    status: dict[str, str] = {}
    member: dict[str, dict] = {}
    for group in diagnostics.get("take_judge_groups") or ():
        if not isinstance(group, dict):
            continue
        for clip_id, value in (group.get("candidate_usability_summary") or {}).items():
            clip_id = str(clip_id)
            if str(value or "").upper() == "UNUSABLE":
                status[clip_id] = "UNUSABLE"
            else:
                status.setdefault(clip_id, str(value or ""))
        for clip_id, value in (group.get("member_usability") or {}).items():
            if isinstance(value, dict):
                member[str(clip_id)] = dict(value)
    return status, member


def _semantic_delete_recommended_ids(diagnostics: dict) -> set[str]:
    """Candidates carrying any explicit semantic deletion recommendation."""
    return {
        str(row.get("clip_id"))
        for chunk in diagnostics.get("hybrid_editorial_chunks") or ()
        for row in (chunk.get("decisions") or ())
        if isinstance(row, dict)
        and row.get("clip_id")
        and bool(row.get("semantic_delete_recommended"))
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
    usability, _member_usability = _take_judge_usability(diagnostics)
    move: set[str] = set()
    add: set[str] = set()
    audit: list[dict] = []
    for row in _deterministic_retry_rows(diagnostics):
        fuller_restart_tie = False
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
            current, peer = all_by_id[current_id], all_by_id[peer_id]
            peer_is_short_open_restart = (
                str(row.get("accepted_by") or "") in {
                    "same_opening_restart", "same_opening_abandoned_start",
                }
                and float(peer.end) - float(peer.start) <= 3.0
                and float(current.end) - float(current.start)
                >= 1.5 * max(0.001, float(peer.end) - float(peer.start))
                and not str(peer.text or "").rstrip().endswith((".", "?", "!", "…"))
            )
            if (
                peer_positive >= 0.90
                and peer_positive - current_positive >= 0.05 - 1e-9
                and not peer_is_short_open_restart
            ):
                winner_id, loser_id = peer_id, current_id
            else:
                # A deterministic restart relation already proves these two
                # deliveries compete.  Provider ties must not leave the shorter
                # earlier attempt selected beside a substantially longer clean
                # retry merely because both received the same positive label.
                # This is deliberately limited to a later same-source retry,
                # strong audience evidence, no strong negative verdict and an
                # appreciably fuller physical delivery.  Critical facts remain
                # protected by the subset check below.
                peer_negative = _strongest(votes, peer_id, {"alternate", "failed"})
                later_fuller_retry = (
                    float(peer.start) > float(current.start)
                    and float(peer.end) - float(peer.start)
                    >= 1.15 * max(0.001, float(current.end) - float(current.start))
                )
                if (
                    peer_positive >= 0.90
                    and peer_positive + 1e-9 >= current_positive
                    and peer_negative < 0.80
                    and complete.get(peer_id) is not False
                    and later_fuller_retry
                ):
                    winner_id, loser_id = peer_id, current_id
                elif (
                    str(row.get("accepted_by") or "") == "same_opening_restart"
                    and float(peer.start) > float(current.start)
                    and float(peer.end) - float(peer.start)
                    >= 1.35 * max(0.001, float(current.end) - float(current.start))
                    and len(_tokens(peer.text)) >= len(_tokens(current.text))
                    and _tokens(peer.text)[:5] == _tokens(current.text)[:5]
                    and peer_positive >= 0.80
                    and current_positive - peer_positive <= 0.15 + 1e-9
                    # Overlapping provider windows commonly describe the
                    # same delivery once as a positive audience take and
                    # once as an alternate at the provider's standard 0.85
                    # confidence.  That exact tie is not independent proof
                    # of failure.  The much fuller retry, deterministic
                    # same-opening relation, critical-fact parity, and the
                    # non-deleting terminal evidence below still all have to
                    # agree before membership changes.
                    and peer_negative <= 0.85 + 1e-9
                    and peer_positive + 1e-9 >= peer_negative
                    and complete.get(peer_id) is not False
                    and (
                        usability.get(peer_id) != "UNUSABLE"
                        or (
                            usability.get(current_id) == "UNUSABLE"
                            and not bool((_member_usability.get(peer_id) or {}).get("delete_recommended"))
                            and not bool((_member_usability.get(peer_id) or {}).get("deterministic_unusable"))
                            and not bool((_member_usability.get(peer_id) or {}).get("local_failure_corroborated"))
                        )
                    )
                ):
                    # Two complete same-opening deliveries can carry
                    # different explanatory endings, so ordinary token
                    # overlap is intentionally not treated as proof.  When
                    # deterministic restart evidence already establishes the
                    # contest and delivery evidence is near-tied, prefer the
                    # substantially fuller later delivery instead of keeping
                    # a clean but abbreviated first attempt.
                    winner_id, loser_id = peer_id, current_id
                    fuller_restart_tie = True
        if not winner_id or loser_id not in selected_by_id:
            continue

        winner, loser = all_by_id[winner_id], all_by_id[loser_id]
        winner_positive = _strongest(votes, winner_id, {"winner", "keep"})
        winner_negative = _strongest(votes, winner_id, {"alternate", "failed"})
        if winner_id not in selected_by_id:
            if winner_positive < 0.80:
                continue
            # Overlapping complete-context windows can legitimately produce
            # both a positive and a negative label.  Preserve the negative
            # safety gate unless the positive verdict wins by a clear 0.10
            # margin; treating any >=0.80 negative as an unconditional veto
            # let a weaker selected retry survive a 0.95 keep / 0.85
            # alternate verdict on its fuller peer.
            if (
                winner_negative >= 0.80
                and winner_positive - winner_negative < 0.10 - 1e-9
                and not (fuller_restart_tie and winner_positive + 1e-9 >= winner_negative)
            ):
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

    # Some retry families are expressed as two deterministic edges through
    # the same abandoned attempt: A->B (a short failed correction) and A->C
    # (the complete audience delivery).  B and C therefore compete even when
    # no direct B/C edge was emitted.  Resolve only the unambiguous case: one
    # strong audience winner and a short selected peer that Hybrid strongly
    # classified as failed/non-audience recording debris.
    graph: dict[str, set[str]] = {}
    for row in _deterministic_retry_rows(diagnostics):
        left_id = str(row.get("left_clip_id") or "")
        right_id = str(row.get("right_clip_id") or "")
        if left_id in all_by_id and right_id in all_by_id:
            graph.setdefault(left_id, set()).add(right_id)
            graph.setdefault(right_id, set()).add(left_id)
    visited: set[str] = set()
    for root in graph:
        if root in visited:
            continue
        component: set[str] = set()
        pending = [root]
        while pending:
            clip_id = pending.pop()
            if clip_id in component:
                continue
            component.add(clip_id)
            pending.extend(graph.get(clip_id, ()))
        visited.update(component)
        if component & move:
            continue
        winner_ids = [
            clip_id for clip_id in component
            if _strongest(votes, clip_id, {"winner", "keep"}) >= 0.90
            and _strongest(votes, clip_id, {"alternate", "failed"}) < 0.80
            and _audience_support(diagnostics, clip_id) >= 0.80
            and complete.get(clip_id) is not False
        ]
        if len(winner_ids) != 1:
            continue
        winner_id = winner_ids[0]
        winner = all_by_id[winner_id]
        for loser_id in sorted(component & set(selected_by_id)):
            if loser_id == winner_id:
                continue
            loser = all_by_id[loser_id]
            duration = max(0.0, float(loser.end) - float(loser.start))
            if (
                duration > 4.0
                or _strongest(votes, loser_id, {"alternate", "failed"}) < 0.90
                or _audience_support(diagnostics, loser_id) >= 0.80
            ):
                continue
            move.add(loser_id)
            if winner_id not in selected_by_id:
                add.add(winner_id)
            audit.append({
                "clip_id": loser_id,
                "winner_clip_id": winner_id,
                "reason": "deterministic_retry_component_failed_debris",
                "component_clip_ids": sorted(component),
                "loser_duration_sec": round(duration, 3),
                "loser_negative_confidence": round(
                    _strongest(votes, loser_id, {"alternate", "failed"}), 4,
                ),
                "winner_positive_confidence": round(
                    _strongest(votes, winner_id, {"winner", "keep"}), 4,
                ),
            })

    # A delivery can be grammatically plausible yet still be only the shared
    # opening of the creator's completed retry.  In that case the ranker may
    # prefer the very short delivery because it has fewer motion events, even
    # though the longer peer carries the actual requested/actionable detail.
    # Settle only a deterministic retry component with exactly one selected
    # member and a much fuller peer that begins with the same four words,
    # preserves every numeric/negation marker, overlaps most of the selected
    # member's substantive vocabulary, and has explicit take-judge evidence
    # that it was not deterministically unusable or delete-worthy.  Semantic
    # completeness therefore precedes delivery polish without treating a
    # generic longer take as automatically better.
    _usability, member_usability = _take_judge_usability(diagnostics)
    visited.clear()
    for root in graph:
        if root in visited:
            continue
        component: set[str] = set()
        pending = [root]
        while pending:
            clip_id = pending.pop()
            if clip_id in component:
                continue
            component.add(clip_id)
            pending.extend(graph.get(clip_id, ()))
        visited.update(component)
        if component & move:
            continue
        selected_ids = component & set(selected_by_id)
        if len(selected_ids) != 1:
            continue
        current_id = next(iter(selected_ids))
        current = selected_by_id[current_id]
        current_tokens = _tokens(current.text)
        current_content = _substantive(current.text)
        current_duration = max(0.001, float(current.end) - float(current.start))
        current_positive = _strongest(votes, current_id, {"winner", "keep"})
        current_negative = _strongest(votes, current_id, {"alternate", "failed"})
        current_negative_blocks = (
            current_negative >= 0.80
            and not (
                current_positive >= 0.90
                and current_positive - current_negative >= 0.10 - 1e-9
                and current_negative <= 0.85 + 1e-9
            )
        )
        if (
            len(current_tokens) < 6
            or len(current_content) < 3
            or current_positive < 0.90
            # A standard-confidence negative observation from an
            # overlapping window does not make a punctuation-open, very
            # short winner complete.  Preserve the safety veto when the
            # negative is stronger than that standard value, or when the
            # positive verdict lacks a clear margin.
            or current_negative_blocks
        ):
            continue

        eligible = []
        for peer_id in component - selected_ids:
            peer = all_by_id[peer_id]
            evidence = member_usability.get(peer_id)
            peer_positive = _strongest(votes, peer_id, {"winner", "keep"})
            peer_negative = _strongest(votes, peer_id, {"alternate", "failed"})
            peer_negative_blocks = (
                peer_negative >= 0.80
                and not (
                    peer_positive >= 0.90
                    and peer_positive - peer_negative >= 0.10 - 1e-9
                    and peer_negative <= 0.85 + 1e-9
                )
            )
            if (
                peer.source_asset_id != current.source_asset_id
                or complete.get(peer_id) is False
                or not evidence
                or bool(evidence.get("deterministic_unusable"))
                or bool(evidence.get("delete_recommended"))
                or peer_negative_blocks
            ):
                continue
            peer_tokens = _tokens(peer.text)
            peer_content = _substantive(peer.text)
            peer_duration = max(0.0, float(peer.end) - float(peer.start))
            if current_tokens[:4] != peer_tokens[:4]:
                continue
            overlap = len(current_content & peer_content) / max(1, len(current_content))
            if (
                overlap < 0.65
                or len(peer_tokens) < 1.50 * len(current_tokens)
                or peer_duration < 1.50 * current_duration
                or _critical(current.text) != _critical(peer.text)
            ):
                continue
            eligible.append((len(peer_content), len(peer_tokens), peer_duration, float(peer.start), peer, overlap))
        if not eligible:
            continue
        _richness, _token_count, _duration, _start, winner, overlap = max(
            eligible, key=lambda row: row[:4]
        )
        move.add(current_id)
        add.add(winner.clip_id)
        audit.append({
            "clip_id": current_id,
            "winner_clip_id": winner.clip_id,
            "reason": "deterministic_retry_semantic_superset_dominance",
            "component_clip_ids": sorted(component),
            "opening_token_count": 4,
            "substantive_overlap": round(overlap, 4),
            "loser_token_count": len(current_tokens),
            "winner_token_count": len(_tokens(winner.text)),
            "winner_duration_sec": round(_duration, 3),
            "winner_positive_confidence": round(
                _strongest(votes, winner.clip_id, {"winner", "keep"}), 4,
            ),
            "winner_negative_confidence": round(
                _strongest(votes, winner.clip_id, {"alternate", "failed"}), 4,
            ),
        })
    return move, add, audit


def unmerged_same_opening_retry_resolution(selected, alternates, discarded, diagnostics: dict):
    """Resolve a full later restart that grouping left in another family.

    This is a deliberately narrow lexical fallback for provider/grouping
    variance: four identical opening words, high bidirectional topic overlap,
    comparable-or-fuller duration, identical numeric/negation markers, and an
    already-conflicted current winner.  Among several later starts it chooses
    the richest complete delivery, so an intervening short restart cannot win.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    candidates = tuple((*alternates, *discarded))
    votes = _hybrid_votes(diagnostics)
    complete = _attempt_completeness(diagnostics)
    usability, _member = _take_judge_usability(diagnostics)
    move: set[str] = set()
    add: set[str] = set()
    audit: list[dict] = []

    for current_id, current in selected_by_id.items():
        if _strongest(votes, current_id, {"alternate", "failed"}) < 0.80:
            continue
        current_tokens = _tokens(current.text)
        current_content = _substantive(current.text)
        if len(current_tokens) < 8 or len(current_content) < 5:
            continue
        current_duration = max(0.001, float(current.end) - float(current.start))
        eligible = []
        for candidate in candidates:
            if candidate.source_asset_id != current.source_asset_id:
                continue
            if not (float(current.end) < float(candidate.start) <= float(current.end) + 30.0):
                continue
            if complete.get(candidate.clip_id) is False or usability.get(candidate.clip_id) == "UNUSABLE":
                continue
            candidate_tokens = _tokens(candidate.text)
            candidate_content = _substantive(candidate.text)
            if len(candidate_tokens) < 8 or current_tokens[:4] != candidate_tokens[:4]:
                continue
            overlap = len(current_content & candidate_content) / max(
                1, min(len(current_content), len(candidate_content))
            )
            candidate_duration = max(0.0, float(candidate.end) - float(candidate.start))
            if overlap < 0.70 or candidate_duration < 0.95 * current_duration:
                continue
            if len(candidate_tokens) < 0.90 * len(current_tokens):
                continue
            if _critical(current.text) != _critical(candidate.text):
                continue
            eligible.append((len(candidate_content), candidate_duration, float(candidate.start), candidate, overlap))
        if not eligible:
            continue
        _richness, _duration, _start, winner, overlap = max(eligible, key=lambda row: row[:3])
        move.add(current_id)
        add.add(winner.clip_id)
        audit.append({
            "clip_id": current_id,
            "winner_clip_id": winner.clip_id,
            "reason": "ungrouped_same_opening_full_retry_resolution",
            "opening_token_count": 4,
            "substantive_overlap": round(overlap, 4),
            "loser_conflict_confidence": round(
                _strongest(votes, current_id, {"alternate", "failed"}), 4
            ),
            "winner_duration_sec": round(_duration, 3),
        })
    return move, add, audit


def unmerged_same_opening_retry_resolution(selected, alternates, discarded, diagnostics: dict):
    """Resolve a full later restart that grouping left in another family.

    This is a deliberately narrow lexical fallback for provider/grouping
    variance: four identical opening words, high bidirectional topic overlap,
    comparable-or-fuller duration, identical numeric/negation markers, and an
    already-conflicted current winner.  Among several later starts it chooses
    the richest complete delivery, so an intervening short restart cannot win.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    candidates = tuple((*alternates, *discarded))
    votes = _hybrid_votes(diagnostics)
    complete = _attempt_completeness(diagnostics)
    usability, member = _take_judge_usability(diagnostics)
    delete_recommended = _semantic_delete_recommended_ids(diagnostics)
    move: set[str] = set()
    add: set[str] = set()
    audit: list[dict] = []

    for current_id, current in selected_by_id.items():
        if _strongest(votes, current_id, {"alternate", "failed"}) < 0.75:
            continue
        current_tokens = _tokens(current.text)
        current_content = _substantive(current.text)
        if len(current_tokens) < 7 or len(current_content) < 3:
            continue
        current_duration = max(0.001, float(current.end) - float(current.start))
        eligible = []
        for candidate in candidates:
            if candidate.source_asset_id != current.source_asset_id:
                continue
            if candidate.clip_id in delete_recommended:
                continue
            if not (float(current.end) < float(candidate.start) <= float(current.end) + 30.0):
                continue
            candidate_positive = _strongest(votes, candidate.clip_id, {"winner", "keep"})
            candidate_negative = _strongest(votes, candidate.clip_id, {"alternate", "failed"})
            unusable_but_positive = (
                usability.get(candidate.clip_id) == "UNUSABLE"
                and not bool((member.get(candidate.clip_id) or {}).get("delete_recommended"))
                and candidate_positive >= 0.90
                and candidate_positive - candidate_negative >= 0.10 - 1e-9
            )
            if complete.get(candidate.clip_id) is False or (
                usability.get(candidate.clip_id) == "UNUSABLE" and not unusable_but_positive
            ):
                continue
            candidate_tokens = _tokens(candidate.text)
            candidate_content = _substantive(candidate.text)
            if len(candidate_tokens) < 8 or current_tokens[:4] != candidate_tokens[:4]:
                continue
            overlap = len(current_content & candidate_content) / max(
                1, min(len(current_content), len(candidate_content))
            )
            candidate_duration = max(0.0, float(candidate.end) - float(candidate.start))
            if overlap < 0.70 or candidate_duration < 0.95 * current_duration:
                continue
            if len(candidate_tokens) < 0.90 * len(current_tokens):
                continue
            if _critical(current.text) != _critical(candidate.text):
                continue
            eligible.append((len(candidate_content), candidate_duration, float(candidate.start), candidate, overlap))
        if not eligible:
            continue
        _richness, _duration, _start, winner, overlap = max(eligible, key=lambda row: row[:3])
        move.add(current_id)
        add.add(winner.clip_id)
        audit.append({
            "clip_id": current_id,
            "winner_clip_id": winner.clip_id,
            "reason": "ungrouped_same_opening_full_retry_resolution",
            "opening_token_count": 4,
            "substantive_overlap": round(overlap, 4),
            "loser_conflict_confidence": round(
                _strongest(votes, current_id, {"alternate", "failed"}), 4
            ),
            "winner_duration_sec": round(_duration, 3),
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
    selected_by_id = {clip.clip_id: clip for clip in selected}
    continuation_units: list[tuple[tuple[object, ...], str]] = []
    equivalence = diagnostics.get("semantic_idea_equivalence") or {}
    for row in equivalence.get("continuation_merges") or ():
        if not isinstance(row, dict) or str(row.get("accepted_by") or "") != "sentence_continuation":
            continue
        ids = (str(row.get("left_clip_id") or ""), str(row.get("right_clip_id") or ""))
        if all(clip_id in selected_by_id for clip_id in ids):
            clips = tuple(selected_by_id[clip_id] for clip_id in ids)
            continuation_units.append((clips, " ".join(clip.text for clip in clips)))
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
            ordinary_positive_bridge = positive >= 0.80 and positive + 1e-9 >= negative
            # Per-fragment Hybrid may call a grammatically open bridge a
            # performance failure even when it is plainly audience speech and
            # the selected next delivery closes it.  Override that fragment-
            # local verdict only when a later deterministic continuation unit
            # independently repeats the same protected claim.  The later unit
            # then supplies a lossless duplicate witness; the ordinary chain
            # collapse below still has to prove the earlier realization covers
            # it before anything is removed.
            protected = _critical(bridge.text)
            combined_content = _substantive(bridge.text + " " + right.text)
            duplicate_unit = next((
                (clips, text) for clips, text in continuation_units
                if min(float(clip.start) for clip in clips) > float(right.end)
                and protected
                and protected.issubset(_critical(text))
                and len(_substantive(text)) >= 2
                and len(combined_content & _substantive(text)) / len(_substantive(text)) >= 0.65
            ), None)
            witnessed_audience_bridge = (
                not ordinary_positive_bridge
                and _audience_support(diagnostics, bridge.clip_id) >= 0.80
                and duplicate_unit is not None
            )
            if not (ordinary_positive_bridge or witnessed_audience_bridge):
                continue
            add.add(bridge.clip_id)
            audit.append({
                "clip_id": bridge.clip_id,
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "reason": (
                    "missing_positive_continuation_bridge_restored"
                    if ordinary_positive_bridge
                    else "audience_continuation_bridge_restored_from_duplicate_witness"
                ),
                "left_gap_sec": round(left_gap, 3),
                "right_gap_sec": round(right_gap, 3),
                "terminal_token": tokens[-1],
                "positive_confidence": round(positive, 4),
                "negative_confidence": round(negative, 4),
                "duplicate_witness_clip_ids": (
                    [clip.clip_id for clip in duplicate_unit[0]] if duplicate_unit else []
                ),
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


def nearby_contained_selected_realization_ids(selected):
    """Remove a later nearby statement wholly contained in an earlier one.

    This is lexical-set containment, not topical similarity: every
    substantive token and every numeric/negation marker in the later clip
    must already occur in the earlier selected delivery.  The earlier clip
    must also be materially richer.  It closes the grouping shape where a
    compound delivery and one of its later restated clauses land in separate
    families, without suppressing a later statement that adds any fact.
    """
    ordered = tuple(sorted(
        selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)
    ))
    move: set[str] = set()
    audit: list[dict] = []
    for index, later in enumerate(ordered):
        later_content = _substantive(later.text)
        if len(later_content) < 5:
            continue
        for earlier in reversed(ordered[max(0, index - 4):index]):
            if earlier.source_asset_id != later.source_asset_id:
                continue
            if not (0 <= float(later.start) - float(earlier.end) <= 30.0):
                continue
            earlier_content = _substantive(earlier.text)
            if len(earlier_content) < 1.5 * len(later_content):
                continue
            if not later_content.issubset(earlier_content):
                continue
            if not _critical(later.text).issubset(_critical(earlier.text)):
                continue
            move.add(later.clip_id)
            audit.append({
                "clip_id": later.clip_id,
                "winner_clip_id": earlier.clip_id,
                "reason": "later_selected_realization_fully_contained_in_nearby_delivery",
                "substantive_token_count": len(later_content),
                "winner_substantive_token_count": len(earlier_content),
            })
            break
    return move, audit


def abandoned_negated_restart_ids(selected, diagnostics: dict):
    """Remove a short spoken correction immediately before its clean retry.

    A creator may begin a sentence, negate that wording ("..., no." /
    "..., not."), pause, and restart with the completed delivery.  This path
    requires the literal terminal negation, the same multiword opening, a
    materially fuller nearby retry, and independent AV evidence that the
    short attempt is mixed/recording-process while the retry is audience
    speech.  It does not infer corrections from negation alone.
    """
    ordered = tuple(sorted(
        selected, key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id)
    ))
    complete = _attempt_completeness(diagnostics)
    move: set[str] = set()
    audit: list[dict] = []
    for current, retry in zip(ordered, ordered[1:]):
        if current.source_asset_id != retry.source_asset_id:
            continue
        current_tokens = _tokens(current.text)
        retry_tokens = _tokens(retry.text)
        current_duration = max(0.001, float(current.end) - float(current.start))
        retry_duration = max(0.0, float(retry.end) - float(retry.start))
        gap = float(retry.start) - float(current.end)
        if not (3 <= len(current_tokens) <= 6 and current_tokens[-1] in _NEGATION):
            continue
        if len(retry_tokens) < 9 or current_tokens[:2] != retry_tokens[:2]:
            continue
        if gap < 0.0 or gap > 8.0 or retry_duration < 2.0 * current_duration:
            continue
        if complete.get(retry.clip_id) is False:
            continue
        current_mixed = _clean_cut_role_support(
            diagnostics, current.clip_id, {"mixed", "recording_process"}
        )
        retry_audience = max(
            _audience_support(diagnostics, retry.clip_id),
            _clean_cut_role_support(diagnostics, retry.clip_id, {"audience"}),
        )
        if current_mixed < 0.80 or retry_audience < 0.80:
            continue
        move.add(current.clip_id)
        audit.append({
            "clip_id": current.clip_id,
            "winner_clip_id": retry.clip_id,
            "reason": "terminal_negation_abandoned_restart",
            "opening_token_count": 2,
            "gap_sec": round(gap, 3),
            "loser_mixed_confidence": round(current_mixed, 4),
            "winner_audience_confidence": round(retry_audience, 4),
        })
    return move, audit


_ANAPHORIC_FRAGMENT_OPENINGS = frozenset({
    ("era", "como"), ("fue", "como"), ("es", "como"),
    ("was", "like"), ("is", "like"), ("it", "was"),
})


def orphaned_anaphoric_retry_fragment_ids(selected, alternates, discarded, diagnostics: dict):
    """Remove a short anaphoric fragment orphaned from a replaced retry.

    The fragment is removable only when it sits immediately before an
    unselected continuation that has an explicit high-confidence equivalence
    to a later selected, materially fuller winner.  This lets an existing
    semantic decision cover a segmentation split without treating arbitrary
    short phrases as duplicates.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    unselected_by_id = {clip.clip_id: clip for clip in (*alternates, *discarded)}
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
        left_id = str(row.get("left_clip_id") or "")
        right_id = str(row.get("right_clip_id") or "")
        for proxy_id, winner_id in ((left_id, right_id), (right_id, left_id)):
            proxy = unselected_by_id.get(proxy_id)
            winner = selected_by_id.get(winner_id)
            if proxy is None or winner is None or proxy.source_asset_id != winner.source_asset_id:
                continue
            if not (float(proxy.end) <= float(winner.start) + 1e-3):
                continue
            winner_content = _substantive(winner.text)
            if len(winner_content) < 3:
                continue
            for fragment in selected:
                if fragment.clip_id == winner_id or fragment.source_asset_id != proxy.source_asset_id:
                    continue
                fragment_tokens = _tokens(fragment.text)
                fragment_content = _substantive(fragment.text)
                if (
                    len(fragment_tokens) > 7
                    or tuple(fragment_tokens[:2]) not in _ANAPHORIC_FRAGMENT_OPENINGS
                    or _critical(fragment.text)
                    or not fragment_content
                ):
                    continue
                leading_gap = float(proxy.start) - float(fragment.end)
                winner_gap = float(winner.start) - float(proxy.end)
                if leading_gap < 0.0 or leading_gap > 3.0 or winner_gap < 0.0 or winner_gap > 6.0:
                    continue
                covered = fragment_content & (_substantive(proxy.text) | winner_content)
                coverage = len(covered) / len(fragment_content)
                if coverage < 0.50 or len(winner_content) < 2 * len(fragment_content):
                    continue
                move.add(fragment.clip_id)
                audit.append({
                    "clip_id": fragment.clip_id,
                    "proxy_clip_id": proxy_id,
                    "winner_clip_id": winner_id,
                    "reason": "orphaned_anaphoric_fragment_of_confirmed_retry",
                    "equivalence_confidence": round(confidence, 4),
                    "substantive_coverage": round(coverage, 4),
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
    usability, _member_usability = _take_judge_usability(diagnostics)
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
        elif (
            confidence >= 0.90
            and right_positive >= 0.90
            and right_positive - left_positive >= 0.02 - 1e-9
            and left_negative < 0.80
            and right_negative < 0.80
        ):
            loser_id, winner_id = left_id, right_id
        elif (
            confidence >= 0.90
            and left_positive >= 0.90
            and left_positive - right_positive >= 0.02 - 1e-9
            and left_negative < 0.80
            and right_negative < 0.80
        ):
            loser_id, winner_id = right_id, left_id
        elif (
            confidence >= 0.95
            and left_positive >= 0.90
            and right_positive >= 0.90
            and left_negative < 0.80
            and right_negative < 0.80
        ):
            # Both selected winners are already proven equivalent and equally
            # audience-safe.  Resolve the tie once, using the normal retake
            # convention (later delivery), instead of rendering both.
            left, right = selected_by_id[left_id], selected_by_id[right_id]
            if left.source_asset_id == right.source_asset_id:
                winner, loser = (right, left) if float(right.start) > float(left.start) else (left, right)
                winner_id, loser_id = winner.clip_id, loser.clip_id
        elif confidence >= 0.90:
            # A direct semantic-equivalence verdict plus the take judge's
            # terminal UNUSABLE finding is enough to settle an otherwise
            # conservative positive-vote tie.  This is the cross-family
            # shape where a local family had no good alternative but a later,
            # independently selected equivalent delivery exists globally.
            left, right = selected_by_id[left_id], selected_by_id[right_id]
            if left.source_asset_id == right.source_asset_id:
                if (
                    usability.get(left_id) == "UNUSABLE"
                    and usability.get(right_id) != "UNUSABLE"
                    and (right_positive >= 0.90 or right_negative < 0.80)
                    and float(right.start) > float(left.start)
                ):
                    loser_id, winner_id = left_id, right_id
                elif (
                    usability.get(right_id) == "UNUSABLE"
                    and usability.get(left_id) != "UNUSABLE"
                    and (left_positive >= 0.90 or left_negative < 0.80)
                    and float(left.start) > float(right.start)
                ):
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


def failed_retry_component_ids(selected, alternates, discarded, diagnostics: dict):
    """Remove a failed wrong-take when its retry component has a clean winner.

    Restart evidence is transitive: an abandoned start can be linked to a
    wrong-take fragment, while that same abandoned start is linked to the
    completed retry.  Looking only at direct pairs lets the middle failed
    fragment survive even though the component already contains the final
    delivery.  This function requires all three independent signals before
    changing membership: deterministic retry edges, a high-confidence
    Hybrid failure, and terminal Best-Take unusability/delete evidence.
    """
    all_by_id = {clip.clip_id: clip for clip in (*selected, *alternates, *discarded)}
    selected_by_id = {clip.clip_id: clip for clip in selected}
    votes = _hybrid_votes(diagnostics)
    complete = _attempt_completeness(diagnostics)
    usability, member_usability = _take_judge_usability(diagnostics)
    graph: dict[str, set[str]] = {}
    wrong_take_peers: dict[str, set[str]] = {}
    for row in _deterministic_retry_rows(diagnostics):
        left_id = str(row.get("left_clip_id") or "")
        right_id = str(row.get("right_clip_id") or "")
        if left_id in all_by_id and right_id in all_by_id:
            graph.setdefault(left_id, set()).add(right_id)
            graph.setdefault(right_id, set()).add(left_id)
            if (
                str(row.get("accepted_by") or "") == "multimodal_corroborated_retry"
                and str(row.get("corroborating_event_kind") or "") == "wrong_take"
            ):
                wrong_take_peers.setdefault(left_id, set()).add(right_id)
                wrong_take_peers.setdefault(right_id, set()).add(left_id)

    move: set[str] = set()
    audit: list[dict] = []
    for clip_id, clip in selected_by_id.items():
        evidence = member_usability.get(clip_id) or {}
        failed_confidence = _strongest(votes, clip_id, {"failed"})
        positive_confidence = _strongest(votes, clip_id, {"winner", "keep"})
        corroborated_wrong_take = bool(wrong_take_peers.get(clip_id))
        duration = max(0.0, float(clip.end) - float(clip.start))
        short_wrong_take_tail = corroborated_wrong_take and duration <= 2.5
        terminal_delete = (
            failed_confidence >= 0.90
            and positive_confidence < 0.80
            and bool(evidence.get("deterministic_unusable"))
            and bool(evidence.get("delete_recommended"))
        )
        if not (terminal_delete or corroborated_wrong_take):
            continue
        if usability.get(clip_id) != "UNUSABLE" and not short_wrong_take_tail:
            continue

        component = {clip_id}
        frontier = [clip_id]
        while frontier:
            current = frontier.pop()
            for peer_id in graph.get(current, ()):
                if peer_id not in component:
                    component.add(peer_id)
                    frontier.append(peer_id)

        winner_id = ""
        for peer_id in component:
            if peer_id == clip_id or peer_id not in selected_by_id:
                continue
            peer = selected_by_id[peer_id]
            if peer.source_asset_id != clip.source_asset_id:
                continue
            if complete.get(peer_id) is False:
                continue
            winner_positive = _strongest(votes, peer_id, {"winner", "keep"})
            audience_support = _audience_support(diagnostics, peer_id)
            if usability.get(peer_id) == "UNUSABLE" and not (
                short_wrong_take_tail
                and winner_positive >= 0.80
                and audience_support >= 0.85
            ):
                continue
            if winner_positive < (0.80 if short_wrong_take_tail else 0.90):
                continue
            if (
                short_wrong_take_tail
                and winner_positive < 0.90
                and audience_support < 0.85
            ):
                continue
            winner_id = peer_id
            break
        if not winner_id:
            continue

        move.add(clip_id)
        audit.append({
            "clip_id": clip_id,
            "winner_clip_id": winner_id,
            "reason": "failed_unusable_retry_component_yields_to_complete_winner",
            "component_clip_ids": sorted(component),
            "failed_confidence": round(failed_confidence, 4),
            "multimodal_wrong_take_corroborated": corroborated_wrong_take,
            "short_wrong_take_tail": short_wrong_take_tail,
            "winner_positive_confidence": round(
                _strongest(votes, winner_id, {"winner", "keep"}), 4
            ),
        })
    return move, audit


def dependent_opening_retry_resolution(selected, alternates, discarded, diagnostics: dict):
    """Replace a selected orphan clause with its complete retry-family peer.

    Provider performance judgments cannot make a delivery beginning with a
    bare dependent opener structurally self-contained.  This repair is
    intentionally confined to an existing take-judge family and requires the
    selected delivery's opening sequence and every critical marker to survive
    in a complete peer.  Explicit delete/unusable evidence still fails closed.
    """
    selected_by_id = {clip.clip_id: clip for clip in selected}
    all_by_id = {clip.clip_id: clip for clip in (*selected, *alternates, *discarded)}
    complete = _attempt_completeness(diagnostics)
    move: set[str] = set()
    add: set[str] = set()
    audit: list[dict] = []

    for group in diagnostics.get("take_judge_groups") or ():
        if not isinstance(group, dict):
            continue
        member_ids = [
            str(row.get("clip_id") or "")
            for row in group.get("ranked") or ()
            if isinstance(row, dict)
        ]
        member_evidence = group.get("member_usability") or {}
        for current_id in member_ids:
            current = selected_by_id.get(current_id)
            if current is None:
                continue
            current_tokens = _tokens(current.text)
            if not current_tokens or current_tokens[0] not in _DEPENDENT_OPENERS:
                continue
            opening = current_tokens[: min(5, len(current_tokens))]
            current_content = _substantive(current.text)
            eligible = []
            for peer_id in member_ids:
                if peer_id == current_id or peer_id in selected_by_id:
                    continue
                peer = all_by_id.get(peer_id)
                if peer is None or peer.source_asset_id != current.source_asset_id:
                    continue
                peer_tokens = _tokens(peer.text)
                if not peer_tokens or peer_tokens[0] in _DEPENDENT_OPENERS:
                    continue
                evidence = member_evidence.get(peer_id) or {}
                if (
                    evidence.get("delete_recommended") is True
                    or evidence.get("deterministic_unusable") is True
                    or complete.get(peer_id) is False
                ):
                    continue
                if not _is_contiguous_subsequence(opening, peer_tokens):
                    continue
                peer_content = _substantive(peer.text)
                coverage = len(current_content & peer_content) / max(1, len(current_content))
                if coverage < 0.80:
                    continue
                if not _critical(current.text).issubset(_critical(peer.text)):
                    continue
                eligible.append((coverage, len(peer_tokens), float(peer.end) - float(peer.start), peer))
            if not eligible:
                continue
            coverage, _token_count, _duration, winner = max(eligible, key=lambda row: row[:3])
            move.add(current_id)
            add.add(winner.clip_id)
            audit.append({
                "clip_id": current_id,
                "winner_clip_id": winner.clip_id,
                "reason": "dependent_opening_yields_to_complete_family_peer",
                "opening_tokens": list(opening),
                "substantive_coverage": round(coverage, 4),
            })
    return move, add, audit


def redundant_selected_restatement_ids(selected, diagnostics: dict):
    """Remove a later provider-rejected restatement already fully delivered."""
    ordered = tuple(sorted(
        selected,
        key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id),
    ))
    votes = _hybrid_votes(diagnostics)
    move: set[str] = set()
    audit: list[dict] = []
    for later_index, later in enumerate(ordered):
        later_negative = _strongest(votes, later.clip_id, {"alternate", "failed"})
        later_content = _substantive(later.text)
        later_critical = _critical(later.text)
        if later_negative < 0.80 or len(later_content) < 2 or not later_critical:
            continue
        for prior in reversed(ordered[max(0, later_index - 4):later_index]):
            if prior.source_asset_id != later.source_asset_id:
                continue
            gap = float(later.start) - float(prior.end)
            if gap < 0.0 or gap > 45.0:
                continue
            prior_positive = _strongest(votes, prior.clip_id, {"winner", "keep"})
            prior_negative = _strongest(votes, prior.clip_id, {"alternate", "failed"})
            if prior_positive < 0.90 or prior_positive - prior_negative < 0.05 - 1e-9:
                continue
            prior_content = _substantive(prior.text)
            coverage = len(later_content & prior_content) / max(1, len(later_content))
            if coverage < 0.80 or not later_critical.issubset(_critical(prior.text)):
                continue
            if float(later.end) - float(later.start) > float(prior.end) - float(prior.start):
                continue
            move.add(later.clip_id)
            audit.append({
                "clip_id": later.clip_id,
                "winner_clip_id": prior.clip_id,
                "reason": "provider_rejected_restatement_already_fully_delivered",
                "substantive_coverage": round(coverage, 4),
                "critical_markers": sorted(later_critical),
                "source_gap_sec": round(gap, 3),
                "loser_negative_confidence": round(later_negative, 4),
                "winner_positive_confidence": round(prior_positive, 4),
            })
            break
    return move, audit


def borderline_subspan_reconstruction_ids(selected, alternates, discarded, diagnostics: dict):
    """Restore a safe prefix when authority selected only its sibling suffix.

    Attempt reconstruction explicitly records these rows when a complete
    parent was split into two independently usable borderline subspans.  If
    the suffix survives authority but the adjacent prefix does not, retaining
    only the suffix can silently drop the parent's opening claim.  Rejoin the
    pair only with full lexical coverage, tight source adjacency, audience
    evidence, and no terminal delete/unusable finding for the prefix.
    """
    selected_ids = {clip.clip_id for clip in selected}
    all_by_id = {clip.clip_id: clip for clip in (*selected, *alternates, *discarded)}
    complete = _attempt_completeness(diagnostics)
    _status, member_usability = _take_judge_usability(diagnostics)
    add: set[str] = set()
    audit: list[dict] = []
    reconstruction = diagnostics.get("attempt_reconstruction") or {}
    for row in reconstruction.get("preserved_borderline_subspans") or ():
        if not isinstance(row, dict):
            continue
        parent_id = str(row.get("parent_clip_id") or "")
        prefix_id = str(row.get("prefix_clip_id") or "")
        suffix_id = str(row.get("suffix_clip_id") or "")
        if suffix_id not in selected_ids or prefix_id in selected_ids:
            continue
        parent = all_by_id.get(parent_id)
        prefix = all_by_id.get(prefix_id)
        suffix = all_by_id.get(suffix_id)
        if parent is None or prefix is None or suffix is None:
            continue
        if len({parent.source_asset_id, prefix.source_asset_id, suffix.source_asset_id}) != 1:
            continue
        gap = float(suffix.start) - float(prefix.end)
        if float(prefix.start) < float(parent.start) - 1e-3 or gap < 0.0 or gap > 0.8:
            continue
        if float(suffix.end) > float(parent.end) + 1e-3 or complete.get(parent_id) is False:
            continue
        evidence = member_usability.get(prefix_id) or {}
        if evidence.get("delete_recommended") is True or evidence.get("deterministic_unusable") is True:
            continue
        if _audience_support(diagnostics, prefix_id) < 0.80:
            continue
        parent_content = _substantive(parent.text)
        combined_content = _substantive(prefix.text + " " + suffix.text)
        coverage = len(parent_content & combined_content) / max(1, len(parent_content))
        if coverage < 0.90 or not _critical(parent.text).issubset(
            _critical(prefix.text + " " + suffix.text)
        ):
            continue
        add.add(prefix_id)
        audit.append({
            "clip_id": prefix_id,
            "suffix_clip_id": suffix_id,
            "parent_clip_id": parent_id,
            "reason": "complete_parent_borderline_prefix_restored",
            "source_gap_sec": round(gap, 3),
            "parent_content_coverage": round(coverage, 4),
        })
    return add, audit


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


def apply_selection_conflicted_bridge_guard(draft, *, allow_membership_additions: bool = True):
    """Reconcile proven conflicted/redundant final membership.

    ``allow_membership_additions`` is disabled after authoritative realization
    resolution.  At that boundary this guard may still remove independently
    proven duplicate/failed material, but it must not restore an alternate or
    discarded realization, nor perform a replacement whose winning peer would
    have to be restored.  The authoritative resolver is the sole owner of
    additions to the final KEEP set.
    """
    diagnostics = dict(draft.diagnostics or {})
    bridge_ids, bridge_audit = conflicted_redundant_bridge_ids(draft.selected, diagnostics)
    duplicate_ids, duplicate_audit = confirmed_selected_duplicate_ids(draft.selected, diagnostics)
    incomplete_ids, incomplete_audit = terminally_incomplete_selected_ids(draft.selected, diagnostics)
    failed_retry_ids, failed_retry_audit = failed_retry_component_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    dependent_ids, dependent_add_ids, dependent_audit = dependent_opening_retry_resolution(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    restatement_ids, restatement_audit = redundant_selected_restatement_ids(
        draft.selected, diagnostics
    )
    borderline_add_ids, borderline_audit = borderline_subspan_reconstruction_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    retry_ids, retry_add_ids, retry_audit = deterministic_retry_resolution(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    unmerged_retry_ids, unmerged_retry_add_ids, unmerged_retry_audit = (
        unmerged_same_opening_retry_resolution(
            draft.selected, draft.alternates, draft.discarded, diagnostics
        )
    )
    proxy_ids, proxy_audit = contained_proxy_duplicate_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    contained_ids, contained_audit = nearby_contained_selected_realization_ids(draft.selected)
    abandoned_ids, abandoned_audit = abandoned_negated_restart_ids(draft.selected, diagnostics)
    anaphoric_ids, anaphoric_audit = orphaned_anaphoric_retry_fragment_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )
    continuation_add_ids, continuation_add_audit = missing_continuation_bridge_ids(
        draft.selected, draft.alternates, draft.discarded, diagnostics
    )

    all_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.alternates, *draft.discarded)}
    effective_continuation_add_ids = (
        continuation_add_ids if allow_membership_additions else set()
    )
    # Build the continuation-chain proof from the membership that will
    # actually survive the other independently proven removals in this same
    # pass.  Recording every pre-pass neighbour as a required witness made a
    # valid coverage proof stale whenever one of those neighbours was itself
    # a duplicate removed concurrently.
    pre_chain_move_ids = (
        bridge_ids | duplicate_ids | incomplete_ids | failed_retry_ids
        | restatement_ids | proxy_ids | contained_ids | abandoned_ids | anaphoric_ids
    )
    if allow_membership_additions:
        pre_chain_move_ids |= dependent_ids | retry_ids | unmerged_retry_ids
    provisional = tuple(
        (
            clip for clip in draft.selected
            if clip.clip_id not in pre_chain_move_ids
        )
    ) + tuple(
        (
            all_by_id[clip_id] for clip_id in effective_continuation_add_ids
            if clip_id in all_by_id
        )
    )
    chain_ids, chain_audit = redundant_continuation_chain_ids(provisional, diagnostics)

    independent_move_ids = (
        bridge_ids | duplicate_ids | incomplete_ids | failed_retry_ids
        | restatement_ids | proxy_ids | contained_ids | chain_ids
        | abandoned_ids | anaphoric_ids
    )
    replacement_move_ids = dependent_ids | retry_ids | unmerged_retry_ids
    # A later authority stage can resurrect a loser that this guard already
    # removed.  Reapply only removal-only proofs whose selected winner still
    # exists and is not itself being removed now.  This never restores or
    # substitutes membership, so resolver ownership of additions remains
    # intact.
    replayable_reasons = {
        "direct_equivalence_confirmed_final_winner",
        "failed_unusable_retry_component_yields_to_complete_winner",
        "deterministic_retry_component_failed_debris",
        "provider_rejected_restatement_already_fully_delivered",
        "later_selected_realization_fully_contained_in_nearby_delivery",
        "contained_fragment_of_confirmed_duplicate",
        "terminal_negation_abandoned_restart",
        "orphaned_anaphoric_fragment_of_confirmed_retry",
    }
    selected_ids_now = {clip.clip_id for clip in draft.selected}
    already_planned = independent_move_ids | (
        replacement_move_ids if allow_membership_additions else set()
    )
    replayed_independent_ids = {
        str(row.get("clip_id") or row.get("removed_clip_id") or "")
        for row in (diagnostics.get("selection_conflicted_bridge_guard") or ())
        if isinstance(row, dict)
        and str(row.get("reason") or "") in replayable_reasons
        and str(row.get("clip_id") or row.get("removed_clip_id") or "") in selected_ids_now
        and str(row.get("winner_clip_id") or "") in selected_ids_now
        and str(row.get("winner_clip_id") or "") not in already_planned
    }
    independent_move_ids |= replayed_independent_ids
    move_ids = independent_move_ids | (
        replacement_move_ids if allow_membership_additions else set()
    )
    requested_add_ids = (
        retry_add_ids | unmerged_retry_add_ids | continuation_add_ids | dependent_add_ids
        | borderline_add_ids
    )
    add_ids = (requested_add_ids - move_ids) if allow_membership_additions else set()
    audit = (
        bridge_audit + duplicate_audit + incomplete_audit + failed_retry_audit
        + dependent_audit + restatement_audit
        + borderline_audit
        + retry_audit + unmerged_retry_audit
        + proxy_audit + contained_audit + continuation_add_audit + chain_audit
        + abandoned_audit + anaphoric_audit
    )
    if not move_ids and not add_ids:
        if requested_add_ids and not allow_membership_additions:
            diagnostics["selection_conflicted_bridge_guard_post_authority"] = {
                "membership_additions_allowed": False,
                "suppressed_add_clip_ids": sorted(requested_add_ids),
            }
            return replace(draft, diagnostics=diagnostics)
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

    # Preserve earlier audited membership proofs if this guard is invoked
    # again on an already-guarded draft.  StoryValidator consumes those
    # proofs; replacing them with only the second pass's delta would make an
    # otherwise idempotent reapplication reopen resolved coverage findings.
    prior_audit = list(diagnostics.get("selection_conflicted_bridge_guard") or ())
    combined_audit = prior_audit[:]
    for row in audit:
        if row not in combined_audit:
            combined_audit.append(row)
    diagnostics["selection_conflicted_bridge_guard"] = combined_audit
    if not allow_membership_additions:
        diagnostics["selection_conflicted_bridge_guard_post_authority"] = {
            "membership_additions_allowed": False,
            "suppressed_add_clip_ids": sorted(requested_add_ids),
        }
    return replace(
        draft,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=tuple(clip for clip in draft.discarded if clip.clip_id not in add_ids),
        diagnostics=diagnostics,
    )
