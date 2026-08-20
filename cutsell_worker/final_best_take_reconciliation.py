"""Final global Best Take reconciliation for Clean Cut.

This module restores an editorial invariant that existed in the earlier EditDNA pipeline:
competing deliveries of the same audience-facing idea must end as one retry family with
exactly one selected winner.  Earlier cleanup and Hybrid passes may split siblings across
different deterministic groups, so this pass operates on the surviving global take set
after semantic cleanup but before temporal refinement/composition.

Important contract:
- this pass groups; it does not delete speech;
- losers remain in ``kept`` and therefore remain available as Draft alternates / Swap Take;
- unique information stays fail-open;
- a retry variant split into an open prefix + immediate continuation can be reconciled
  against a complete competing delivery as one logical variant;
- semantic winner evidence is preferred only when it is unique and strong.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Iterable

from .contracts import CandidateTake, RankedTake, TakeGroup

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "what", "with", "you", "your",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "mi", "mis", "o",
    "para", "pero", "por", "porque", "que", "se", "si", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})
_CRITICAL = frozenset({"no", "not", "never", "nunca", "sin", "without"})
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    return {
        token for token in _tokens(text)
        if token in _CRITICAL or any(ch.isdigit() for ch in token)
    }


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, right.start - left.end)
    if right.end <= left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _coverage(subject_text: str, peer_text: str) -> tuple[int, float]:
    subject = _content(subject_text)
    peer = _content(peer_text)
    if not subject:
        return 0, 0.0
    shared = len(subject & peer)
    return shared, shared / max(1, len(subject))


def _critical_preserved(subject_text: str, peer_text: str) -> bool:
    return _critical(subject_text).issubset(_critical(peer_text))


def _open_delivery(take: CandidateTake) -> bool:
    text = str(take.text or "").strip()
    return bool(text) and (not take.complete_idea or not _SENTENCE_END_RE.search(text))


def _immediate_continuation(
    take: CandidateTake,
    ordered: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 3.0,
) -> CandidateTake | None:
    try:
        index = ordered.index(take)
    except ValueError:
        return None
    if index + 1 >= len(ordered):
        return None
    nxt = ordered[index + 1]
    if nxt.source_asset_id != take.source_asset_id:
        return None
    if nxt.start < take.end or nxt.start - take.end > maximum_gap_sec:
        return None
    return nxt


def _same_retry_idea(
    candidate_text: str,
    authoritative_text: str,
    *,
    candidate_duration_sec: float,
) -> tuple[bool, dict]:
    shared, candidate_coverage = _coverage(candidate_text, authoritative_text)
    reverse_shared, authoritative_coverage = _coverage(authoritative_text, candidate_text)
    critical_ok = _critical_preserved(candidate_text, authoritative_text)
    if candidate_duration_sec <= 6.0:
        enough = shared >= 2 and candidate_coverage >= 0.50
    elif candidate_duration_sec <= 14.0:
        enough = shared >= 4 and candidate_coverage >= 0.45
    else:
        enough = shared >= 5 and candidate_coverage >= 0.42
    # A competing delivery can be a reformulation, so symmetric containment need not be
    # extreme; it still must explain a meaningful fraction of the authoritative take.
    symmetric = authoritative_coverage >= 0.28 or shared >= 7
    return bool(enough and symmetric and critical_ok), {
        "shared": shared,
        "candidate_coverage": round(candidate_coverage, 4),
        "authoritative_coverage": round(authoritative_coverage, 4),
        "critical_preserved": critical_ok,
    }


def _variant_matches_authoritative(
    candidate: CandidateTake,
    authoritative: CandidateTake,
    ordered: tuple[CandidateTake, ...],
) -> tuple[bool, tuple[CandidateTake, ...], dict]:
    if candidate.source_asset_id != authoritative.source_asset_id:
        return False, (), {}
    if _gap(candidate, authoritative) > 60.0:
        return False, (), {}

    direct, evidence = _same_retry_idea(
        candidate.text,
        authoritative.text,
        candidate_duration_sec=candidate.duration_sec,
    )
    if direct:
        return True, (candidate,), {"variant_shape": "single", **evidence}

    if not _open_delivery(candidate):
        return False, (), {}
    continuation = _immediate_continuation(candidate, ordered)
    if continuation is None or continuation.clip_id == authoritative.clip_id:
        return False, (), {}
    combined_text = f"{candidate.text} {continuation.text}".strip()
    combined_duration = candidate.duration_sec + continuation.duration_sec
    combined, combined_evidence = _same_retry_idea(
        combined_text,
        authoritative.text,
        candidate_duration_sec=combined_duration,
    )
    if not combined:
        return False, (), {}
    return True, (candidate, continuation), {
        "variant_shape": "prefix_plus_continuation",
        "continuation_clip_id": continuation.clip_id,
        **combined_evidence,
    }


def _group_lookup(groups: tuple[TakeGroup, ...]) -> dict[str, int]:
    output: dict[str, int] = {}
    for index, group in enumerate(groups):
        for clip_id in group.candidate_ids:
            output[clip_id] = index
    return output


def _semantic_map(semantic_decisions: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    return {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }


def reconcile_final_retry_families(
    project_id: str,
    takes: Iterable[CandidateTake],
    groups: Iterable[TakeGroup],
    semantic_decisions: Iterable[tuple[str, str, float]],
) -> tuple[tuple[TakeGroup, ...], tuple[dict, ...]]:
    """Merge stranded sibling groups and enforce one winner per final retry family."""
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    groups_tuple = tuple(groups)
    if not ordered or not groups_tuple:
        return groups_tuple, ()

    semantic = _semantic_map(semantic_decisions)
    clip_to_group = _group_lookup(groups_tuple)
    parent = list(range(len(groups_tuple)))
    diagnostics: list[dict] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    winners = [
        take for take in ordered
        if semantic.get(take.clip_id, ("", 0.0))[0] == "winner"
        and semantic.get(take.clip_id, ("", 0.0))[1] >= 0.90
    ]

    for candidate in ordered:
        label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
        if label not in {"failed", "alternate"} or confidence < 0.70:
            continue
        candidate_group = clip_to_group.get(candidate.clip_id)
        if candidate_group is None:
            continue

        matches = []
        for winner in winners:
            if winner.clip_id == candidate.clip_id:
                continue
            winner_group = clip_to_group.get(winner.clip_id)
            if winner_group is None or winner_group == candidate_group:
                continue
            matched, variant_members, evidence = _variant_matches_authoritative(candidate, winner, ordered)
            if not matched:
                continue
            matches.append((semantic[winner.clip_id][1], winner, winner_group, variant_members, evidence))

        # Ambiguous competing winners stay fail-open.
        if not matches:
            continue
        matches.sort(key=lambda item: item[0], reverse=True)
        if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.08:
            continue

        _, winner, winner_group, variant_members, evidence = matches[0]
        union(candidate_group, winner_group)
        merged_member_ids = [candidate.clip_id, winner.clip_id]
        for member in variant_members:
            member_group = clip_to_group.get(member.clip_id)
            if member_group is not None:
                union(candidate_group, member_group)
            if member.clip_id not in merged_member_ids:
                merged_member_ids.append(member.clip_id)
        diagnostics.append({
            "reason": "final_best_take_retry_family_reconciled",
            "candidate_clip_id": candidate.clip_id,
            "candidate_label": label,
            "candidate_confidence": round(confidence, 4),
            "winner_clip_id": winner.clip_id,
            "winner_confidence": round(semantic[winner.clip_id][1], 4),
            "family_member_ids": merged_member_ids,
            **evidence,
        })

    components: dict[int, list[int]] = {}
    for index in range(len(groups_tuple)):
        components.setdefault(find(index), []).append(index)

    output: list[TakeGroup] = []
    for component_indices in components.values():
        if len(component_indices) == 1:
            output.append(groups_tuple[component_indices[0]])
            continue

        member_groups = [groups_tuple[index] for index in component_indices]
        candidate_ids: list[str] = []
        ranked_by_id: dict[str, RankedTake] = {}
        for group in member_groups:
            for clip_id in group.candidate_ids:
                if clip_id not in candidate_ids:
                    candidate_ids.append(clip_id)
            for ranked in group.ranked:
                current = ranked_by_id.get(ranked.clip_id)
                if current is None or ranked.score > current.score:
                    ranked_by_id[ranked.clip_id] = ranked

        semantic_winners = [
            (clip_id, semantic.get(clip_id, ("", 0.0))[1])
            for clip_id in candidate_ids
            if semantic.get(clip_id, ("", 0.0))[0] == "winner"
            and semantic.get(clip_id, ("", 0.0))[1] >= 0.90
        ]
        semantic_winners.sort(key=lambda item: item[1], reverse=True)
        if semantic_winners and (len(semantic_winners) == 1 or semantic_winners[0][1] - semantic_winners[1][1] >= 0.08):
            selected_clip_id = semantic_winners[0][0]
        else:
            # Fall back to the strongest pre-existing Best Take score.
            selected_clip_id = max(
                candidate_ids,
                key=lambda clip_id: ranked_by_id.get(clip_id, RankedTake(clip_id, 0.0, "final_reconciliation_fallback")).score,
            )

        ranked = tuple(sorted(
            (
                ranked_by_id.get(
                    clip_id,
                    RankedTake(
                        clip_id=clip_id,
                        score=(1.0 if clip_id == selected_clip_id else 0.0),
                        reason="final_retry_family_reconciliation",
                    ),
                )
                for clip_id in candidate_ids
            ),
            key=lambda item: (item.clip_id != selected_clip_id, -item.score, item.clip_id),
        ))
        membership_key = "final-semantic:" + hashlib.sha256("|".join(sorted(candidate_ids)).encode()).hexdigest()[:16]
        group_id = "tg_" + hashlib.sha256(f"{project_id}|{membership_key}".encode()).hexdigest()[:18]
        output.append(TakeGroup(
            group_id=group_id,
            semantic_key=membership_key,
            candidate_ids=tuple(candidate_ids),
            ranked=ranked,
            selected_clip_id=selected_clip_id,
        ))

    # Preserve deterministic group order by earliest candidate appearance.
    position = {take.clip_id: index for index, take in enumerate(ordered)}
    output.sort(key=lambda group: min(position.get(cid, 10**9) for cid in group.candidate_ids))
    return tuple(output), tuple(diagnostics)
