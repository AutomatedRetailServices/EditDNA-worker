"""Final sibling-family reconciliation before Best Take.

The old EditDNA editor had one useful invariant that Clean Cut must preserve: deliveries
that are genuinely competing versions of the same idea belong to one sibling family so
Best Take can choose exactly one winner.  This wrapper runs after all existing local
retry-group repair and before session grouping is rebound into production.

It never deletes a take and never chooses a winner.  It only merges groups when strong
retry evidence exists.  The normal TakeJudge + Hybrid winner logic then decides which
member is selected; the remaining family members stay available as alternates / Swap
Take.
"""
from __future__ import annotations

import re

from .contracts import CandidateTake
from .take_grouping_provider import TakeGroupingProviderResult

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
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


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    return {
        token for token in _tokens(text)
        if token in _CRITICAL or any(ch.isdigit() for ch in token)
    }


def _coverage(subject_text: str, peer_text: str) -> tuple[int, float]:
    subject = _content(subject_text)
    peer = _content(peer_text)
    if not subject:
        return 0, 0.0
    shared = len(subject & peer)
    return shared, shared / max(1, len(subject))


def _same_retry_idea(left_text: str, right_text: str) -> bool:
    left = _content(left_text)
    right = _content(right_text)
    if len(left) < 4 or len(right) < 4:
        return False
    shared = len(left & right)
    left_coverage = shared / len(left)
    right_coverage = shared / len(right)
    if not _critical(left_text).issubset(_critical(right_text) | _critical(left_text)):
        return False
    # Full reformulations need substantial overlap in both directions.  This is stricter
    # than broad topic similarity and is intentionally not a sales/story clustering rule.
    return shared >= 5 and min(left_coverage, right_coverage) >= 0.38 and max(left_coverage, right_coverage) >= 0.55


def _group_members(group, take_map):
    return tuple(
        sorted(
            (take_map[cid] for cid in group if cid in take_map),
            key=lambda item: (item.source_order, item.start, item.end, item.clip_id),
        )
    )


def _representative(group, take_map) -> CandidateTake | None:
    members = _group_members(group, take_map)
    if not members:
        return None
    return max(members, key=lambda take: (take.duration_sec, len(_content(take.text)), -take.start))


def _group_gap(left_group, right_group, take_map) -> float:
    left_members = _group_members(left_group, take_map)
    right_members = _group_members(right_group, take_map)
    if not left_members or not right_members:
        return float("inf")
    if left_members[0].source_asset_id != right_members[0].source_asset_id:
        return float("inf")
    left_end = max(item.end for item in left_members)
    right_start = min(item.start for item in right_members)
    if right_start >= left_end:
        return right_start - left_end
    right_end = max(item.end for item in right_members)
    left_start = min(item.start for item in left_members)
    if left_start >= right_end:
        return left_start - right_end
    return 0.0


def _open_take(take: CandidateTake) -> bool:
    text = str(take.text or "").strip()
    return bool(text) and (not take.complete_idea or not _SENTENCE_END_RE.search(text))


def _variant_chain_matches_complete(
    complete_group,
    prefix_group,
    continuation_group,
    take_map,
) -> bool:
    complete = _representative(complete_group, take_map)
    prefix = _representative(prefix_group, take_map)
    continuation = _representative(continuation_group, take_map)
    if complete is None or prefix is None or continuation is None:
        return False
    if len({complete.source_asset_id, prefix.source_asset_id, continuation.source_asset_id}) != 1:
        return False
    if not _open_take(prefix):
        return False
    if _group_gap(prefix_group, continuation_group, take_map) > 3.0:
        return False
    if _group_gap(complete_group, prefix_group, take_map) > 60.0:
        return False

    combined = f"{prefix.text} {continuation.text}".strip()
    complete_content = _content(complete.text)
    combined_content = _content(combined)
    if len(complete_content) < 6 or len(combined_content) < 6:
        return False
    shared = len(complete_content & combined_content)
    complete_coverage = shared / len(complete_content)
    combined_coverage = shared / len(combined_content)
    critical_ok = _critical(complete.text).issubset(_critical(combined)) and _critical(combined).issubset(_critical(complete.text))
    return bool(
        shared >= 7
        and complete_coverage >= 0.45
        and combined_coverage >= 0.38
        and critical_ok
    )


def reconcile_final_sibling_groups(groups, takes):
    """Merge strong cross-group siblings, including one split retry variant."""
    groups = tuple(tuple(group) for group in groups if group)
    takes = tuple(takes)
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    work = [list(group) for group in groups]
    changed = False

    # First merge direct full-delivery reformulations.  Require proximity because this is
    # recording-retry grouping, not thematic/story grouping.
    progress = True
    while progress:
        progress = False
        i = 0
        while i < len(work):
            left = _representative(work[i], take_map)
            if left is None:
                i += 1
                continue
            j = i + 1
            while j < len(work):
                right = _representative(work[j], take_map)
                if right is None:
                    j += 1
                    continue
                if right.source_order != left.source_order or right.source_asset_id != left.source_asset_id:
                    break
                if _group_gap(work[i], work[j], take_map) > 60.0:
                    break
                if _same_retry_idea(left.text, right.text):
                    work[i].extend(work[j])
                    del work[j]
                    changed = progress = True
                    left = _representative(work[i], take_map) or left
                    continue
                j += 1
            i += 1

    # Then handle the common long-form shape: one complete delivery competes with a
    # reformulated retry that ASR/attempt reconstruction split into prefix+continuation.
    # Merge all three groups so Best Take sees one family and can choose the complete
    # semantic winner; the two split children remain alternates instead of final speech.
    i = 0
    while i < len(work):
        merged_here = False
        for j in range(len(work)):
            if j == i:
                continue
            for k in range(len(work)):
                if k in {i, j}:
                    continue
                if _variant_chain_matches_complete(work[i], work[j], work[k], take_map):
                    target = min(i, j, k)
                    member_ids = []
                    for index in sorted({i, j, k}):
                        member_ids.extend(work[index])
                    for index in sorted({i, j, k}, reverse=True):
                        del work[index]
                    work.insert(target, member_ids)
                    changed = merged_here = True
                    break
            if merged_here:
                break
        if merged_here:
            i = 0
            continue
        i += 1

    normalized = []
    for group in work:
        unique = sorted(
            set(group),
            key=lambda cid: (
                take_map[cid].source_order,
                take_map[cid].start,
                take_map[cid].end,
                cid,
            ),
        )
        normalized.append(tuple(unique))
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def install_final_sibling_grouping() -> None:
    from . import take_grouping_provider

    original = take_grouping_provider.safe_group_takes
    if getattr(original, "_cutsell_final_sibling_grouping", False):
        return

    def safe_group_with_final_siblings(provider, takes, context_text=""):
        take_tuple = tuple(takes)
        result = original(provider, take_tuple, context_text=context_text)
        groups, changed = reconcile_final_sibling_groups(result.groups, take_tuple)
        if not changed:
            return result
        reason = (result.reason + "; " if result.reason else "") + "final_sibling_reconciled"
        return TakeGroupingProviderResult(groups=groups, status=result.status, reason=reason)

    safe_group_with_final_siblings._cutsell_final_sibling_grouping = True
    take_grouping_provider.safe_group_takes = safe_group_with_final_siblings
