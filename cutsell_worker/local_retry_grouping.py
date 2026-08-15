"""Enable full deterministic retry reconciliation on the RunPod-local path.

The provider boundary already contains conservative repair stages for missed retries,
adjacent retake extension, and interstitial false-start debris. Historically those
stages only ran after an external grouping provider returned output. Clean Cut now
runs with ``provider=None`` by design, so the local path stopped at seed lexical
groups and Best Take never got a chance to compare many obvious retakes.

This installer wraps the grouping boundary before ``pipeline`` imports it. It does not
change deletion thresholds and it never drops a candidate: it only makes retry groups
more complete so the existing Best Take ranking can choose one representative.
"""
from __future__ import annotations

from collections import Counter
import re

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "that", "the", "this",
    "to", "was", "we", "what", "with", "you", "your", "okay", "ok", "now", "whole",
    "sentence",
})
_META_RE = re.compile(r"\bwhole\s+sentence\b|\b(?:say|do)\s+that\s+again\b", re.IGNORECASE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 4 and token not in _STOP)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    short, long = (left, right) if len(left) < len(right) else (right, left)
    i = j = edits = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _shared_content_strength(left: str, right: str) -> int:
    a = set(_content_tokens(left))
    b = set(_content_tokens(right))
    exact = len(a & b)
    if exact >= 2:
        return exact
    fuzzy = exact
    for token_a in a - b:
        if any(_edit_distance_at_most_one(token_a, token_b) for token_b in b - a):
            fuzzy += 1
    return fuzzy


def _restart_heavy(text: str) -> bool:
    tokens = _content_tokens(text)
    if len(_tokens(text)) <= 6:
        return True
    if _META_RE.search(str(text or "")):
        return True
    counts = Counter(tokens)
    return any(count >= 2 for count in counts.values())


def _serial_retry_envelope(groups, takes, *, maximum_gap_sec: float = 20.0):
    """Absorb weak serial retries into the following linked retry group.

    Only the left/current group may trigger this bridge: it must be made entirely of
    short/repetitive/recording-meta takes. The adjacent next group must share at least
    two content tokens (a one-character typo may count as one match). This lets a chain
    like ``popular croc`` -> ``crop popular crop popular`` -> ``popular crop black...``
    reach the final full retake while protecting two distinct substantive sentences.
    """
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    work = [list(group) for group in groups]
    changed = False

    def group_members(group):
        return sorted((take_map[cid] for cid in group), key=lambda t: (t.start, t.end, t.clip_id))

    made_progress = True
    while made_progress:
        made_progress = False
        index = 0
        while index < len(work) - 1:
            left_members = group_members(work[index])
            right_members = group_members(work[index + 1])
            if not left_members or not right_members:
                index += 1
                continue
            if left_members[0].source_asset_id != right_members[0].source_asset_id:
                index += 1
                continue
            gap = max(0.0, right_members[0].start - left_members[-1].end)
            left_is_weak = all(_restart_heavy(member.text) for member in left_members)
            linked = any(
                _shared_content_strength(left.text, right.text) >= 2
                for left in left_members
                for right in right_members
            )
            if left_is_weak and gap <= maximum_gap_sec and linked:
                work[index + 1] = work[index] + work[index + 1]
                del work[index]
                changed = True
                made_progress = True
                index = max(0, index - 1)
                continue
            index += 1

    normalized = []
    for group in work:
        ordered = sorted(
            set(group),
            key=lambda cid: (
                take_map[cid].source_order,
                take_map[cid].start,
                take_map[cid].end,
                cid,
            ),
        )
        normalized.append(tuple(ordered))
    return tuple(normalized), changed


def install_local_retry_grouping() -> None:
    from . import take_grouping_provider as grouping

    original = grouping.safe_group_takes
    if getattr(original, "_cutsell_local_retry_grouping", False):
        return

    def safe_group_takes_with_local_reconciliation(provider, takes, context_text=""):
        result = original(provider, takes, context_text=context_text)
        if provider is not None or len(takes) <= 1:
            return result

        groups, reconciled = grouping._reconcile_missed_retries(result.groups, takes)
        groups, extended = grouping._extend_adjacent_retry_groups(groups, takes)
        groups, debris_absorbed = grouping._absorb_interstitial_retry_debris(groups, takes)
        groups, serial_envelope = _serial_retry_envelope(groups, takes)

        reasons = ["baseline_local"]
        if reconciled:
            reasons.append("local_retry_reconciled")
        if extended:
            reasons.append("adjacent_retry_extended")
        if debris_absorbed:
            reasons.append("interstitial_retry_debris_absorbed")
        if serial_envelope:
            reasons.append("serial_retry_envelope_collapsed")

        return grouping.TakeGroupingProviderResult(
            groups=groups,
            status=grouping.ProviderStatus(
                "baseline",
                False,
                True,
                "local_retry_reconciled",
            ),
            reason="; ".join(reasons),
        )

    safe_group_takes_with_local_reconciliation._cutsell_local_retry_grouping = True
    grouping.safe_group_takes = safe_group_takes_with_local_reconciliation
