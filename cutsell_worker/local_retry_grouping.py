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
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "that", "the", "this",
    "to", "was", "we", "what", "with", "you", "your", "okay", "ok", "now", "whole",
    "sentence",
    # Spanish
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis",
    "o", "para", "pero", "por", "porque", "que", "se", "si", "sin", "su", "sus", "un",
    "una", "unos", "unas", "y", "yo",
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


def _content_containment(left: str, right: str) -> float:
    a = set(_content_tokens(left))
    b = set(_content_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _same_semantic_opening(left: str, right: str, *, width: int = 2) -> bool:
    a = _content_tokens(left)
    b = _content_tokens(right)
    if len(a) < width or len(b) < width:
        return False
    return a[:width] == b[:width]


def _restart_heavy(text: str) -> bool:
    tokens = _content_tokens(text)
    if len(_tokens(text)) <= 6:
        return True
    if _META_RE.search(str(text or "")):
        return True
    counts = Counter(tokens)
    return any(count >= 2 for count in counts.values())


def _dominant_weak_group_with_fuller_retry(left_members, right_members) -> bool:
    """Allow one trailing partial phrase inside an otherwise obvious retry envelope."""
    if len(left_members) < 3 or not right_members:
        return False
    weak_count = sum(1 for member in left_members if _restart_heavy(member.text))
    if weak_count < len(left_members) - 1:
        return False
    trailing = left_members[-2:]
    for right in right_members:
        right_tokens = _tokens(right.text)
        for left in trailing:
            left_tokens = _tokens(left.text)
            if len(right_tokens) - len(left_tokens) < 3:
                continue
            if right.duration_sec < left.duration_sec + 0.70:
                continue
            if _shared_content_strength(left.text, right.text) >= 3:
                return True
    return False


def _serial_retry_envelope(groups, takes, *, maximum_gap_sec: float = 20.0):
    """Absorb weak serial retries into the following linked retry group."""
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
            dominant_weak_bridge = _dominant_weak_group_with_fuller_retry(left_members, right_members)
            linked = any(
                _shared_content_strength(left.text, right.text) >= 2
                for left in left_members
                for right in right_members
            )
            if (left_is_weak or dominant_weak_bridge) and gap <= maximum_gap_sec and linked:
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


def _adjacent_reformulated_retries(
    groups,
    takes,
    *,
    maximum_gap_sec: float = 18.0,
    minimum_shared_content: int = 4,
    minimum_containment: float = 0.68,
):
    """Merge nearby full retries that keep the same semantic opening.

    Long-form creators often restart an idea with small wording changes rather than
    repeating it verbatim. Exact/fuzzy string similarity can miss those retries. This
    bridge is deliberately strict: groups must be adjacent, close in time, share the
    first two meaningful content tokens, share at least four content tokens overall,
    and have high content containment. It therefore captures ``Al terminar mi contrato``
    reformulations without clustering unrelated neighboring sentences about the same
    broad topic.
    """
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    work = [list(group) for group in groups]
    changed = False

    def members(group):
        return sorted((take_map[cid] for cid in group), key=lambda t: (t.start, t.end, t.clip_id))

    index = 0
    while index < len(work) - 1:
        left_members = members(work[index])
        right_members = members(work[index + 1])
        if not left_members or not right_members:
            index += 1
            continue
        left = max(left_members, key=lambda item: (item.duration_sec, len(_tokens(item.text))))
        right = max(right_members, key=lambda item: (item.duration_sec, len(_tokens(item.text))))
        if left.source_asset_id != right.source_asset_id:
            index += 1
            continue
        gap = max(0.0, right_members[0].start - left_members[-1].end)
        shared = _shared_content_strength(left.text, right.text)
        containment = _content_containment(left.text, right.text)
        if (
            gap <= maximum_gap_sec
            and min(len(_content_tokens(left.text)), len(_content_tokens(right.text))) >= 5
            and _same_semantic_opening(left.text, right.text)
            and shared >= minimum_shared_content
            and containment >= minimum_containment
        ):
            work[index] = work[index] + work[index + 1]
            del work[index + 1]
            changed = True
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
        groups, reformulated = _adjacent_reformulated_retries(groups, takes)

        reasons = ["baseline_local"]
        if reconciled:
            reasons.append("local_retry_reconciled")
        if extended:
            reasons.append("adjacent_retry_extended")
        if debris_absorbed:
            reasons.append("interstitial_retry_debris_absorbed")
        if serial_envelope:
            reasons.append("serial_retry_envelope_collapsed")
        if reformulated:
            reasons.append("adjacent_reformulated_retries_collapsed")

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
