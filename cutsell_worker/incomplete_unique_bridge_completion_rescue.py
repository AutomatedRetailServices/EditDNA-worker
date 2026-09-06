"""Rescue an incomplete unique clause when the next kept take grammatically completes it.

A human editor may keep an otherwise incomplete fragment when it carries unique
information and the next selected take begins with a very short grammatical completion
before moving to the next sentence. Example structure: ``...resolved it with`` ->
``resorcinol. Next symptom...``. Treating the first fragment as a failed retry destroys
unique audience-facing information.

This module contains pure conservative detection plus an optional installer. It does not
encode benchmark ids, phrases, or timestamps.
"""
from __future__ import annotations

from contextvars import ContextVar
import re
import sys
import unicodedata
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})
_BRIDGE_END = frozenset({
    "a", "al", "and", "because", "but", "by", "con", "de", "del", "for", "of", "para",
    "pero", "por", "porque", "que", "the", "to", "with", "y",
})

_SPLIT_IDS: ContextVar[frozenset[str]] = ContextVar(
    "cutsell_incomplete_unique_bridge_split_ids", default=frozenset()
)


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_canon(item) for item in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _ends_on_bridge(text: str) -> bool:
    tokens = _tokens(text)
    return bool(tokens and tokens[-1] in _BRIDGE_END)


def _short_initial_completion(text: str, *, maximum_tokens: int = 3) -> tuple[str, ...]:
    raw = str(text or "").strip()
    if not raw:
        return ()
    first = re.split(r"[.!?]+", raw, maxsplit=1)[0]
    tokens = _tokens(first)
    if not tokens or len(tokens) > maximum_tokens:
        return ()
    return tokens


def bridge_completion_relation(
    incomplete: CandidateTake,
    following: CandidateTake,
    *,
    maximum_gap_sec: float = 3.0,
    minimum_unique_content_tokens: int = 3,
) -> dict | None:
    if bool(incomplete.complete_idea) or not bool(following.complete_idea):
        return None
    if incomplete.source_asset_id != following.source_asset_id:
        return None
    gap = float(following.start) - float(incomplete.end)
    if gap < 0.0 or gap > maximum_gap_sec:
        return None
    if not _ends_on_bridge(incomplete.text):
        return None
    completion = _short_initial_completion(following.text)
    if not completion:
        return None
    own = _content(incomplete.text)
    later = _content(following.text)
    unique = own - later
    if len(unique) < minimum_unique_content_tokens:
        return None
    return {
        "clip_id": incomplete.clip_id,
        "following_clip_id": following.clip_id,
        "reason": "incomplete_unique_clause_completed_by_short_following_phrase",
        "gap_sec": round(gap, 3),
        "completion_tokens": list(completion),
        "unique_content_tokens": sorted(unique),
    }


def _semantic_map(rows: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for clip_id, label, confidence in rows:
        clip_id = str(clip_id)
        confidence = float(confidence)
        current = best.get(clip_id)
        if current is None or confidence > current[1]:
            best[clip_id] = (str(label), confidence)
    return best


def _has_uncorroborated_semantic_failure(clip_id: str, diagnostics: Iterable[dict]) -> bool:
    seen_failed = False
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        decisions = row.get("decisions")
        if not isinstance(decisions, list):
            continue
        for item in decisions:
            if not isinstance(item, dict) or str(item.get("clip_id") or "") != clip_id:
                continue
            if str(item.get("label") or "") != "failed" or float(item.get("confidence") or 0.0) < 0.80:
                continue
            seen_failed = True
            if bool(item.get("local_failure_corroborated")):
                return False
    return seen_failed


def _split_groups(groups, split_ids: set[str], natural_ids: tuple[str, ...]):
    order = {clip_id: i for i, clip_id in enumerate(natural_ids)}
    out: list[tuple[str, ...]] = []
    for raw_group in groups:
        group = tuple(str(x) for x in raw_group)
        remainder = tuple(cid for cid in group if cid not in split_ids)
        hits = tuple(cid for cid in group if cid in split_ids)
        if remainder:
            out.append(remainder)
        out.extend((cid,) for cid in hits)
    out.sort(key=lambda group: min(order.get(cid, 10**9) for cid in group))
    return tuple(out)


def install_incomplete_unique_bridge_completion_rescue() -> None:
    from . import hybrid_session_cleanup, session_boundaries

    original_hybrid = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if not getattr(original_hybrid, "_cutsell_incomplete_unique_bridge_rescue", False):
        def apply_with_bridge_rescue(*args, **kwargs):
            _SPLIT_IDS.set(frozenset())
            if args:
                source_takes = tuple(args[0])
                result = original_hybrid(source_takes, *args[1:], **kwargs)
            else:
                source_takes = tuple(kwargs.get("takes") or ())
                call_kwargs = dict(kwargs)
                call_kwargs["takes"] = source_takes
                result = original_hybrid(**call_kwargs)

            semantic = _semantic_map(result.semantic_decisions)
            deleted_ids = {take.clip_id for take in result.deleted}
            kept_ids = {take.clip_id for take in result.kept}
            ordered = tuple(sorted(source_takes, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
            restore_ids: set[str] = set()
            split_ids: set[str] = set()
            audit: list[dict] = []
            for index, candidate in enumerate(ordered[:-1]):
                if candidate.clip_id not in deleted_ids:
                    continue
                label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
                if label != "failed" or confidence < 0.80:
                    continue
                if not _has_uncorroborated_semantic_failure(candidate.clip_id, result.diagnostics):
                    continue
                following = ordered[index + 1]
                if following.clip_id not in kept_ids:
                    continue
                follow_label, follow_conf = semantic.get(following.clip_id, ("", 0.0))
                if follow_label not in {"winner", "keep"} or follow_conf < 0.85:
                    continue
                relation = bridge_completion_relation(candidate, following)
                if relation is None:
                    continue
                restore_ids.add(candidate.clip_id)
                split_ids.update((candidate.clip_id, following.clip_id))
                audit.append({
                    **relation,
                    "semantic_confidence": round(confidence, 4),
                    "following_semantic_confidence": round(follow_conf, 4),
                })

            if not restore_ids:
                return result
            final_kept_ids = kept_ids | restore_ids
            kept = tuple(take for take in source_takes if take.clip_id in final_kept_ids)
            deleted = tuple(take for take in source_takes if take.clip_id not in final_kept_ids)
            _SPLIT_IDS.set(frozenset(split_ids))
            return type(result)(
                kept=kept,
                deleted=deleted,
                requested_chunk_count=result.requested_chunk_count,
                available_chunk_count=result.available_chunk_count,
                diagnostics=tuple(result.diagnostics) + ({
                    "incomplete_unique_bridge_completion_rescue": audit,
                    "restored_ids": sorted(restore_ids),
                    "split_group_clip_ids": sorted(split_ids),
                },),
                semantic_decisions=result.semantic_decisions,
            )

        apply_with_bridge_rescue._cutsell_incomplete_unique_bridge_rescue = True
        hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_bridge_rescue
        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.apply_hybrid_session_cleanup = apply_with_bridge_rescue

    original_grouping = session_boundaries.safe_group_takes_by_sessions
    if not getattr(original_grouping, "_cutsell_incomplete_unique_bridge_group_split", False):
        def group_with_bridge_split(*args, **kwargs):
            result = original_grouping(*args, **kwargs)
            takes = tuple(args[1]) if len(args) >= 2 else tuple(kwargs.get("takes") or ())
            split_ids = set(_SPLIT_IDS.get())
            _SPLIT_IDS.set(frozenset())
            if not takes or not split_ids:
                return result
            natural_ids = tuple(take.clip_id for take in takes)
            relevant = split_ids & set(natural_ids)
            if not relevant:
                return result
            groups = _split_groups(result.groups, relevant, natural_ids)
            if groups == tuple(result.groups):
                return result
            return type(result)(
                groups=groups,
                status=result.status,
                reason="; ".join(part for part in (
                    result.reason,
                    f"incomplete_unique_bridge_group_split:{len(relevant)}",
                ) if part),
            )

        group_with_bridge_split._cutsell_incomplete_unique_bridge_group_split = True
        session_boundaries.safe_group_takes_by_sessions = group_with_bridge_split
        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.safe_group_takes_by_sessions = group_with_bridge_split
