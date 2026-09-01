"""Final Hybrid composite Best Take authority.

Humans do not always choose one monolithic retry. A cleaner edit may combine two
complete, complementary sub-deliveries when they jointly cover the same retry family
and each contributes useful audience-facing information.

This module stays conservative:
- only a performance-only Hybrid deletion may be restored here;
- the restored delivery must be complete and materially overlap a strong kept winner;
- a composite may replace that winner only when two restored deliveries point to the
  same winner, preserve its critical numeric/negation facts, and jointly cover most of
  its audience-facing content;
- composite members are split into singleton retry groups so ordinary one-winner Best
  Take logic cannot collapse the intended multi-part delivery again;
- when Hybrid is unavailable, an undecided incomplete immediate restart may yield to a
  nearby prior complete delivery under a stricter strong-prefix fallback.

No benchmark ids, timestamps, phrases, or Human Gold data are embedded here.
"""
from __future__ import annotations

from contextvars import ContextVar
from itertools import combinations
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
_EXPLICIT_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without"})
_PERFORMANCE_ONLY_PREFIXES = (
    "dense_physical_reset:",
    "visual_fumble:",
    "event:retry_setup:",
)

_COMPOSITE_SPLIT_IDS: ContextVar[frozenset[str]] = ContextVar(
    "cutsell_composite_best_take_split_ids",
    default=frozenset(),
)


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(char for char in raw if not unicodedata.combining(char))


def _lexeme(token: str) -> str:
    token = _canon(token)
    if len(token) >= 5 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _ordered_content(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_lexeme(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    )


def _content(text: str) -> set[str]:
    return set(_ordered_content(text))


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _EXPLICIT_NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, float(right.start) - float(left.end))
    if right.end <= left.start:
        return max(0.0, float(left.start) - float(right.end))
    return 0.0


def _semantic_map(rows: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for clip_id, label, confidence in rows:
        clip_id = str(clip_id)
        confidence = float(confidence)
        current = best.get(clip_id)
        if current is None or confidence > current[1]:
            best[clip_id] = (str(label), confidence)
    return best


def _decision_map(diagnostics: Iterable[dict]) -> dict[str, dict]:
    """Prefer the strongest applied-delete decision for each clip."""
    out: dict[str, dict] = {}
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        decisions = row.get("decisions")
        if not isinstance(decisions, list):
            continue
        for item in decisions:
            if not isinstance(item, dict) or not item.get("clip_id"):
                continue
            clip_id = str(item["clip_id"])
            current = out.get(clip_id)
            score = (
                1 if bool(item.get("applied_delete")) else 0,
                float(item.get("confidence") or 0.0),
            )
            current_score = (
                1 if current and bool(current.get("applied_delete")) else 0,
                float(current.get("confidence") or 0.0) if current else 0.0,
            )
            if current is None or score > current_score:
                out[clip_id] = item
    return out


def _existing_restored_rows(diagnostics: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        guard = row.get("hybrid_complementary_delivery_guard")
        if not isinstance(guard, dict):
            continue
        restored = guard.get("restored")
        if isinstance(restored, list):
            rows.extend(item for item in restored if isinstance(item, dict))
    return rows


def _performance_only_failure(decision: dict | None) -> bool:
    if not decision or not bool(decision.get("applied_delete")):
        return False
    if str(decision.get("delete_basis") or "") != "semantic_failed_plus_local_performance":
        return False
    if str(decision.get("reason_code") or ""):
        return False
    reasons = tuple(str(item) for item in (decision.get("local_failure_reasons") or ()))
    if not reasons:
        return False
    return all(reason.startswith(_PERFORMANCE_ONLY_PREFIXES) for reason in reasons)


def _restore_performance_only_unique_deliveries(
    kept: tuple[CandidateTake, ...],
    deleted: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    decisions: dict[str, dict],
    *,
    maximum_gap_sec: float = 45.0,
) -> tuple[set[str], list[dict]]:
    """Rescue a complete failed delivery when only performance evidence condemned it."""
    restore_ids: set[str] = set()
    rows: list[dict] = []

    for candidate in deleted:
        decision = decisions.get(candidate.clip_id)
        if not _performance_only_failure(decision):
            continue
        if not bool(candidate.complete_idea) or candidate.duration_sec < 3.0:
            continue
        label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
        if label != "failed" or confidence < 0.75:
            continue
        own = _content(candidate.text)
        if len(own) < 5:
            continue

        best = None
        best_shared = 0
        best_coverage = 0.0
        best_unique: set[str] = set()
        for peer in kept:
            if peer.source_asset_id != candidate.source_asset_id:
                continue
            if _gap(candidate, peer) > maximum_gap_sec:
                continue
            peer_label, peer_conf = semantic.get(peer.clip_id, ("", 0.0))
            if peer_label not in {"winner", "keep"} or peer_conf < 0.80:
                continue
            peer_content = _content(peer.text)
            shared = len(own & peer_content)
            coverage = shared / max(1, len(own))
            unique = own - peer_content
            if shared > best_shared or (shared == best_shared and coverage > best_coverage):
                best = peer
                best_shared = shared
                best_coverage = coverage
                best_unique = unique

        if best is None or best_shared < 4 or best_coverage < 0.50:
            continue
        unique_fraction = len(best_unique) / max(1, len(own))
        if not best_unique or unique_fraction < 0.15:
            continue

        restore_ids.add(candidate.clip_id)
        rows.append({
            "clip_id": candidate.clip_id,
            "peer_clip_id": best.clip_id,
            "reason": "restore_complete_performance_failed_delivery_with_unique_content",
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "shared_content_tokens": best_shared,
            "coverage": round(best_coverage, 4),
            "unique_content_tokens": sorted(best_unique),
            "unique_fraction": round(unique_fraction, 4),
            "delete_basis": str(decision.get("delete_basis") or ""),
        })

    return restore_ids, rows


def _delete_strong_prefix_prior_restarts(
    kept: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    *,
    maximum_prior_gap_sec: float = 8.0,
) -> tuple[set[str], list[dict]]:
    """Catch an incomplete retry just below the ordinary coverage threshold."""
    decided_ids = set(semantic)
    delete_ids: set[str] = set()
    rows: list[dict] = []
    ordered = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))

    for candidate in ordered:
        if candidate.clip_id in decided_ids or bool(candidate.complete_idea):
            continue
        own_ordered = _ordered_content(candidate.text)
        own = set(own_ordered)
        if len(own) < 8:
            continue

        best = None
        best_shared = 0
        best_coverage = 0.0
        best_prefix_shared = 0
        best_prefix_ratio = 0.0
        for peer in ordered:
            if peer.clip_id == candidate.clip_id:
                continue
            if peer.source_asset_id != candidate.source_asset_id or not bool(peer.complete_idea):
                continue
            if peer.end > candidate.start:
                continue
            gap = float(candidate.start) - float(peer.end)
            if gap < 0.0 or gap > maximum_prior_gap_sec:
                continue
            if not _critical(candidate.text).issubset(_critical(peer.text)):
                continue

            peer_content = _content(peer.text)
            shared = len(own & peer_content)
            coverage = shared / max(1, len(own))
            prefix = tuple(own_ordered[:8])
            prefix_shared = sum(1 for token in prefix if token in peer_content)
            prefix_ratio = prefix_shared / max(1, len(prefix))

            if shared < 6 or coverage < 0.40:
                continue
            if prefix_shared < 5 or prefix_ratio < 0.60:
                continue
            if coverage > best_coverage or (
                coverage == best_coverage and prefix_ratio > best_prefix_ratio
            ):
                best = peer
                best_shared = shared
                best_coverage = coverage
                best_prefix_shared = prefix_shared
                best_prefix_ratio = prefix_ratio

        if best is None:
            continue
        delete_ids.add(candidate.clip_id)
        rows.append({
            "clip_id": candidate.clip_id,
            "prior_complete_clip_id": best.clip_id,
            "reason": "hybrid_unavailable_strong_prefix_incomplete_restart_yields_to_prior_complete_delivery",
            "shared_content_tokens": best_shared,
            "coverage": round(best_coverage, 4),
            "prefix_shared_tokens": best_prefix_shared,
            "prefix_ratio": round(best_prefix_ratio, 4),
            "gap_sec": round(float(candidate.start) - float(best.end), 3),
        })

    return delete_ids, rows


def _choose_composite_replacements(
    kept: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    restored_rows: Iterable[dict],
    *,
    maximum_pair_gap_sec: float = 24.0,
) -> tuple[set[str], set[str], list[dict]]:
    """Replace one monolithic winner with two proven complementary restores."""
    by_id = {take.clip_id: take for take in kept}
    by_peer: dict[str, set[str]] = {}
    for row in restored_rows:
        peer_id = str(row.get("peer_clip_id") or "")
        clip_id = str(row.get("clip_id") or "")
        if peer_id and clip_id and peer_id in by_id and clip_id in by_id:
            by_peer.setdefault(peer_id, set()).add(clip_id)

    suppress_ids: set[str] = set()
    split_ids: set[str] = set()
    rows: list[dict] = []

    for peer_id, candidate_ids in by_peer.items():
        if len(candidate_ids) < 2:
            continue
        peer = by_id[peer_id]
        peer_label, peer_conf = semantic.get(peer_id, ("", 0.0))
        if peer_label not in {"winner", "keep"} or peer_conf < 0.80:
            continue
        peer_content = _content(peer.text)
        if len(peer_content) < 6:
            continue

        best_pair = None
        best_score = None
        best_metrics = None
        candidates = [by_id[clip_id] for clip_id in sorted(candidate_ids) if clip_id in by_id]
        for left, right in combinations(candidates, 2):
            left, right = sorted((left, right), key=lambda t: (t.start, t.end, t.clip_id))
            if left.source_asset_id != right.source_asset_id or left.source_asset_id != peer.source_asset_id:
                continue
            if not bool(left.complete_idea) or not bool(right.complete_idea):
                continue
            if left.duration_sec < 3.0 or right.duration_sec < 3.0:
                continue
            gap = max(0.0, float(right.start) - float(left.end))
            if gap > maximum_pair_gap_sec:
                continue

            left_content = _content(left.text)
            right_content = _content(right.text)
            if len(left_content) < 4 or len(right_content) < 4:
                continue
            if not (left_content - right_content) or not (right_content - left_content):
                continue
            overlap_ratio = len(left_content & right_content) / max(1, min(len(left_content), len(right_content)))
            if overlap_ratio >= 0.90:
                continue

            union = left_content | right_content
            shared = len(union & peer_content)
            coverage = shared / max(1, len(peer_content))
            if shared < 6 or coverage < 0.60:
                continue
            if not _critical(peer.text).issubset(_critical(left.text + " " + right.text)):
                continue
            union_unique = union - peer_content
            if len(union_unique) < 1:
                continue

            combined_duration = left.duration_sec + right.duration_sec
            if combined_duration > (peer.duration_sec * 1.60 + 2.0):
                continue

            score = (coverage, len(union_unique), -combined_duration)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (left, right)
                best_metrics = (shared, coverage, union_unique, gap, combined_duration)

        if best_pair is None or best_metrics is None:
            continue

        left, right = best_pair
        shared, coverage, union_unique, gap, combined_duration = best_metrics
        suppress_ids.add(peer_id)
        split_ids.update((left.clip_id, right.clip_id))
        rows.append({
            "suppressed_peer_clip_id": peer_id,
            "composite_clip_ids": [left.clip_id, right.clip_id],
            "reason": "composite_best_take_two_complementary_deliveries_replace_monolithic_retry",
            "peer_semantic_label": peer_label,
            "peer_semantic_confidence": round(peer_conf, 4),
            "peer_content_coverage": round(coverage, 4),
            "shared_peer_content_tokens": shared,
            "composite_unique_content_tokens": sorted(union_unique),
            "pair_gap_sec": round(gap, 3),
            "combined_duration_sec": round(combined_duration, 3),
        })

    return suppress_ids, split_ids, rows


def _split_groups_for_composite(
    groups: Iterable[Iterable[str]],
    split_ids: set[str] | frozenset[str],
    natural_ids: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    split_ids = set(split_ids)
    order = {clip_id: index for index, clip_id in enumerate(natural_ids)}
    out: list[tuple[str, ...]] = []
    for raw_group in groups:
        group = tuple(str(item) for item in raw_group)
        hits = tuple(clip_id for clip_id in group if clip_id in split_ids)
        remainder = tuple(clip_id for clip_id in group if clip_id not in split_ids)
        if remainder:
            out.append(remainder)
        out.extend((clip_id,) for clip_id in hits)
    out.sort(key=lambda group: min(order.get(clip_id, 10**9) for clip_id in group))
    return tuple(out)


def install_hybrid_composite_best_take() -> None:
    from . import hybrid_session_cleanup, session_boundaries

    original_hybrid = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if not getattr(original_hybrid, "_cutsell_hybrid_composite_best_take", False):
        def apply_with_composite_best_take(*args, **kwargs):
            _COMPOSITE_SPLIT_IDS.set(frozenset())
            if args:
                source_takes = tuple(args[0])
                result = original_hybrid(source_takes, *args[1:], **kwargs)
            else:
                source_takes = tuple(kwargs.get("takes") or ())
                call_kwargs = dict(kwargs)
                call_kwargs["takes"] = source_takes
                result = original_hybrid(**call_kwargs)

            semantic = _semantic_map(result.semantic_decisions)
            decisions = _decision_map(result.diagnostics)
            perf_restore_ids, perf_restore_rows = _restore_performance_only_unique_deliveries(
                tuple(result.kept),
                tuple(result.deleted),
                semantic,
                decisions,
            )

            kept_ids = {take.clip_id for take in result.kept} | perf_restore_ids
            kept = tuple(take for take in source_takes if take.clip_id in kept_ids)

            strong_delete_ids: set[str] = set()
            strong_delete_rows: list[dict] = []
            if result.requested_chunk_count > result.available_chunk_count:
                strong_delete_ids, strong_delete_rows = _delete_strong_prefix_prior_restarts(
                    kept,
                    semantic,
                )
                if strong_delete_ids:
                    kept = tuple(take for take in kept if take.clip_id not in strong_delete_ids)

            restored_rows = [*_existing_restored_rows(result.diagnostics), *perf_restore_rows]
            suppress_ids, split_ids, composite_rows = _choose_composite_replacements(
                kept,
                semantic,
                restored_rows,
            )
            if suppress_ids:
                kept = tuple(take for take in kept if take.clip_id not in suppress_ids)
            if split_ids:
                _COMPOSITE_SPLIT_IDS.set(frozenset(split_ids))

            if not perf_restore_rows and not strong_delete_rows and not composite_rows:
                return result

            final_kept_ids = {take.clip_id for take in kept}
            deleted = tuple(take for take in source_takes if take.clip_id not in final_kept_ids)
            diagnostics = tuple(result.diagnostics) + ({
                "hybrid_composite_best_take": {
                    "restored_performance_only": perf_restore_rows,
                    "deleted_strong_prefix_unavailable_restarts": strong_delete_rows,
                    "composite_replacements": composite_rows,
                    "split_group_clip_ids": sorted(split_ids),
                },
                "restored_ids": sorted(perf_restore_ids),
                "deleted_ids": sorted(strong_delete_ids | suppress_ids),
            },)
            return type(result)(
                kept=kept,
                deleted=deleted,
                requested_chunk_count=result.requested_chunk_count,
                available_chunk_count=result.available_chunk_count,
                diagnostics=diagnostics,
                semantic_decisions=result.semantic_decisions,
            )

        apply_with_composite_best_take._cutsell_hybrid_composite_best_take = True
        hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_composite_best_take

        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.apply_hybrid_session_cleanup = apply_with_composite_best_take

    original_grouping = session_boundaries.safe_group_takes_by_sessions
    if not getattr(original_grouping, "_cutsell_hybrid_composite_group_split", False):
        def group_with_composite_split(*args, **kwargs):
            result = original_grouping(*args, **kwargs)
            if len(args) >= 2:
                takes = tuple(args[1])
            else:
                takes = tuple(kwargs.get("takes") or ())
            split_ids = set(_COMPOSITE_SPLIT_IDS.get())
            _COMPOSITE_SPLIT_IDS.set(frozenset())
            if not split_ids or not takes:
                return result
            natural_ids = tuple(take.clip_id for take in takes)
            relevant = split_ids & set(natural_ids)
            if not relevant:
                return result
            groups = _split_groups_for_composite(result.groups, relevant, natural_ids)
            if groups == tuple(result.groups):
                return result
            return type(result)(
                groups=groups,
                status=result.status,
                reason="; ".join(
                    part for part in (
                        result.reason,
                        f"composite_best_take_group_split:{len(relevant)}",
                    ) if part
                ),
            )

        group_with_composite_split._cutsell_hybrid_composite_group_split = True
        session_boundaries.safe_group_takes_by_sessions = group_with_composite_split

        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.safe_group_takes_by_sessions = group_with_composite_split
