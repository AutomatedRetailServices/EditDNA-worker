"""Final draft reconciliation for retry structures exposed by focused Round 8.

Round 8 proved that two retry failures can survive all earlier local guards:

1) the broken attempt and its failed restart bridge are both discarded, but the clean
   terminal retake is also discarded as redundant.  Once the failed chain is known, a
   positively-judged clean peer must be allowed back into the draft;
2) a later retry continuation is correctly discarded while its semantically failed open
   prefix remains selected.  When an earlier selected complete delivery covers the whole
   losing chain, the orphan prefix must also leave the draft.

This pass runs on DraftClip objects after Best Take/Hybrid/final-draft integrity.  It never
invents words, timestamps, retry groups, or semantic evidence and fails open unless the
combined structural evidence is strong.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import DraftClip
from . import final_draft_retry_integrity as base

_POSITIVE_LABELS = frozenset({"alternate", "keep", "winner"})


def _positive_semantics(diagnostics: dict) -> dict[str, tuple[str, float]]:
    """Return strongest positive Hybrid decision per clip, excluding strong failures."""
    failures = base._semantic_failures(diagnostics)
    positive: dict[str, tuple[str, float]] = {}
    priority = {"alternate": 1, "keep": 2, "winner": 3}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for item in chunk.get("decisions") or ():
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip().lower()
            if label not in _POSITIVE_LABELS:
                continue
            clip_id = str(item.get("clip_id") or "")
            if not clip_id or failures.get(clip_id, 0.0) >= 0.80:
                continue
            confidence = float(item.get("confidence") or 0.0)
            current = positive.get(clip_id)
            if current is None or (priority[label], confidence) > (priority[current[0]], current[1]):
                positive[clip_id] = (label, confidence)
    return positive


def _opening_repeat_coverage(prefix_text: str, attempt_text: str) -> tuple[int, float]:
    prefix = base._tokens(prefix_text)
    attempt = base._tokens(attempt_text)
    if len(prefix) < 4 or len(attempt) < len(prefix):
        return 0, 0.0
    matched = 0
    for left, right in zip(prefix, attempt[: len(prefix)]):
        if left != right:
            break
        matched += 1
    return matched, matched / max(1, len(prefix))


def restore_clean_retake_after_failed_discard_chain(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_bridge_gap_sec: float = 3.5,
    maximum_attempt_window_sec: float = 16.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    """Restore a clean discarded retake when the discarded retry chain proves relation."""
    selected_list = list(selected)
    discarded_list = sorted(discarded, key=lambda c: (c.source_order, c.start, c.end, c.clip_id))
    failures = base._semantic_failures(diagnostics)
    positive = _positive_semantics(diagnostics)
    restored_ids: set[str] = set()
    audit: list[dict] = []

    for candidate in discarded_list:
        label, confidence = positive.get(candidate.clip_id, ("", 0.0))
        if label not in _POSITIVE_LABELS or confidence < 0.70:
            continue
        if failures.get(candidate.clip_id, 0.0) >= 0.80:
            continue
        if len(base._tokens(candidate.text)) < 4:
            continue

        options = []
        for bridge in discarded_list:
            if bridge.clip_id == candidate.clip_id or bridge.source_asset_id != candidate.source_asset_id:
                continue
            if float(bridge.end) > float(candidate.start):
                continue
            bridge_gap = float(candidate.start) - float(bridge.end)
            if bridge_gap < 0 or bridge_gap > maximum_bridge_gap_sec:
                continue
            bridge_failure = failures.get(bridge.clip_id, 0.0)
            if bridge_failure < 0.80:
                continue

            for attempt in discarded_list:
                if attempt.clip_id in {candidate.clip_id, bridge.clip_id}:
                    continue
                if attempt.source_asset_id != candidate.source_asset_id:
                    continue
                if float(attempt.end) > float(bridge.start):
                    continue
                if float(candidate.start) - float(attempt.start) > maximum_attempt_window_sec:
                    continue

                opening_words, opening_cov = _opening_repeat_coverage(bridge.text, attempt.text)
                lexical_retry = opening_words >= 4 and opening_cov >= 0.80
                if not lexical_retry:
                    continue

                shared, attempt_cov, candidate_cov = base._coverage(attempt.text, candidate.text)
                if shared < 2 or candidate_cov < 0.45:
                    continue

                reset_count, reset_conf = base._reset_count_between(attempt, candidate, diagnostics)
                if reset_count < 2 and not (bridge_failure >= 0.90 and opening_cov >= 0.80):
                    continue

                options.append((
                    candidate_cov,
                    attempt_cov,
                    shared,
                    opening_cov,
                    bridge_failure,
                    reset_count,
                    reset_conf,
                    attempt,
                    bridge,
                ))

        if not options:
            continue

        (
            candidate_cov,
            attempt_cov,
            shared,
            opening_cov,
            bridge_failure,
            reset_count,
            reset_conf,
            attempt,
            bridge,
        ) = max(options, key=lambda item: item[:7])

        restored_ids.add(candidate.clip_id)
        selected_list.append(replace(candidate, selected=True))
        audit.append({
            "reason": "restore_clean_retake_after_failed_discard_chain",
            "restored_clip_id": candidate.clip_id,
            "failed_attempt_clip_id": attempt.clip_id,
            "failed_bridge_clip_id": bridge.clip_id,
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "bridge_failure_confidence": round(bridge_failure, 4),
            "opening_repeat_coverage": round(opening_cov, 4),
            "shared_content_tokens": shared,
            "candidate_coverage": round(candidate_cov, 4),
            "attempt_coverage": round(attempt_cov, 4),
            "reset_event_count": reset_count,
            "best_reset_confidence": round(reset_conf, 4),
            "restored_text": candidate.text,
        })

    if not restored_ids:
        return tuple(selected), tuple(discarded), ()

    selected_out = tuple(sorted(selected_list, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    discarded_out = tuple(c for c in discarded_list if c.clip_id not in restored_ids)
    return selected_out, discarded_out, tuple(audit)


def _critical_compatible(left_text: str, right_text: str) -> bool:
    left = base._critical(left_text)
    right = base._critical(right_text)
    left_nums = {value for value in left if value.startswith("num:")}
    right_nums = {value for value in right if value.startswith("num:")}
    if left_nums and right_nums and left_nums != right_nums:
        return False
    if ("__negation__" in left) != ("__negation__" in right):
        return False
    return True


def suppress_orphan_failed_open_prefix(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_continuation_gap_sec: float = 3.5,
    maximum_prior_gap_sec: float = 45.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    """Remove a failed open prefix whose continuation already lost to a prior complete take."""
    selected_list = list(sorted(selected, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    discarded_list = list(sorted(discarded, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    failures = base._semantic_failures(diagnostics)
    removed_ids: set[str] = set()
    audit: list[dict] = []

    for prefix in selected_list:
        failure_conf = failures.get(prefix.clip_id, 0.0)
        if failure_conf < 0.80 or not base._is_open_text(prefix.text):
            continue

        continuation_options = [
            item for item in discarded_list
            if item.source_asset_id == prefix.source_asset_id
            and float(item.start) >= float(prefix.end)
            and 0.0 <= float(item.start) - float(prefix.end) <= maximum_continuation_gap_sec
        ]
        if not continuation_options:
            continue
        continuation = min(continuation_options, key=lambda item: (item.start, item.end))
        combined_text = f"{prefix.text} {continuation.text}".strip()

        prior_options = []
        for prior in selected_list:
            if prior.clip_id == prefix.clip_id or prior.clip_id in removed_ids:
                continue
            if prior.source_asset_id != prefix.source_asset_id or float(prior.end) > float(prefix.start):
                continue
            prior_gap = float(prefix.start) - float(prior.end)
            if prior_gap > maximum_prior_gap_sec:
                continue
            shared, combined_cov, prior_cov = base._coverage(combined_text, prior.text)
            if shared < 7 or combined_cov < 0.35 or prior_cov < 0.35:
                continue
            if not _critical_compatible(combined_text, prior.text):
                continue
            prior_options.append((combined_cov, prior_cov, shared, -prior_gap, prior))
        if not prior_options:
            continue

        combined_cov, prior_cov, shared, _, prior = max(prior_options, key=lambda item: item[:4])
        removed_ids.add(prefix.clip_id)
        audit.append({
            "reason": "orphan_failed_open_prefix_yields_to_prior_complete_delivery",
            "removed_clip_id": prefix.clip_id,
            "discarded_continuation_clip_id": continuation.clip_id,
            "prior_winner_clip_id": prior.clip_id,
            "failed_prefix_confidence": round(failure_conf, 4),
            "combined_shared_content_tokens": shared,
            "combined_coverage": round(combined_cov, 4),
            "prior_coverage": round(prior_cov, 4),
            "removed_text": prefix.text,
            "continuation_text": continuation.text,
            "prior_text": prior.text,
        })

    if not removed_ids:
        return tuple(selected_list), tuple(discarded_list), ()

    removed = [clip for clip in selected_list if clip.clip_id in removed_ids]
    selected_out = tuple(clip for clip in selected_list if clip.clip_id not in removed_ids)
    existing = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [
        replace(clip, selected=False)
        for clip in removed
        if clip.clip_id not in existing
    ])
    return selected_out, discarded_out, tuple(audit)


def install_round8_retry_reconciliation() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_round8_retry_reconciliation", False):
        return

    def build_with_round8_retry_reconciliation(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})

        selected, discarded, restore_diag = restore_clean_retake_after_failed_discard_chain(
            draft.selected,
            draft.discarded,
            diagnostics,
        )
        selected, discarded, orphan_diag = suppress_orphan_failed_open_prefix(
            selected,
            discarded,
            diagnostics,
        )
        if not restore_diag and not orphan_diag:
            return result

        diagnostics["round8_retry_reconciliation"] = [*list(restore_diag), *list(orphan_diag)]
        repaired = replace(
            draft,
            selected=selected,
            discarded=discarded,
            diagnostics=diagnostics,
        )
        return replace(result, draft=repaired)

    build_with_round8_retry_reconciliation._cutsell_round8_retry_reconciliation = True
    pipeline.build_flow_b_draft = build_with_round8_retry_reconciliation
