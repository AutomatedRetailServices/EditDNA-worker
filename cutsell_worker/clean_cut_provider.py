"""Optional high-confidence judge for obvious recording mistakes.

This provider is deliberately separate from semantic/commercial classification. It
may only judge whether an already-segmented candidate is valid speech, a whole-take
recording mistake, or mixed speech that can be trimmed at real ASR word boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .source_identity import stable_clip_id


@dataclass(frozen=True)
class CleanCutJudgement:
    clip_id: str
    action: str  # keep | delete | mixed
    confidence: float
    reason: str
    keep_start_word_index: int | None = None
    keep_end_word_index: int | None = None  # inclusive


@dataclass(frozen=True)
class CleanCutProviderResult:
    judgements: Tuple[CleanCutJudgement, ...]
    status: ProviderStatus


class CleanCutProvider(Protocol):
    def judge(self, takes: Tuple[CandidateTake, ...]) -> CleanCutProviderResult: ...


def _validate_provider_result(
    result: CleanCutProviderResult,
    takes: Tuple[CandidateTake, ...],
) -> CleanCutProviderResult:
    expected = {take.clip_id for take in takes}
    seen = set()
    normalized = []
    for item in result.judgements:
        if item.clip_id not in expected or item.clip_id in seen:
            raise ValueError("clean cut judge returned invalid clip id")
        action = str(item.action).lower().strip()
        if action not in {"keep", "delete", "mixed"}:
            raise ValueError("clean cut judge returned invalid action")
        confidence = float(item.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("clean cut judge confidence outside 0..1")

        start_index = item.keep_start_word_index
        end_index = item.keep_end_word_index
        if start_index is not None:
            start_index = int(start_index)
        if end_index is not None:
            end_index = int(end_index)
        if action == "mixed" and ((start_index is None) != (end_index is None)):
            raise ValueError("mixed judgement must provide both word-boundary indexes or neither")

        normalized.append(CleanCutJudgement(
            clip_id=item.clip_id,
            action=action,
            confidence=confidence,
            reason=str(item.reason or "")[:240],
            keep_start_word_index=start_index,
            keep_end_word_index=end_index,
        ))
        seen.add(item.clip_id)
    if seen != expected:
        raise ValueError("clean cut judge omitted candidates")
    return CleanCutProviderResult(tuple(normalized), result.status)


def safe_clean_cut_judge(
    provider: CleanCutProvider | None,
    takes: Tuple[CandidateTake, ...],
) -> CleanCutProviderResult:
    """Fail open: malformed output gets one strict retry, then keeps all speech."""
    if provider is None or not takes:
        return CleanCutProviderResult(
            (),
            ProviderStatus("none", False, False, "not_requested"),
        )

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = provider.judge(takes)
            return _validate_provider_result(result, takes)
        except ValueError as exc:
            last_exc = exc
            if attempt == 0:
                continue
            break
        except Exception as exc:
            last_exc = exc
            break

    assert last_exc is not None
    detail = str(last_exc).strip()
    reason = last_exc.__class__.__name__
    if detail:
        reason = f"{reason}: {detail[:180]}"
    return CleanCutProviderResult(
        (),
        ProviderStatus(
            provider=provider.__class__.__name__,
            requested=True,
            available=False,
            status="provider_error",
            reason=reason,
        ),
    )


def _candidate_from_words(parent: CandidateTake, words) -> CandidateTake:
    word_tuple = tuple(words)
    if not word_tuple:
        raise ValueError("cannot build candidate from empty word range")
    text = " ".join(word.text.strip() for word in word_tuple if word.text.strip()).strip()
    if not text:
        raise ValueError("word range produced empty text")
    start = float(word_tuple[0].start)
    end = float(word_tuple[-1].end)
    signals = parent.signals
    if signals is not None:
        signals = replace(signals, start=start, end=end)
    return CandidateTake(
        clip_id=stable_clip_id(parent.source_asset_id, start, end, text),
        source_asset_id=parent.source_asset_id,
        source_order=parent.source_order,
        start=start,
        end=end,
        text=text,
        words=word_tuple,
        signals=signals,
        complete_idea=parent.complete_idea,
        # D-050D1: a mixed trim's kept child and discarded edge fragments
        # are physical word-boundary adjustments of ONE already-minted
        # recorded delivery, not a genuinely new semantic realization --
        # same reasoning as human_boundary_polish_v5's own fragment
        # splitting, which carries realization_id/attempt_id forward via
        # dataclasses.replace(). This bare CandidateTake(...) constructor
        # used to drop both silently; now explicit. `source_span_id` is
        # deliberately NOT carried forward -- it is a NEW physical span
        # (different start/end than the parent's), and re-minting it is
        # out of this directive's scope.
        attempt_id=parent.attempt_id,
        realization_id=parent.realization_id,
    )


def _safe_mixed_trim(
    take: CandidateTake,
    judgement: CleanCutJudgement,
    *,
    mixed_threshold: float,
    min_keep_words: int,
    min_keep_duration_sec: float,
) -> tuple[CandidateTake | None, tuple[CandidateTake, ...]]:
    """Return a word-snapped kept child and discarded edge fragments, or fail open."""
    if judgement.action != "mixed" or judgement.confidence < mixed_threshold:
        return None, ()
    if judgement.keep_start_word_index is None or judgement.keep_end_word_index is None:
        return None, ()
    words = tuple(take.words)
    if not words:
        return None, ()

    start_index = judgement.keep_start_word_index
    end_index = judgement.keep_end_word_index
    if start_index < 0 or end_index < start_index or end_index >= len(words):
        return None, ()
    if start_index == 0 and end_index == len(words) - 1:
        return None, ()

    kept_words = words[start_index : end_index + 1]
    if len(kept_words) < min_keep_words:
        return None, ()
    kept = _candidate_from_words(take, kept_words)
    if kept.duration_sec < min_keep_duration_sec:
        return None, ()

    discarded = []
    prefix = words[:start_index]
    suffix = words[end_index + 1 :]
    if prefix:
        discarded.append(_candidate_from_words(take, prefix))
    if suffix:
        discarded.append(_candidate_from_words(take, suffix))
    if not discarded:
        return None, ()
    return kept, tuple(discarded)


def apply_provider_judgements(
    takes: Tuple[CandidateTake, ...],
    result: CleanCutProviderResult,
    *,
    delete_threshold: float = 0.94,
    mixed_threshold: float = 0.97,
    min_keep_words: int = 2,
    min_keep_duration_sec: float = 0.35,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Apply only high-confidence deletes/trims; all malformed or uncertain cases keep."""
    effective_delete_threshold = delete_threshold
    judgement_by_id = {item.clip_id: item for item in result.judgements}
    kept, deleted, diagnostics = [], [], []
    for take in takes:
        item = judgement_by_id.get(take.clip_id)
        applied_delete = bool(
            item is not None
            and item.action == "delete"
            and item.confidence >= effective_delete_threshold
        )
        applied_mixed_trim = False
        kept_child = None
        discarded_children: tuple[CandidateTake, ...] = ()

        if applied_delete:
            deleted.append(take)
        elif item is not None:
            kept_child, discarded_children = _safe_mixed_trim(
                take,
                item,
                mixed_threshold=mixed_threshold,
                min_keep_words=min_keep_words,
                min_keep_duration_sec=min_keep_duration_sec,
            )
            if kept_child is not None:
                applied_mixed_trim = True
                kept.append(kept_child)
                deleted.extend(discarded_children)
            else:
                kept.append(take)
        else:
            kept.append(take)

        if item is not None:
            diagnostics.append({
                "clip_id": take.clip_id,
                "action": item.action,
                "confidence": item.confidence,
                "reason": item.reason,
                "delete_threshold": effective_delete_threshold,
                "applied_delete": applied_delete,
                "applied_mixed_trim": applied_mixed_trim,
                "keep_start_word_index": item.keep_start_word_index,
                "keep_end_word_index": item.keep_end_word_index,
                "kept_clip_id": kept_child.clip_id if kept_child is not None else None,
                "discarded_clip_ids": [child.clip_id for child in discarded_children],
            })
    return tuple(kept), tuple(deleted), tuple(diagnostics)
