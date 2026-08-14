"""Strict recovery and structural safety wrapper for Clean Cut provider output."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clean_cut_provider import CleanCutJudgement, CleanCutProvider, CleanCutProviderResult
from .contracts import CandidateTake
from .providers import ProviderStatus
from .take_grouping import retry_similarity, semantic_key

_REPAIRABLE_MESSAGES = frozenset({
    "clean cut judge returned invalid target id",
    "clean cut judge omitted ambiguous microtake",
})
_BATCH_SIZE = 12
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_SELF_CRITIQUE_PATTERNS = (
    re.compile(r"\b(?:do you know\s+)?i (?:do not|don't|dont) like (?:the |this |that )?(?:beginning|start|take|one)\b", re.IGNORECASE),
    re.compile(r"\b(?:that|this) (?:was|is|sounded|sounds|looked|looks) (?:bad|wrong|weird|awkward)\b", re.IGNORECASE),
    re.compile(r"\bi (?:messed|screwed) (?:that|this|it) up\b", re.IGNORECASE),
)


def _validate_exact_ids(result: CleanCutProviderResult, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
    expected = {take.clip_id for take in takes}
    seen: set[str] = set()
    for item in result.judgements:
        if item.clip_id not in expected or item.clip_id in seen:
            raise ValueError("clean cut recovery returned invalid clip id")
        seen.add(item.clip_id)
    if seen != expected:
        raise ValueError("clean cut recovery omitted candidates")
    if result.status.status != "applied":
        raise ValueError("clean cut recovery provider status not applied")
    return result


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(str(text or "")))


def _collapse_internal_restart(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if len(tokens) < 2:
        return tokens
    for width in (4, 3, 2):
        for index in range(0, len(tokens) - (2 * width) + 1):
            first = tokens[index:index + width]
            second = tokens[index + width:index + (2 * width)]
            if first == second:
                return tokens[:index] + first + tokens[index + (2 * width):]
    for index in range(len(tokens) - 2):
        if tokens[index] == tokens[index + 1] == tokens[index + 2]:
            end = index + 3
            while end < len(tokens) and tokens[end] == tokens[index]:
                end += 1
            return tokens[:index + 1] + tokens[end:]
    return tokens


def _has_internal_restart_pattern(text: str) -> bool:
    tokens = _tokens(text)
    return len(tokens) >= 3 and _collapse_internal_restart(tokens) != tokens


def _initial_repeated_phrase_trim(take: CandidateTake) -> tuple[int, int] | None:
    """Return a safe kept-word span when a multi-word prefix is repeated once."""
    tokens = _tokens(take.text)
    words = tuple(take.words)
    if not words or len(words) != len(tokens):
        return None
    for width in (4, 3, 2):
        if len(tokens) <= 2 * width:
            continue
        if tokens[:width] == tokens[width:2 * width]:
            return width, len(words) - 1
    return None


def _token_retry_match(left_tokens: tuple[str, ...], right_tokens: tuple[str, ...]) -> bool:
    if not left_tokens or not right_tokens:
        return False
    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if longer[:len(shorter)] == shorter:
        return True
    shorter_set = set(shorter)
    return bool(shorter_set) and len(shorter_set.intersection(longer)) / len(shorter_set) >= 0.80


def _strong_retry_match(left: CandidateTake, right: CandidateTake) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    left_tokens = tuple(semantic_key(left.text).split())
    right_tokens = tuple(semantic_key(right.text).split())
    if not left_tokens or not right_tokens:
        return False
    if _token_retry_match(left_tokens, right_tokens):
        return True
    collapsed_left = _collapse_internal_restart(left_tokens)
    collapsed_right = _collapse_internal_restart(right_tokens)
    if _token_retry_match(collapsed_left, collapsed_right):
        return True
    return retry_similarity(" ".join(collapsed_left), " ".join(collapsed_right)) >= 0.80


def _gap_between(left: CandidateTake, right: CandidateTake) -> float:
    if right.end <= left.start:
        return left.start - right.end
    if right.start >= left.end:
        return right.start - left.end
    return 0.0


def _nearby_retry(take: CandidateTake, takes: tuple[CandidateTake, ...], *, maximum_gap_sec: float = 6.0) -> bool:
    return any(
        other.clip_id != take.clip_id
        and other.source_asset_id == take.source_asset_id
        and _gap_between(take, other) <= maximum_gap_sec
        and _strong_retry_match(take, other)
        for other in takes
    )


def _self_critique_prefix(text: str) -> tuple[str, ...]:
    value = str(text or "")
    matches = [pattern.search(value) for pattern in _SELF_CRITIQUE_PATTERNS]
    matches = [match for match in matches if match is not None]
    if not matches:
        return ()
    match = min(matches, key=lambda item: item.start())
    return _tokens(value[:match.start()])


def _nearby_prefix_retry(
    take: CandidateTake,
    takes: tuple[CandidateTake, ...],
    prefix: tuple[str, ...],
    *,
    maximum_gap_sec: float = 15.0,
) -> bool:
    if len(prefix) < 2:
        return False
    for other in takes:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        if _gap_between(take, other) > maximum_gap_sec:
            continue
        other_tokens = _tokens(other.text)
        if len(other_tokens) >= len(prefix) and other_tokens[:len(prefix)] == prefix:
            return True
    return False


def _apply_structural_corroboration(result: CleanCutProviderResult, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
    by_id = {item.clip_id: item for item in result.judgements}
    changed = 0
    output: list[CleanCutJudgement] = []

    for take in takes:
        item = by_id[take.clip_id]

        trim = _initial_repeated_phrase_trim(take)
        if trim is not None:
            output.append(CleanCutJudgement(
                clip_id=take.clip_id,
                action="mixed",
                confidence=max(0.99, float(item.confidence)),
                reason="repeated_prefix_trim",
                keep_start_word_index=trim[0],
                keep_end_word_index=trim[1],
            ))
            changed += 1
            continue

        prefix = _self_critique_prefix(take.text)
        if prefix and _nearby_prefix_retry(take, takes, prefix):
            output.append(CleanCutJudgement(
                clip_id=take.clip_id,
                action="delete",
                confidence=max(0.96, float(item.confidence)),
                reason="recording_self_critique_with_retry",
                keep_start_word_index=None,
                keep_end_word_index=None,
            ))
            changed += 1
            continue

        if _has_internal_restart_pattern(take.text) and _nearby_retry(take, takes):
            output.append(CleanCutJudgement(
                clip_id=take.clip_id,
                action="delete",
                confidence=max(0.96, float(item.confidence)),
                reason="structural_retry_corroborated",
                keep_start_word_index=None,
                keep_end_word_index=None,
            ))
            changed += 1
        else:
            output.append(item)

    if not changed:
        return result
    status = result.status
    reason = str(status.reason or "").strip()
    reason = f"{reason},structural_corroboration:{changed}".strip(",")
    return CleanCutProviderResult(
        tuple(output),
        ProviderStatus(
            provider=status.provider,
            requested=status.requested,
            available=status.available,
            status=status.status,
            reason=reason,
        ),
    )


@dataclass
class OneShotCleanCutContractRetry:
    provider: CleanCutProvider

    def _judge_batched(self, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
        recovered: dict[str, CleanCutJudgement] = {}
        statuses: list[ProviderStatus] = []
        for core_start in range(0, len(takes), _BATCH_SIZE):
            core_end = min(len(takes), core_start + _BATCH_SIZE)
            context_start = max(0, core_start - 1)
            context_end = min(len(takes), core_end + 1)
            context = tuple(takes[context_start:context_end])
            result = _validate_exact_ids(self.provider.judge(context), context)
            statuses.append(result.status)
            core_ids = {take.clip_id for take in takes[core_start:core_end]}
            for item in result.judgements:
                if item.clip_id not in core_ids:
                    continue
                if item.clip_id in recovered:
                    raise ValueError("clean cut recovery duplicated candidate")
                recovered[item.clip_id] = item
        expected = {take.clip_id for take in takes}
        if set(recovered) != expected:
            raise ValueError("clean cut recovery omitted candidates")
        first = statuses[0]
        reason = f"{str(first.reason or '').strip()},batched_contract_recovery".strip(",")
        return CleanCutProviderResult(
            tuple(recovered[take.clip_id] for take in takes),
            ProviderStatus(first.provider, True, True, "applied", reason),
        )

    def judge(self, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
        try:
            result = self.provider.judge(takes)
        except ValueError as exc:
            if str(exc) not in _REPAIRABLE_MESSAGES or len(takes) <= _BATCH_SIZE:
                raise
            result = self._judge_batched(takes)
        return _apply_structural_corroboration(result, takes)


def install_clean_cut_contract_recovery() -> None:
    from . import clean_cut_openai
    original = clean_cut_openai.OpenAICleanCutProvider
    if getattr(original, "_cutsell_contract_recovery", False):
        return

    def build_reliable_provider(*args, **kwargs):
        return OneShotCleanCutContractRetry(original(*args, **kwargs))

    build_reliable_provider._cutsell_contract_recovery = True
    clean_cut_openai.OpenAICleanCutProvider = build_reliable_provider
