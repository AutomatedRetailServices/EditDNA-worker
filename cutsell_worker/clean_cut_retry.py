"""Strict recovery wrapper for transient Clean Cut provider contract errors."""
from __future__ import annotations

from dataclasses import dataclass

from .clean_cut_provider import CleanCutJudgement, CleanCutProvider, CleanCutProviderResult
from .contracts import CandidateTake
from .providers import ProviderStatus

_REPAIRABLE_MESSAGES = frozenset({
    "clean cut judge returned invalid target id",
    "clean cut judge omitted ambiguous microtake",
})
_BATCH_SIZE = 12


def _validate_exact_ids(
    result: CleanCutProviderResult,
    takes: tuple[CandidateTake, ...],
) -> CleanCutProviderResult:
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


@dataclass
class OneShotCleanCutContractRetry:
    """Recover provider identity/omission failures without weakening edit safety.

    Normal calls are unchanged. If the selective provider returns a repairable
    contract error, retry once using smaller contiguous batches. Each batch keeps
    one neighboring take on both sides as read-only local context, every response
    must cover the exact context ids supplied, and only the core batch judgements
    are collected. No ids, actions, confidences, or edit decisions are invented.
    """

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
        reason = str(first.reason or "").strip()
        reason = f"{reason},batched_contract_recovery".strip(",")
        return CleanCutProviderResult(
            tuple(recovered[take.clip_id] for take in takes),
            ProviderStatus(
                provider=first.provider,
                requested=True,
                available=True,
                status="applied",
                reason=reason,
            ),
        )

    def judge(self, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
        try:
            return self.provider.judge(takes)
        except ValueError as exc:
            if str(exc) not in _REPAIRABLE_MESSAGES or len(takes) <= _BATCH_SIZE:
                raise
            return self._judge_batched(takes)


def install_clean_cut_contract_recovery() -> None:
    """Install the recovery constructor once before runtime modules import it."""
    from . import clean_cut_openai

    original = clean_cut_openai.OpenAICleanCutProvider
    if getattr(original, "_cutsell_contract_recovery", False):
        return

    def build_reliable_provider(*args, **kwargs):
        return OneShotCleanCutContractRetry(original(*args, **kwargs))

    build_reliable_provider._cutsell_contract_recovery = True
    clean_cut_openai.OpenAICleanCutProvider = build_reliable_provider
