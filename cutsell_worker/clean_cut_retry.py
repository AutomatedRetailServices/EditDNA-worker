"""One-shot retry wrapper for transient Clean Cut provider contract errors."""
from __future__ import annotations

from dataclasses import dataclass

from .clean_cut_provider import CleanCutProvider, CleanCutProviderResult
from .contracts import CandidateTake

_REPAIRABLE_MESSAGES = frozenset({
    "clean cut judge returned invalid target id",
    "clean cut judge omitted ambiguous microtake",
})


@dataclass
class OneShotCleanCutContractRetry:
    """Retry once when a provider omits/duplicates/misidentifies target candidates.

    The wrapped provider performs the same strict validation on the second response.
    No IDs, actions, or confidences are synthesized locally. If the second response
    is still invalid, its exception propagates and the normal fail-open/provider
    health path records the failure.
    """

    provider: CleanCutProvider

    def judge(self, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
        try:
            return self.provider.judge(takes)
        except ValueError as exc:
            if str(exc) not in _REPAIRABLE_MESSAGES:
                raise
            return self.provider.judge(takes)


def install_clean_cut_contract_retry() -> None:
    """Wrap the package's OpenAI Clean Cut constructor once for all runtime paths."""
    from . import clean_cut_openai

    original = clean_cut_openai.OpenAICleanCutProvider
    if getattr(original, "_cutsell_contract_retry_installed", False):
        return

    def reliable_openai_clean_cut_provider(*args, **kwargs):
        return OneShotCleanCutContractRetry(original(*args, **kwargs))

    reliable_openai_clean_cut_provider._cutsell_contract_retry_installed = True
    clean_cut_openai.OpenAICleanCutProvider = reliable_openai_clean_cut_provider
